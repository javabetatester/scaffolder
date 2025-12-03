package scaffolder

var restTemplates = []FileTemplate{
	{
		Path: "go.mod",
		Content: `module github.com/{{.ModuleName}}

go 1.21

require (
	github.com/gin-gonic/gin v1.9.1
	github.com/gin-contrib/cors v1.5.0
	github.com/go-playground/validator/v10 v10.16.0
	go.uber.org/zap v1.26.0
	github.com/prometheus/client_golang v1.18.0
	go.opentelemetry.io/otel v1.21.0
	go.opentelemetry.io/otel/trace v1.21.0
	go.opentelemetry.io/otel/exporters/jaeger v1.17.0
	go.opentelemetry.io/otel/sdk v1.21.0
	golang.org/x/time v0.5.0
	github.com/stretchr/testify v1.8.4
	gorm.io/gorm v1.25.5
	gorm.io/driver/postgres v1.5.4
	github.com/redis/go-redis/v9 v9.3.0
	github.com/golang-migrate/migrate/v4 v4.16.2
)
`,
		Permissions: 0644,
	},
	{
		Path: "cmd/server/main.go",
		Content: `package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/{{.ModuleName}}/internal/infrastructure/cache"
	"github.com/{{.ModuleName}}/internal/infrastructure/config"
	"github.com/{{.ModuleName}}/internal/infrastructure/database"
	"github.com/{{.ModuleName}}/internal/infrastructure/logger"
	"github.com/{{.ModuleName}}/internal/infrastructure/metrics"
	"github.com/{{.ModuleName}}/internal/infrastructure/router"
	"github.com/{{.ModuleName}}/internal/infrastructure/tracing"
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

func main() {
	cfg := config.Load()

	log := logger.New(cfg)
	defer log.Sync()

	tracer, err := tracing.NewTracer(cfg, log)
	if err != nil {
		log.Fatal("failed to initialize tracer", zap.Error(err))
	}
	defer tracer.Shutdown(context.Background())

	metricsServer := metrics.NewServer(cfg, log)
	go func() {
		if err := metricsServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatal("failed to start metrics server", zap.Error(err))
		}
	}()

	db, err := database.NewDB(cfg, log)
	if err != nil {
		log.Fatal("failed to initialize database", zap.Error(err))
	}

	redisCache, err := cache.NewCache(cfg, log)
	if err != nil {
		log.Fatal("failed to initialize cache", zap.Error(err))
	}
	defer redisCache.Close()

	if cfg.Environment == "production" {
		gin.SetMode(gin.ReleaseMode)
	}

	r := router.New(cfg, log, tracer, db, redisCache)

	srv := &http.Server{
		Addr:    fmt.Sprintf(":%d", cfg.ServerPort),
		Handler: r,
	}

	go func() {
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatal("failed to start server", zap.Error(err))
		}
	}()

	log.Info("server started", zap.Int("port", cfg.ServerPort), zap.Int("metrics_port", cfg.MetricsPort))

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Info("shutting down server...")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		log.Fatal("server forced to shutdown", zap.Error(err))
	}

	log.Info("server exited")
}
`,
		Permissions: 0644,
	},
	{
		Path: "internal/infrastructure/config/config.go",
		Content: `package config

import (
	"os"
	"strconv"
)

type Config struct {
	Environment      string
	ServerPort        int
	MetricsPort       int
	JaegerEndpoint    string
	LogLevel          string
	DatabaseURL       string
	DatabaseMaxConns  int
	RedisURL          string
	RedisPassword     string
	RedisDB           int
}

func Load() *Config {
	port := 8080
	if portStr := os.Getenv("PORT"); portStr != "" {
		if p, err := strconv.Atoi(portStr); err == nil {
			port = p
		}
	}

	metricsPort := 9090
	if portStr := os.Getenv("METRICS_PORT"); portStr != "" {
		if p, err := strconv.Atoi(portStr); err == nil {
			metricsPort = p
		}
	}

	env := os.Getenv("ENVIRONMENT")
	if env == "" {
		env = "development"
	}

	logLevel := os.Getenv("LOG_LEVEL")
	if logLevel == "" {
		logLevel = "info"
	}

	jaegerEndpoint := os.Getenv("JAEGER_ENDPOINT")
	if jaegerEndpoint == "" {
		jaegerEndpoint = "http://localhost:14268/api/traces"
	}

	databaseURL := os.Getenv("DATABASE_URL")
	if databaseURL == "" {
		databaseURL = "postgres://user:password@localhost:5432/{{.PackageName}}?sslmode=disable"
	}

	maxConns := 25
	if maxConnsStr := os.Getenv("DATABASE_MAX_CONNS"); maxConnsStr != "" {
		if mc, err := strconv.Atoi(maxConnsStr); err == nil {
			maxConns = mc
		}
	}

	redisURL := os.Getenv("REDIS_URL")
	if redisURL == "" {
		redisURL = "localhost:6379"
	}

	redisPassword := os.Getenv("REDIS_PASSWORD")

	redisDB := 0
	if redisDBStr := os.Getenv("REDIS_DB"); redisDBStr != "" {
		if db, err := strconv.Atoi(redisDBStr); err == nil {
			redisDB = db
		}
	}

	return &Config{
		Environment:     env,
		ServerPort:       port,
		MetricsPort:      metricsPort,
		JaegerEndpoint:   jaegerEndpoint,
		LogLevel:         logLevel,
		DatabaseURL:      databaseURL,
		DatabaseMaxConns: maxConns,
		RedisURL:         redisURL,
		RedisPassword:    redisPassword,
		RedisDB:          redisDB,
	}
}
`,
		Permissions: 0644,
	},
	{
		Path: "internal/infrastructure/router/router.go",
		Content: `package router

import (
	"github.com/{{.ModuleName}}/internal/infrastructure/cache"
	"github.com/{{.ModuleName}}/internal/infrastructure/config"
	"github.com/{{.ModuleName}}/internal/infrastructure/database"
	"github.com/{{.ModuleName}}/internal/infrastructure/middleware"
	"github.com/{{.ModuleName}}/internal/interface/http/handler"
	"github.com/gin-gonic/gin"
	"go.opentelemetry.io/otel/trace"
	"go.uber.org/zap"
	"gorm.io/gorm"
)

func New(cfg *config.Config, log *zap.Logger, tracer trace.TracerProvider, db *gorm.DB, cache *cache.Cache) *gin.Engine {
	r := gin.New()

	r.Use(middleware.Logger(log))
	r.Use(middleware.Recovery(log))
	r.Use(middleware.SecurityHeaders())
	r.Use(middleware.CORS(cfg))
	r.Use(middleware.RateLimit())
	r.Use(middleware.Metrics())
	r.Use(middleware.Tracing(tracer))

	api := r.Group("/api/v1")
	{
		healthHandler := handler.NewHealthHandler(log, db, cache)
		api.GET("/health", healthHandler.Check)
		api.GET("/metrics", middleware.PrometheusHandler())
	}

	return r
}
`,
		Permissions: 0644,
	},
	{
		Path: "internal/interface/http/handler/health_handler.go",
		Content: `package handler

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

type HealthHandler struct{}

func NewHealthHandler() *HealthHandler {
	return &HealthHandler{}
}

func (h *HealthHandler) Check(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status": "ok",
	})
}
`,
		Permissions: 0644,
	},
	{
		Path: "internal/domain/entity/entity.go",
		Content: `package entity

type Entity struct {
	ID string
}
`,
		Permissions: 0644,
	},
	{
		Path: "internal/domain/repository/repository.go",
		Content: `package repository

import "github.com/{{.ModuleName}}/internal/domain/entity"

type Repository interface {
	FindByID(id string) (*entity.Entity, error)
	Create(e *entity.Entity) error
	Update(e *entity.Entity) error
	Delete(id string) error
}
`,
		Permissions: 0644,
	},
	{
		Path: "internal/usecase/usecase.go",
		Content: `package usecase

import "github.com/{{.ModuleName}}/internal/domain/repository"

type UseCase struct {
	repo repository.Repository
}

func NewUseCase(repo repository.Repository) *UseCase {
	return &UseCase{
		repo: repo,
	}
}
`,
		Permissions: 0644,
	},
	{
		Path: "internal/infrastructure/repository/memory_repository.go",
		Content: `package repository

import (
	"fmt"
	"sync"

	"github.com/{{.ModuleName}}/internal/domain/entity"
	domainRepo "github.com/{{.ModuleName}}/internal/domain/repository"
)

type MemoryRepository struct {
	mu   sync.RWMutex
	data map[string]*entity.Entity
}

func NewMemoryRepository() domainRepo.Repository {
	return &MemoryRepository{
		data: make(map[string]*entity.Entity),
	}
}

func (r *MemoryRepository) FindByID(id string) (*entity.Entity, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	e, exists := r.data[id]
	if !exists {
		return nil, fmt.Errorf("entity not found: %s", id)
	}

	return e, nil
}

func (r *MemoryRepository) Create(e *entity.Entity) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.data[e.ID]; exists {
		return fmt.Errorf("entity already exists: %s", e.ID)
	}

	r.data[e.ID] = e
	return nil
}

func (r *MemoryRepository) Update(e *entity.Entity) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.data[e.ID]; !exists {
		return fmt.Errorf("entity not found: %s", e.ID)
	}

	r.data[e.ID] = e
	return nil
}

func (r *MemoryRepository) Delete(id string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.data[id]; !exists {
		return fmt.Errorf("entity not found: %s", id)
	}

	delete(r.data, id)
	return nil
}
`,
		Permissions: 0644,
	},
	{
		Path: "internal/infrastructure/logger/logger.go",
		Content: `package logger

import (
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"

	"github.com/{{.ModuleName}}/internal/infrastructure/config"
)

func New(cfg *config.Config) *zap.Logger {
	var logger *zap.Logger
	var err error

	if cfg.Environment == "production" {
		config := zap.NewProductionConfig()
		config.Level = parseLogLevel(cfg.LogLevel)
		logger, err = config.Build()
	} else {
		config := zap.NewDevelopmentConfig()
		config.Level = parseLogLevel(cfg.LogLevel)
		logger, err = config.Build()
	}

	if err != nil {
		panic(err)
	}

	return logger.With(
		zap.String("service", "{{.PackageName}}"),
		zap.String("environment", cfg.Environment),
	)
}

func parseLogLevel(level string) zapcore.Level {
	switch level {
	case "debug":
		return zapcore.DebugLevel
	case "info":
		return zapcore.InfoLevel
	case "warn":
		return zapcore.WarnLevel
	case "error":
		return zapcore.ErrorLevel
	default:
		return zapcore.InfoLevel
	}
}
`,
		Permissions: 0644,
	},
	{
		Path: "internal/infrastructure/metrics/metrics.go",
		Content: `package metrics

import (
	"fmt"
	"net/http"

	"github.com/{{.ModuleName}}/internal/infrastructure/config"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.uber.org/zap"
)

var (
	HTTPRequestsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "http_requests_total",
			Help: "Total number of HTTP requests",
		},
		[]string{"method", "endpoint", "status"},
	)

	HTTPRequestDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "http_request_duration_seconds",
			Help:    "HTTP request duration in seconds",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"method", "endpoint"},
	)
)

func init() {
	prometheus.MustRegister(HTTPRequestsTotal)
	prometheus.MustRegister(HTTPRequestDuration)
}

type Server struct {
	*http.Server
}

func NewServer(cfg *config.Config, log *zap.Logger) *Server {
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.Handler())

	return &Server{
		Server: &http.Server{
			Addr:    fmt.Sprintf(":%d", cfg.MetricsPort),
			Handler: mux,
		},
	}
}
`,
		Permissions: 0644,
	},
	{
		Path: "internal/infrastructure/tracing/tracing.go",
		Content: `package tracing

import (
	"context"

	"github.com/{{.ModuleName}}/internal/infrastructure/config"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/jaeger"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	tracesdk "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.21.0"
	"go.opentelemetry.io/otel/trace"
	"go.uber.org/zap"
)

type Tracer struct {
	tp *tracesdk.TracerProvider
}

func NewTracer(cfg *config.Config, log *zap.Logger) (trace.TracerProvider, error) {
	exp, err := jaeger.New(jaeger.WithCollectorEndpoint(jaeger.WithEndpoint(cfg.JaegerEndpoint)))
	if err != nil {
		return nil, err
	}

	tp := tracesdk.NewTracerProvider(
		tracesdk.WithBatcher(exp),
		tracesdk.WithResource(resource.NewWithAttributes(
			semconv.SchemaURL,
			semconv.ServiceNameKey.String("{{.PackageName}}"),
			semconv.ServiceVersionKey.String("1.0.0"),
		)),
	)

	otel.SetTracerProvider(tp)
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{},
		propagation.Baggage{},
	))

	return &Tracer{tp: tp}, nil
}

func (t *Tracer) Shutdown(ctx context.Context) error {
	return t.tp.Shutdown(ctx)
}
`,
		Permissions: 0644,
	},
	{
		Path: "internal/infrastructure/middleware/middleware.go",
		Content: `package middleware

import (
	"time"

	"github.com/{{.ModuleName}}/internal/infrastructure/config"
	"github.com/{{.ModuleName}}/internal/infrastructure/metrics"
	"github.com/gin-gonic/gin"
	"golang.org/x/time/rate"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/trace"
	"go.uber.org/zap"
)

func Logger(log *zap.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		path := c.Request.URL.Path
		query := c.Request.URL.RawQuery

		c.Next()

		latency := time.Since(start)

		log.Info("request",
			zap.Int("status", c.Writer.Status()),
			zap.String("method", c.Request.Method),
			zap.String("path", path),
			zap.String("query", query),
			zap.String("ip", c.ClientIP()),
			zap.String("user-agent", c.Request.UserAgent()),
			zap.Duration("latency", latency),
		)
	}
}

func Recovery(log *zap.Logger) gin.HandlerFunc {
	return gin.CustomRecoveryWithWriter(nil, func(c *gin.Context, recovered interface{}) {
		log.Error("panic recovered",
			zap.Any("error", recovered),
			zap.String("path", c.Request.URL.Path),
		)
		c.AbortWithStatus(500)
	})
}

func Metrics() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		method := c.Request.Method
		path := c.Request.URL.Path

		c.Next()

		status := c.Writer.Status()
		duration := time.Since(start).Seconds()

		metrics.HTTPRequestsTotal.WithLabelValues(method, path, fmt.Sprintf("%d", status)).Inc()
		metrics.HTTPRequestDuration.WithLabelValues(method, path).Observe(duration)
	}
}

func Tracing(tp trace.TracerProvider) gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx := otel.GetTextMapPropagator().Extract(c.Request.Context(), propagation.HeaderCarrier(c.Request.Header))
		tr := tp.Tracer("{{.PackageName}}")
		ctx, span := tr.Start(ctx, c.Request.Method+" "+c.Request.URL.Path)
		defer span.End()

		c.Request = c.Request.WithContext(ctx)

		c.Next()

		span.SetAttributes(
			attribute.String("http.method", c.Request.Method),
			attribute.String("http.path", c.Request.URL.Path),
			attribute.Int("http.status", c.Writer.Status()),
		)
	}
}

func PrometheusHandler() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Redirect(302, "/metrics")
	}
}

func SecurityHeaders() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Header("X-Content-Type-Options", "nosniff")
		c.Header("X-Frame-Options", "DENY")
		c.Header("X-XSS-Protection", "1; mode=block")
		c.Header("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
		c.Header("Content-Security-Policy", "default-src 'self'")
		c.Next()
	}
}

func CORS(cfg *config.Config) gin.HandlerFunc {
	return func(c *gin.Context) {
		origin := c.GetHeader("Origin")
		allowedOrigins := []string{"*"}
		
		if cfg.Environment == "production" {
			allowedOrigins = []string{"https://example.com"}
		}

		allowed := false
		for _, o := range allowedOrigins {
			if o == "*" || o == origin {
				allowed = true
				break
			}
		}

		if allowed {
			c.Header("Access-Control-Allow-Origin", origin)
			c.Header("Access-Control-Allow-Credentials", "true")
			c.Header("Access-Control-Allow-Headers", "Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization, accept, origin, Cache-Control, X-Requested-With")
			c.Header("Access-Control-Allow-Methods", "POST, OPTIONS, GET, PUT, DELETE, PATCH")
		}

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}

		c.Next()
	}
}

func RateLimit() gin.HandlerFunc {
	limiter := rate.NewLimiter(rate.Limit(100), 200)
	return func(c *gin.Context) {
		if !limiter.Allow() {
			c.JSON(429, gin.H{"error": "too many requests"})
			c.Abort()
			return
		}
		c.Next()
	}
}
`,
		Permissions: 0644,
	},
	{
		Path: "internal/interface/http/handler/health_handler.go",
		Content: `package handler

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

type HealthHandler struct {
	log    *zap.Logger
	db     interface{}
	cache  interface{}
}

func NewHealthHandler(log *zap.Logger, db interface{}, cache interface{}) *HealthHandler {
	return &HealthHandler{
		log:   log,
		db:    db,
		cache: cache,
	}
}

func (h *HealthHandler) Check(c *gin.Context) {
	h.log.Debug("health check requested")
	
	status := "ok"
	
	if h.db != nil {
		if db, ok := h.db.(interface{ Ping() error }); ok {
			if err := db.Ping(); err != nil {
				status = "degraded"
				h.log.Warn("database health check failed", zap.Error(err))
			}
		}
	}
	
	c.JSON(http.StatusOK, gin.H{
		"status": status,
	})
}
`,
		Permissions: 0644,
	},
	{
		Path: "internal/pkg/validation/validation.go",
		Content: `package validation

import (
	"fmt"
	"strings"

	"github.com/go-playground/validator/v10"
	"github.com/gin-gonic/gin"
)

var validate *validator.Validate

func init() {
	validate = validator.New(validator.WithRequiredStructEnabled())
}

func ValidateStruct(s interface{}) error {
	return validate.Struct(s)
}

func ValidateRequest(c *gin.Context, req interface{}) error {
	if err := c.ShouldBindJSON(req); err != nil {
		return err
	}

	if err := ValidateStruct(req); err != nil {
		validationErrors := err.(validator.ValidationErrors)
		var errors []string
		for _, e := range validationErrors {
			errors = append(errors, fmt.Sprintf("%s: %s", e.Field(), e.Tag()))
		}
		return fmt.Errorf("validation failed: %s", strings.Join(errors, ", "))
	}

	return nil
}
`,
		Permissions: 0644,
	},
	{
		Path: "internal/interface/http/handler/health_handler_test.go",
		Content: `package handler

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"go.uber.org/zap"
)

func TestHealthHandler_Check(t *testing.T) {
	gin.SetMode(gin.TestMode)
	log := zap.NewNop()

	handler := NewHealthHandler(log)

	tests := []struct {
		name           string
		expectedStatus int
		expectedBody   string
	}{
		{
			name:           "should return ok status",
			expectedStatus: http.StatusOK,
			expectedBody:   "{\"status\":\"ok\"}",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)
			c.Request = httptest.NewRequest("GET", "/health", nil)

			handler.Check(c)

			assert.Equal(t, tt.expectedStatus, w.Code)
			assert.JSONEq(t, tt.expectedBody, w.Body.String())
		})
	}
}
`,
		Permissions: 0644,
	},
	{
		Path: "internal/usecase/usecase_test.go",
		Content: `package usecase

import (
	"context"
	"testing"

	"github.com/{{.ModuleName}}/internal/domain/entity"
	"github.com/{{.ModuleName}}/internal/domain/repository"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

type MockRepository struct {
	mock.Mock
}

var _ repository.Repository = (*MockRepository)(nil)

func (m *MockRepository) FindByID(id string) (*entity.Entity, error) {
	args := m.Called(id)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*entity.Entity), args.Error(1)
}

func (m *MockRepository) Create(e *entity.Entity) error {
	args := m.Called(e)
	return args.Error(0)
}

func (m *MockRepository) Update(e *entity.Entity) error {
	args := m.Called(e)
	return args.Error(0)
}

func (m *MockRepository) Delete(id string) error {
	args := m.Called(id)
	return args.Error(0)
}

func TestUseCase_GetEntity(t *testing.T) {
	mockRepo := new(MockRepository)
	useCase := NewUseCase(mockRepo)

	ctx := context.Background()
	expectedEntity := &entity.Entity{ID: "123"}

	mockRepo.On("FindByID", "123").Return(expectedEntity, nil)

	result, err := useCase.repo.FindByID("123")

	assert.NoError(t, err)
	assert.Equal(t, expectedEntity, result)
	mockRepo.AssertExpectations(t)
}
`,
		Permissions: 0644,
	},
	{
		Path: "internal/infrastructure/repository/memory_repository_test.go",
		Content: `package repository

import (
	"testing"

	"github.com/{{.ModuleName}}/internal/domain/entity"
	"github.com/stretchr/testify/assert"
)

func TestMemoryRepository_Create(t *testing.T) {
	repo := NewMemoryRepository()

	e := &entity.Entity{ID: "123"}

	err := repo.Create(e)
	assert.NoError(t, err)

	err = repo.Create(e)
	assert.Error(t, err)
}

func TestMemoryRepository_FindByID(t *testing.T) {
	repo := NewMemoryRepository()

	e := &entity.Entity{ID: "123"}
	repo.Create(e)

	found, err := repo.FindByID("123")
	assert.NoError(t, err)
	assert.Equal(t, e, found)

	_, err = repo.FindByID("nonexistent")
	assert.Error(t, err)
}

func TestMemoryRepository_Update(t *testing.T) {
	repo := NewMemoryRepository()

	e := &entity.Entity{ID: "123"}
	repo.Create(e)

	err := repo.Update(e)
	assert.NoError(t, err)

	nonexistent := &entity.Entity{ID: "nonexistent"}
	err = repo.Update(nonexistent)
	assert.Error(t, err)
}

func TestMemoryRepository_Delete(t *testing.T) {
	repo := NewMemoryRepository()

	e := &entity.Entity{ID: "123"}
	repo.Create(e)

	err := repo.Delete("123")
	assert.NoError(t, err)

	_, err = repo.FindByID("123")
	assert.Error(t, err)

	err = repo.Delete("nonexistent")
	assert.Error(t, err)
}
`,
		Permissions: 0644,
	},
	{
		Path: "internal/pkg/validation/validation_test.go",
		Content: `package validation

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

type TestStruct struct {
	Name  string ` + "`validate:\"required,min=3,max=100\"`" + `
	Email string ` + "`validate:\"required,email\"`" + `
	Age   int    ` + "`validate:\"min=18,max=120\"`" + `
}

func TestValidateStruct(t *testing.T) {
	tests := []struct {
		name    string
		input   TestStruct
		wantErr bool
	}{
		{
			name: "valid struct",
			input: TestStruct{
				Name:  "John Doe",
				Email: "john@example.com",
				Age:   25,
			},
			wantErr: false,
		},
		{
			name: "invalid name too short",
			input: TestStruct{
				Name:  "Jo",
				Email: "john@example.com",
				Age:   25,
			},
			wantErr: true,
		},
		{
			name: "invalid email",
			input: TestStruct{
				Name:  "John Doe",
				Email: "invalid-email",
				Age:   25,
			},
			wantErr: true,
		},
		{
			name: "invalid age",
			input: TestStruct{
				Name:  "John Doe",
				Email: "john@example.com",
				Age:   10,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateStruct(tt.input)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}
`,
		Permissions: 0644,
	},
	{
		Path: "Makefile",
		Content: `BINARY_NAME={{.PackageName}}
CMD_PATH=cmd/server

.PHONY: build
build:
	go build -o bin/$(BINARY_NAME) $(CMD_PATH)/main.go

.PHONY: run
run:
	go run $(CMD_PATH)/main.go

.PHONY: test
test:
	go test -v ./...

.PHONY: test-coverage
test-coverage:
	go test -v -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out -o coverage.html

.PHONY: test-unit
test-unit:
	go test -v -short ./...

.PHONY: test-integration
test-integration:
	go test -v -tags=integration ./...

.PHONY: migrate-up
migrate-up:
	migrate -path migrations -database "$$DATABASE_URL" up

.PHONY: migrate-down
migrate-down:
	migrate -path migrations -database "$$DATABASE_URL" down

.PHONY: migrate-create
migrate-create:
	@read -p "Migration name: " name; \
	migrate create -ext sql -dir migrations -seq $$name

.PHONY: docker-build
docker-build:
	docker build -t {{.PackageName}}:latest .

.PHONY: docker-run
docker-run:
	docker run -p 8080:8080 {{.PackageName}}:latest

.PHONY: docker-compose-up
docker-compose-up:
	docker-compose up -d

.PHONY: docker-compose-down
docker-compose-down:
	docker-compose down

.PHONY: lint
lint:
	golangci-lint run

.PHONY: fmt
fmt:
	go fmt ./...

.PHONY: vet
vet:
	go vet ./...

.PHONY: clean
clean:
	rm -rf bin/
	rm -f coverage.out coverage.html
`,
		Permissions: 0644,
	},
	{
		Path: "internal/infrastructure/database/database.go",
		Content: `package database

import (
	"fmt"
	"time"

	"github.com/{{.ModuleName}}/internal/infrastructure/config"
	"go.uber.org/zap"
	"gorm.io/driver/postgres"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
)

func NewDB(cfg *config.Config, log *zap.Logger) (*gorm.DB, error) {
	db, err := gorm.Open(postgres.Open(cfg.DatabaseURL), &gorm.Config{
		Logger: logger.Default.LogMode(logger.Silent),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	sqlDB, err := db.DB()
	if err != nil {
		return nil, fmt.Errorf("failed to get sql.DB: %w", err)
	}

	sqlDB.SetMaxOpenConns(cfg.DatabaseMaxConns)
	sqlDB.SetMaxIdleConns(5)
	sqlDB.SetConnMaxLifetime(time.Hour)

	if err := sqlDB.Ping(); err != nil {
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	log.Info("database connected successfully")

	return db, nil
}
`,
		Permissions: 0644,
	},
	{
		Path: "internal/infrastructure/cache/cache.go",
		Content: `package cache

import (
	"context"
	"fmt"
	"time"

	"github.com/{{.ModuleName}}/internal/infrastructure/config"
	"github.com/redis/go-redis/v9"
	"go.uber.org/zap"
)

type Cache struct {
	client *redis.Client
}

func NewCache(cfg *config.Config, log *zap.Logger) (*Cache, error) {
	client := redis.NewClient(&redis.Options{
		Addr:     cfg.RedisURL,
		Password: cfg.RedisPassword,
		DB:       cfg.RedisDB,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := client.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("failed to connect to redis: %w", err)
	}

	log.Info("redis connected successfully")

	return &Cache{client: client}, nil
}

func (c *Cache) Get(ctx context.Context, key string) (string, error) {
	return c.client.Get(ctx, key).Result()
}

func (c *Cache) Set(ctx context.Context, key string, value interface{}, expiration time.Duration) error {
	return c.client.Set(ctx, key, value, expiration).Err()
}

func (c *Cache) Delete(ctx context.Context, key string) error {
	return c.client.Del(ctx, key).Err()
}

func (c *Cache) Close() error {
	return c.client.Close()
}
`,
		Permissions: 0644,
	},
	{
		Path: "internal/infrastructure/repository/postgres_repository.go",
		Content: `package repository

import (
	"context"
	"fmt"

	"github.com/{{.ModuleName}}/internal/domain/entity"
	domainRepo "github.com/{{.ModuleName}}/internal/domain/repository"
	"gorm.io/gorm"
)

type PostgresRepository struct {
	db *gorm.DB
}

func NewPostgresRepository(db *gorm.DB) domainRepo.Repository {
	return &PostgresRepository{db: db}
}

func (r *PostgresRepository) FindByID(id string) (*entity.Entity, error) {
	var e entity.Entity
	if err := r.db.First(&e, "id = ?", id).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			return nil, fmt.Errorf("entity not found: %s", id)
		}
		return nil, fmt.Errorf("failed to find entity: %w", err)
	}
	return &e, nil
}

func (r *PostgresRepository) Create(e *entity.Entity) error {
	if err := r.db.Create(e).Error; err != nil {
		return fmt.Errorf("failed to create entity: %w", err)
	}
	return nil
}

func (r *PostgresRepository) Update(e *entity.Entity) error {
	if err := r.db.Save(e).Error; err != nil {
		return fmt.Errorf("failed to update entity: %w", err)
	}
	return nil
}

func (r *PostgresRepository) Delete(id string) error {
	if err := r.db.Delete(&entity.Entity{}, "id = ?", id).Error; err != nil {
		return fmt.Errorf("failed to delete entity: %w", err)
	}
	return nil
}
`,
		Permissions: 0644,
	},
	{
		Path: "migrations/000001_create_entities.up.sql",
		Content: `CREATE TABLE IF NOT EXISTS entities (
    id VARCHAR(255) PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_entities_created_at ON entities(created_at);
`,
		Permissions: 0644,
	},
	{
		Path: "migrations/000001_create_entities.down.sql",
		Content: `DROP INDEX IF EXISTS idx_entities_created_at;
DROP TABLE IF EXISTS entities;
`,
		Permissions: 0644,
	},
	{
		Path: ".gitignore",
		Content: `bin/
dist/
*.exe
*.exe~
*.dll
*.so
*.dylib

*.test
*.out
coverage.txt

.idea/
.vscode/
*.swp
*.swo
*~

.env
.env.local
`,
		Permissions: 0644,
	},
	{
		Path: "Dockerfile",
		Content: `FROM golang:1.21-alpine AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o server ./cmd/server

FROM alpine:latest

RUN apk --no-cache add ca-certificates tzdata
WORKDIR /root/

COPY --from=builder /app/server .
COPY --from=builder /app/migrations ./migrations

EXPOSE 8080

CMD ["./server"]
`,
		Permissions: 0644,
	},
	{
		Path: ".dockerignore",
		Content: `bin/
dist/
*.exe
*.exe~
*.dll
*.so
*.dylib

*.test
*.out
coverage.txt
coverage.out
coverage.html

.idea/
.vscode/
*.swp
*.swo
*~

.env
.env.local

.git/
.gitignore
README.md
Makefile

*.md
`,
		Permissions: 0644,
	},
	{
		Path: "docker-compose.yml",
		Content: `version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: {{.PackageName}}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  app:
    build: .
    ports:
      - "8080:8080"
    environment:
      PORT: 8080
      ENVIRONMENT: development
      DATABASE_URL: postgres://user:password@postgres:5432/{{.PackageName}}?sslmode=disable
      REDIS_URL: redis:6379
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped

volumes:
  postgres_data:
`,
		Permissions: 0644,
	},
	{
		Path: "k8s/deployment.yaml",
		Content: `apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{.PackageName}}
  labels:
    app: {{.PackageName}}
spec:
  replicas: 3
  selector:
    matchLabels:
      app: {{.PackageName}}
  template:
    metadata:
      labels:
        app: {{.PackageName}}
    spec:
      containers:
      - name: {{.PackageName}}
        image: {{.PackageName}}:latest
        ports:
        - containerPort: 8080
        env:
        - name: PORT
          value: "8080"
        - name: ENVIRONMENT
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: {{.PackageName}}-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: {{.PackageName}}-config
              key: redis-url
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /api/v1/health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/v1/health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: {{.PackageName}}
spec:
  selector:
    app: {{.PackageName}}
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{.PackageName}}-config
data:
  redis-url: "redis-service:6379"
---
apiVersion: v1
kind: Secret
metadata:
  name: {{.PackageName}}-secrets
type: Opaque
stringData:
  database-url: "postgres://user:password@postgres-service:5432/{{.PackageName}}?sslmode=disable"
`,
		Permissions: 0644,
	},
	{
		Path: ".github/workflows/ci.yml",
		Content: `name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Set up Go
      uses: actions/setup-go@v5
      with:
        go-version: '1.21'

    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/go/pkg/mod
        key: ${{ runner.os }}-go-${{ hashFiles('**/go.sum') }}
        restore-keys: |
          ${{ runner.os }}-go-

    - name: Download dependencies
      run: go mod download

    - name: Run tests
      run: go test -v -coverprofile=coverage.out ./...

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.out

  lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Set up Go
      uses: actions/setup-go@v5
      with:
        go-version: '1.21'

    - name: Run golangci-lint
      uses: golangci/golangci-lint-action@v3
      with:
        version: latest

  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Set up Go
      uses: actions/setup-go@v5
      with:
        go-version: '1.21'

    - name: Build
      run: go build -o bin/server ./cmd/server

    - name: Build Docker image
      run: docker build -t {{.PackageName}}:${{ github.sha }} .
`,
		Permissions: 0644,
	},
	{
		Path: "CONTRIBUTING.md",
		Content: "# Contributing\n\n" +
			"Thank you for considering contributing to {{.ProjectName}}!\n\n" +
			"## Development Setup\n\n" +
			"1. Fork the repository\n" +
			"2. Clone your fork\n" +
			"3. Create a feature branch\n" +
			"4. Make your changes\n" +
			"5. Run tests: make test\n" +
			"6. Run linter: make lint\n" +
			"7. Commit your changes\n" +
			"8. Push to your fork\n" +
			"9. Create a Pull Request\n\n" +
			"## Code Style\n\n" +
			"- Follow Go conventions\n" +
			"- Use gofmt for formatting\n" +
			"- Follow SOLID principles\n" +
			"- Write tests for new features\n" +
			"- Update documentation as needed\n\n" +
			"## Commit Messages\n\n" +
			"Use clear, descriptive commit messages following conventional commits format.\n",
		Permissions: 0644,
	},
	{
		Path: "LICENSE",
		Content: `MIT License

Copyright (c) 2024 {{.ProjectName}}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
`,
		Permissions: 0644,
	},
	{
		Path: "CHANGELOG.md",
		Content: `# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure
- REST API with Gin framework
- Clean Architecture implementation
- Structured logging with Zap
- Prometheus metrics
- Distributed tracing with OpenTelemetry
- Input validation
- Rate limiting
- Security headers and CORS
- PostgreSQL database support
- Redis caching
- Database migrations
- Docker and Docker Compose setup
- Kubernetes manifests
- CI/CD with GitHub Actions
`,
		Permissions: 0644,
	},
	{
		Path: "README.md",
		Content: "# {{.ProjectName}}\n\n" +
			"REST API built with Go and Gin framework following Clean Architecture principles.\n\n" +
			"## Structure\n\n" +
			"```\n" +
			".\n" +
			"cmd/\n" +
			"  server/          # Application entry point\n" +
			"internal/\n" +
			"  domain/          # Business entities and interfaces\n" +
			"  usecase/         # Business logic\n" +
			"  interface/       # HTTP handlers and routers\n" +
			"  infrastructure/  # External concerns (DB, config, etc)\n" +
			"```\n\n" +
			"## Getting Started\n\n" +
			"### Local Development\n\n" +
			"```bash\n" +
			"go mod download\n" +
			"go run cmd/server/main.go\n" +
			"```\n\n" +
			"### Docker Compose\n\n" +
			"```bash\n" +
			"docker-compose up -d\n" +
			"```\n\n" +
			"### Docker\n\n" +
			"```bash\n" +
			"docker build -t {{.PackageName}} .\n" +
			"docker run -p 8080:8080 {{.PackageName}}\n" +
			"```\n\n" +
			"### Kubernetes\n\n" +
			"```bash\n" +
			"kubectl apply -f k8s/deployment.yaml\n" +
			"```\n\n" +
			"## Environment Variables\n\n" +
			"- PORT: Server port (default: 8080)\n" +
			"- ENVIRONMENT: Environment name (default: development)\n" +
			"- DATABASE_URL: PostgreSQL connection string\n" +
			"- DATABASE_MAX_CONNS: Maximum database connections (default: 25)\n" +
			"- REDIS_URL: Redis server address (default: localhost:6379)\n" +
			"- REDIS_PASSWORD: Redis password (optional)\n" +
			"- REDIS_DB: Redis database number (default: 0)\n\n" +
			"## Database Migrations\n\n" +
			"```bash\n" +
			"migrate -path migrations -database \"$DATABASE_URL\" up\n" +
			"```\n\n" +
			"## API Endpoints\n\n" +
			"- GET /api/v1/health - Health check endpoint\n" +
			"- GET /metrics - Prometheus metrics endpoint\n\n" +
			"## Architecture\n\n" +
			"This project follows Clean Architecture principles:\n\n" +
			"- **Domain Layer**: Business entities and repository interfaces\n" +
			"- **UseCase Layer**: Business logic and orchestration\n" +
			"- **Interface Layer**: HTTP handlers and routers\n" +
			"- **Infrastructure Layer**: Database, cache, config, and external services\n\n" +
			"## Features\n\n" +
			"- Structured logging with Zap\n" +
			"- Prometheus metrics\n" +
			"- Distributed tracing with OpenTelemetry\n" +
			"- Input validation\n" +
			"- Rate limiting\n" +
			"- Security headers and CORS\n" +
			"- PostgreSQL database support\n" +
			"- Redis caching\n" +
			"- Database migrations\n\n" +
			"## Development\n\n" +
			"```bash\n" +
			"make test          # Run tests\n" +
			"make test-coverage # Run tests with coverage\n" +
			"make lint         # Run linter\n" +
			"make migrate-up    # Run database migrations\n" +
			"```\n",
		Permissions: 0644,
	},
}

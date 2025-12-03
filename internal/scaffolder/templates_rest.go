package scaffolder

var restTemplates = []FileTemplate{
	{
		Path: "go.mod",
		Content: `module github.com/{{.ModuleName}}

go 1.21

require (
	github.com/gin-gonic/gin v1.9.1
	go.uber.org/zap v1.26.0
	github.com/prometheus/client_golang v1.18.0
	go.opentelemetry.io/otel v1.21.0
	go.opentelemetry.io/otel/trace v1.21.0
	go.opentelemetry.io/otel/exporters/jaeger v1.17.0
	go.opentelemetry.io/otel/sdk v1.21.0
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

	"github.com/{{.ModuleName}}/internal/infrastructure/config"
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

	if cfg.Environment == "production" {
		gin.SetMode(gin.ReleaseMode)
	}

	r := router.New(cfg, log, tracer)

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
	ServerPort       int
	MetricsPort      int
	JaegerEndpoint   string
	LogLevel         string
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

	return &Config{
		Environment:    env,
		ServerPort:     port,
		MetricsPort:    metricsPort,
		JaegerEndpoint: jaegerEndpoint,
		LogLevel:       logLevel,
	}
}
`,
		Permissions: 0644,
	},
	{
		Path: "internal/infrastructure/router/router.go",
		Content: `package router

import (
	"github.com/{{.ModuleName}}/internal/infrastructure/config"
	"github.com/{{.ModuleName}}/internal/infrastructure/middleware"
	"github.com/{{.ModuleName}}/internal/interface/http/handler"
	"github.com/gin-gonic/gin"
	"go.opentelemetry.io/otel/trace"
	"go.uber.org/zap"
)

func New(cfg *config.Config, log *zap.Logger, tracer trace.TracerProvider) *gin.Engine {
	r := gin.New()

	r.Use(middleware.Logger(log))
	r.Use(middleware.Recovery(log))
	r.Use(middleware.Metrics())
	r.Use(middleware.Tracing(tracer))

	api := r.Group("/api/v1")
	{
		healthHandler := handler.NewHealthHandler(log)
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

	"github.com/{{.ModuleName}}/internal/infrastructure/metrics"
	"github.com/gin-gonic/gin"
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
	log *zap.Logger
}

func NewHealthHandler(log *zap.Logger) *HealthHandler {
	return &HealthHandler{
		log: log,
	}
}

func (h *HealthHandler) Check(c *gin.Context) {
	h.log.Debug("health check requested")
	c.JSON(http.StatusOK, gin.H{
		"status": "ok",
	})
}
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
			"```bash\n" +
			"go mod download\n" +
			"go run cmd/server/main.go\n" +
			"```\n\n" +
			"## Environment Variables\n\n" +
			"- PORT: Server port (default: 8080)\n" +
			"- ENVIRONMENT: Environment name (default: development)\n\n" +
			"## API Endpoints\n\n" +
			"- GET /api/v1/health - Health check endpoint\n",
		Permissions: 0644,
	},
}

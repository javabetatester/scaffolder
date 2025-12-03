package scaffolder

var grpcTemplates = []FileTemplate{
	{
		Path: "go.mod",
		Content: `module github.com/{{.ModuleName}}

go 1.21

require (
	github.com/google/uuid v1.5.0
	go.uber.org/zap v1.26.0
	github.com/prometheus/client_golang v1.18.0
	go.opentelemetry.io/otel v1.21.0
	go.opentelemetry.io/otel/trace v1.21.0
	go.opentelemetry.io/otel/exporters/jaeger v1.17.0
	go.opentelemetry.io/otel/sdk v1.21.0
	google.golang.org/grpc v1.60.1
	google.golang.org/protobuf v1.31.0
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
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/{{.ModuleName}}/internal/infrastructure/cache"
	"github.com/{{.ModuleName}}/internal/infrastructure/config"
	"github.com/{{.ModuleName}}/internal/infrastructure/database"
	"github.com/{{.ModuleName}}/internal/infrastructure/interceptor"
	"github.com/{{.ModuleName}}/internal/infrastructure/logger"
	"github.com/{{.ModuleName}}/internal/infrastructure/metrics"
	"github.com/{{.ModuleName}}/internal/infrastructure/server"
	"github.com/{{.ModuleName}}/internal/infrastructure/tracing"
	pb "github.com/{{.ModuleName}}/proto/{{.PackageName}}"
	"google.golang.org/grpc"
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

	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", cfg.ServerPort))
	if err != nil {
		log.Fatal("failed to listen", zap.Error(err))
	}

	s := grpc.NewServer(
		grpc.UnaryInterceptor(interceptor.ChainUnary(
			interceptor.Logger(log),
			interceptor.Metrics(),
			interceptor.Tracing(tracer),
		)),
	)

	grpcServer := server.NewGRPCServer(log, db, redisCache)
	pb.Register{{.ProjectNamePascal}}ServiceServer(s, grpcServer)

	go func() {
		log.Info("gRPC server started", zap.Int("port", cfg.ServerPort), zap.Int("metrics_port", cfg.MetricsPort))
		if err := s.Serve(lis); err != nil {
			log.Fatal("failed to serve", zap.Error(err))
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Info("shutting down server...")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	done := make(chan bool)
	go func() {
		s.GracefulStop()
		done <- true
	}()

	select {
	case <-done:
		log.Info("server stopped gracefully")
	case <-ctx.Done():
		log.Info("server forced to shutdown")
		s.Stop()
	}

	log.Info("server exited")
}
`,
		Permissions: 0644,
	},
	{
		Path: "proto/{{.PackageName}}.proto",
		Content: `syntax = "proto3";

package {{.PackageName}};

option go_package = "github.com/{{.ModuleName}}/proto/{{.PackageName}}";

service {{.ProjectNamePascal}}Service {
  rpc HealthCheck(HealthCheckRequest) returns (HealthCheckResponse);
  rpc GetEntity(GetEntityRequest) returns (GetEntityResponse);
  rpc CreateEntity(CreateEntityRequest) returns (CreateEntityResponse);
}

message HealthCheckRequest {}

message HealthCheckResponse {
  string status = 1;
}

message GetEntityRequest {
  string id = 1;
}

message GetEntityResponse {
  string id = 1;
  string name = 2;
}

message CreateEntityRequest {
  string name = 1;
}

message CreateEntityResponse {
  string id = 1;
  string name = 2;
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
	Environment     string
	ServerPort       int
	MetricsPort      int
	JaegerEndpoint   string
	LogLevel         string
	DatabaseURL      string
	DatabaseMaxConns int
	RedisURL         string
	RedisPassword    string
	RedisDB          int
}

func Load() *Config {
	port := 50051
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
		Path: "internal/infrastructure/server/grpc_server.go",
		Content: `package server

import (
	"context"

	"github.com/{{.ModuleName}}/internal/infrastructure/repository"
	"github.com/{{.ModuleName}}/internal/usecase"
	pb "github.com/{{.ModuleName}}/proto/{{.PackageName}}"
	"go.uber.org/zap"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type GRPCServer struct {
	pb.Unimplemented{{.ProjectNamePascal}}ServiceServer
	useCase *usecase.UseCase
	log     *zap.Logger
	db      interface{}
	cache   interface{}
}

func NewGRPCServer(log *zap.Logger, db interface{}, cache interface{}) *GRPCServer {
	repo := repository.NewRepository()
	useCase := usecase.NewUseCase(repo)

	return &GRPCServer{
		useCase: useCase,
		log:     log,
		db:      db,
		cache:   cache,
	}
}

func (s *GRPCServer) HealthCheck(ctx context.Context, req *pb.HealthCheckRequest) (*pb.HealthCheckResponse, error) {
	return &pb.HealthCheckResponse{
		Status: "ok",
	}, nil
}

func (s *GRPCServer) GetEntity(ctx context.Context, req *pb.GetEntityRequest) (*pb.GetEntityResponse, error) {
	if req.Id == "" {
		return nil, status.Errorf(codes.InvalidArgument, "id is required")
	}

	entity, err := s.useCase.GetEntity(ctx, req.Id)
	if err != nil {
		return nil, status.Errorf(codes.NotFound, "failed to get entity: %v", err)
	}

	return &pb.GetEntityResponse{
		Id:   entity.ID,
		Name: entity.Name,
	}, nil
}

func (s *GRPCServer) CreateEntity(ctx context.Context, req *pb.CreateEntityRequest) (*pb.CreateEntityResponse, error) {
	if req.Name == "" {
		return nil, status.Errorf(codes.InvalidArgument, "name is required")
	}

	if len(req.Name) > 100 {
		return nil, status.Errorf(codes.InvalidArgument, "name must be less than 100 characters")
	}

	entity, err := s.useCase.CreateEntity(ctx, req.Name)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to create entity: %v", err)
	}

	return &pb.CreateEntityResponse{
		Id:   entity.ID,
		Name: entity.Name,
	}, nil
}
`,
		Permissions: 0644,
	},
	{
		Path: "internal/domain/entity/entity.go",
		Content: `package entity

type Entity struct {
	ID   string
	Name string
}
`,
		Permissions: 0644,
	},
	{
		Path: "internal/domain/repository/repository.go",
		Content: `package repository

import (
	"context"

	"github.com/{{.ModuleName}}/internal/domain/entity"
)

type Repository interface {
	FindByID(ctx context.Context, id string) (*entity.Entity, error)
	Create(ctx context.Context, e *entity.Entity) error
}
`,
		Permissions: 0644,
	},
	{
		Path: "internal/usecase/usecase.go",
		Content: `package usecase

import (
	"context"
	"fmt"

	"github.com/{{.ModuleName}}/internal/domain/entity"
	"github.com/{{.ModuleName}}/internal/domain/repository"
	"github.com/google/uuid"
)

type UseCase struct {
	repo repository.Repository
}

func NewUseCase(repo repository.Repository) *UseCase {
	return &UseCase{
		repo: repo,
	}
}

func (uc *UseCase) GetEntity(ctx context.Context, id string) (*entity.Entity, error) {
	return uc.repo.FindByID(ctx, id)
}

func (uc *UseCase) CreateEntity(ctx context.Context, name string) (*entity.Entity, error) {
	e := &entity.Entity{
		ID:   uuid.New().String(),
		Name: name,
	}

	if err := uc.repo.Create(ctx, e); err != nil {
		return nil, fmt.Errorf("failed to create entity: %w", err)
	}

	return e, nil
}
`,
		Permissions: 0644,
	},
	{
		Path: "internal/infrastructure/repository/memory_repository.go",
		Content: `package repository

import (
	"context"
	"fmt"
	"sync"

	"github.com/{{.ModuleName}}/internal/domain/entity"
	domainRepo "github.com/{{.ModuleName}}/internal/domain/repository"
)

type MemoryRepository struct {
	mu   sync.RWMutex
	data map[string]*entity.Entity
}

func NewRepository() domainRepo.Repository {
	return &MemoryRepository{
		data: make(map[string]*entity.Entity),
	}
}

func (r *MemoryRepository) FindByID(ctx context.Context, id string) (*entity.Entity, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	e, exists := r.data[id]
	if !exists {
		return nil, fmt.Errorf("entity not found: %s", id)
	}

	return e, nil
}

func (r *MemoryRepository) Create(ctx context.Context, e *entity.Entity) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.data[e.ID]; exists {
		return fmt.Errorf("entity already exists: %s", e.ID)
	}

	r.data[e.ID] = e
	return nil
}
`,
		Permissions: 0644,
	},
	{
		Path: "Makefile",
		Content: `PROTOC_GEN_GO := $(shell which protoc-gen-go)
PROTOC_GEN_GRPC_GO := $(shell which protoc-gen-go-grpc)

.PHONY: proto
proto:
	@if [ -z "$(PROTOC_GEN_GO)" ]; then \
		echo "Installing protoc-gen-go..."; \
		go install google.golang.org/protobuf/cmd/protoc-gen-go@latest; \
	fi
	@if [ -z "$(PROTOC_GEN_GRPC_GO)" ]; then \
		echo "Installing protoc-gen-go-grpc..."; \
		go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest; \
	fi
	protoc --go_out=. --go_opt=paths=source_relative \\
		--go-grpc_out=. --go-grpc_opt=paths=source_relative \\
		proto/*.proto

.PHONY: run
run:
	go run cmd/server/main.go

.PHONY: build
build:
	go build -o bin/server cmd/server/main.go
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
	GRPCRequestsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "grpc_requests_total",
			Help: "Total number of gRPC requests",
		},
		[]string{"method", "code"},
	)

	GRPCRequestDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "grpc_request_duration_seconds",
			Help:    "gRPC request duration in seconds",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"method"},
	)
)

func init() {
	prometheus.MustRegister(GRPCRequestsTotal)
	prometheus.MustRegister(GRPCRequestDuration)
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
		Path: "internal/infrastructure/interceptor/interceptor.go",
		Content: `package interceptor

import (
	"context"
	"fmt"
	"time"

	"github.com/{{.ModuleName}}/internal/infrastructure/metrics"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/trace"
	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/status"
)

func ChainUnary(interceptors ...grpc.UnaryServerInterceptor) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		chain := handler
		for i := len(interceptors) - 1; i >= 0; i-- {
			chain = buildChain(interceptors[i], chain, info)
		}
		return chain(ctx, req)
	}
}

func buildChain(c grpc.UnaryServerInterceptor, n grpc.UnaryHandler, info *grpc.UnaryServerInfo) grpc.UnaryHandler {
	return func(ctx context.Context, req interface{}) (interface{}, error) {
		return c(ctx, req, info, n)
	}
}

func Logger(log *zap.Logger) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		start := time.Now()

		resp, err := handler(ctx, req)

		duration := time.Since(start)
		code := status.Code(err)

		log.Info("grpc request",
			zap.String("method", info.FullMethod),
			zap.String("code", code.String()),
			zap.Duration("duration", duration),
			zap.Error(err),
		)

		return resp, err
	}
}

func Metrics() grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		start := time.Now()

		resp, err := handler(ctx, req)

		duration := time.Since(start).Seconds()
		code := status.Code(err)

		metrics.GRPCRequestsTotal.WithLabelValues(info.FullMethod, code.String()).Inc()
		metrics.GRPCRequestDuration.WithLabelValues(info.FullMethod).Observe(duration)

		return resp, err
	}
}

func Tracing(tp trace.TracerProvider) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		ctx = otel.GetTextMapPropagator().Extract(ctx, propagation.HeaderCarrier{})
		tr := tp.Tracer("{{.PackageName}}")
		ctx, span := tr.Start(ctx, info.FullMethod)
		defer span.End()

		resp, err := handler(ctx, req)

		if err != nil {
			span.RecordError(err)
			span.SetStatus(codes.Error, err.Error())
		} else {
			span.SetStatus(codes.Ok, "")
		}

		span.SetAttributes(
			attribute.String("grpc.method", info.FullMethod),
			attribute.String("grpc.code", status.Code(err).String()),
		)

		return resp, err
	}
}
`,
		Permissions: 0644,
	},
	{
		Path: "internal/infrastructure/server/grpc_server_test.go",
		Content: `package server

import (
	"context"
	"testing"

	pb "github.com/{{.ModuleName}}/proto/{{.PackageName}}"
	"github.com/stretchr/testify/assert"
	"go.uber.org/zap"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestGRPCServer_HealthCheck(t *testing.T) {
	log := zap.NewNop()
	server := NewGRPCServer(log)

	req := &pb.HealthCheckRequest{}
	resp, err := server.HealthCheck(context.Background(), req)

	assert.NoError(t, err)
	assert.Equal(t, "ok", resp.Status)
}

func TestGRPCServer_CreateEntity(t *testing.T) {
	log := zap.NewNop()
	server := NewGRPCServer(log)

	tests := []struct {
		name    string
		req     *pb.CreateEntityRequest
		wantErr bool
		errCode codes.Code
	}{
		{
			name:    "valid request",
			req:     &pb.CreateEntityRequest{Name: "Test Entity"},
			wantErr: false,
		},
		{
			name:    "empty name",
			req:     &pb.CreateEntityRequest{Name: ""},
			wantErr: true,
			errCode: codes.InvalidArgument,
		},
		{
			name:    "name too long",
			req:     &pb.CreateEntityRequest{Name: string(make([]byte, 101))},
			wantErr: true,
			errCode: codes.InvalidArgument,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := server.CreateEntity(context.Background(), tt.req)

			if tt.wantErr {
				assert.Error(t, err)
				assert.Nil(t, resp)
				st, ok := status.FromError(err)
				assert.True(t, ok)
				assert.Equal(t, tt.errCode, st.Code())
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, resp)
				assert.NotEmpty(t, resp.Id)
				assert.Equal(t, tt.req.Name, resp.Name)
			}
		})
	}
}

func TestGRPCServer_GetEntity(t *testing.T) {
	log := zap.NewNop()
	server := NewGRPCServer(log)

	createReq := &pb.CreateEntityRequest{Name: "Test Entity"}
	created, err := server.CreateEntity(context.Background(), createReq)
	assert.NoError(t, err)

	tests := []struct {
		name    string
		req     *pb.GetEntityRequest
		wantErr bool
		errCode codes.Code
	}{
		{
			name:    "valid request",
			req:     &pb.GetEntityRequest{Id: created.Id},
			wantErr: false,
		},
		{
			name:    "empty id",
			req:     &pb.GetEntityRequest{Id: ""},
			wantErr: true,
			errCode: codes.InvalidArgument,
		},
		{
			name:    "nonexistent entity",
			req:     &pb.GetEntityRequest{Id: "nonexistent"},
			wantErr: true,
			errCode: codes.NotFound,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := server.GetEntity(context.Background(), tt.req)

			if tt.wantErr {
				assert.Error(t, err)
				assert.Nil(t, resp)
				st, ok := status.FromError(err)
				assert.True(t, ok)
				assert.Equal(t, tt.errCode, st.Code())
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, resp)
				assert.Equal(t, created.Id, resp.Id)
			}
		})
	}
}
`,
		Permissions: 0644,
	},
	{
		Path: "Makefile",
		Content: `PROTOC_GEN_GO := $(shell which protoc-gen-go)
PROTOC_GEN_GRPC_GO := $(shell which protoc-gen-go-grpc)

.PHONY: proto
proto:
	@if [ -z "$(PROTOC_GEN_GO)" ]; then \
		echo "Installing protoc-gen-go..."; \
		go install google.golang.org/protobuf/cmd/protoc-gen-go@latest; \
	fi
	@if [ -z "$(PROTOC_GEN_GRPC_GO)" ]; then \
		echo "Installing protoc-gen-go-grpc..."; \
		go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest; \
	fi
	protoc --go_out=. --go_opt=paths=source_relative \\
		--go-grpc_out=. --go-grpc_opt=paths=source_relative \\
		proto/*.proto

.PHONY: run
run:
	go run cmd/server/main.go

.PHONY: build
build:
	go build -o bin/server cmd/server/main.go

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
	docker run -p 50051:50051 {{.PackageName}}:latest

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

proto/*.pb.go
proto/*_grpc.pb.go
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
COPY --from=builder /app/proto ./proto

EXPOSE 50051

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

proto/*.pb.go
proto/*_grpc.pb.go
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
      - "50051:50051"
    environment:
      PORT: 50051
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
        - containerPort: 50051
        env:
        - name: PORT
          value: "50051"
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
          exec:
            command:
            - /bin/sh
            - -c
            - "grpc_health_probe -addr=:50051 || exit 1"
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - /bin/sh
            - -c
            - "grpc_health_probe -addr=:50051 || exit 1"
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
    port: 50051
    targetPort: 50051
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

    - name: Generate protobuf
      run: make proto

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

    - name: Generate protobuf
      run: make proto

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
			"3. Install Protocol Buffers compiler\n" +
			"4. Create a feature branch\n" +
			"5. Make your changes\n" +
			"6. Generate protobuf code: make proto\n" +
			"7. Run tests: make test\n" +
			"8. Run linter: make lint\n" +
			"9. Commit your changes\n" +
			"10. Push to your fork\n" +
			"11. Create a Pull Request\n\n" +
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
- gRPC service with Protocol Buffers
- Clean Architecture implementation
- Structured logging with Zap
- Prometheus metrics
- Distributed tracing with OpenTelemetry
- Input validation
- gRPC interceptors
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
			"gRPC service built with Go and Protocol Buffers following Clean Architecture principles.\n\n" +
			"## Structure\n\n" +
			"```\n" +
			".\n" +
			"cmd/\n" +
			"  server/          # Application entry point\n" +
			"proto/             # Protocol Buffer definitions\n" +
			"internal/\n" +
			"  domain/          # Business entities and interfaces\n" +
			"  usecase/         # Business logic\n" +
			"  infrastructure/  # External concerns (gRPC server, config, etc)\n" +
			"```\n\n" +
			"## Getting Started\n\n" +
			"### Prerequisites\n\n" +
			"- Go 1.21+\n" +
			"- Protocol Buffers compiler (protoc)\n" +
			"- protoc-gen-go and protoc-gen-go-grpc plugins\n\n" +
			"### Generate Protocol Buffer Code\n\n" +
			"```bash\n" +
			"make proto\n" +
			"```\n\n" +
			"### Run Server\n\n" +
			"```bash\n" +
			"go mod download\n" +
			"make proto\n" +
			"go run cmd/server/main.go\n" +
			"```\n\n" +
			"### Docker Compose\n\n" +
			"```bash\n" +
			"docker-compose up -d\n" +
			"```\n\n" +
			"### Docker\n\n" +
			"```bash\n" +
			"docker build -t {{.PackageName}} .\n" +
			"docker run -p 50051:50051 {{.PackageName}}\n" +
			"```\n\n" +
			"### Kubernetes\n\n" +
			"```bash\n" +
			"kubectl apply -f k8s/deployment.yaml\n" +
			"```\n\n" +
			"## Environment Variables\n\n" +
			"- PORT: Server port (default: 50051)\n" +
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
			"## gRPC Services\n\n" +
			"- HealthCheck: Health check endpoint\n" +
			"- GetEntity: Get entity by ID\n" +
			"- CreateEntity: Create new entity\n\n" +
			"## Architecture\n\n" +
			"This project follows Clean Architecture principles:\n\n" +
			"- **Domain Layer**: Business entities and repository interfaces\n" +
			"- **UseCase Layer**: Business logic and orchestration\n" +
			"- **Interface Layer**: gRPC handlers and interceptors\n" +
			"- **Infrastructure Layer**: Database, cache, config, and external services\n\n" +
			"## Features\n\n" +
			"- Structured logging with Zap\n" +
			"- Prometheus metrics\n" +
			"- Distributed tracing with OpenTelemetry\n" +
			"- Input validation\n" +
			"- gRPC interceptors for observability\n" +
			"- PostgreSQL database support\n" +
			"- Redis caching\n" +
			"- Database migrations\n" +
			"- Protocol Buffers code generation\n\n" +
			"## Development\n\n" +
			"```bash\n" +
			"make proto          # Generate protobuf code\n" +
			"make test          # Run tests\n" +
			"make test-coverage # Run tests with coverage\n" +
			"make lint          # Run linter\n" +
			"make migrate-up    # Run database migrations\n" +
			"```\n",
		Permissions: 0644,
	},
}

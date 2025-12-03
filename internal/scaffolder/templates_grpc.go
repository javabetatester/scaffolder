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

	"github.com/{{.ModuleName}}/internal/infrastructure/config"
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

	grpcServer := server.NewGRPCServer(log)
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
	Environment    string
	ServerPort     int
	MetricsPort    int
	JaegerEndpoint string
	LogLevel       string
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
}

func NewGRPCServer(log *zap.Logger) *GRPCServer {
	repo := repository.NewRepository()
	useCase := usecase.NewUseCase(repo)

	return &GRPCServer{
		useCase: useCase,
		log:     log,
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
			"go run cmd/server/main.go\n" +
			"```\n\n" +
			"## Environment Variables\n\n" +
			"- PORT: Server port (default: 50051)\n" +
			"- ENVIRONMENT: Environment name (default: development)\n\n" +
			"## gRPC Services\n\n" +
			"- HealthCheck: Health check endpoint\n" +
			"- GetEntity: Get entity by ID\n" +
			"- CreateEntity: Create new entity\n",
		Permissions: 0644,
	},
}

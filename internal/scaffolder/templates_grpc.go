package scaffolder

var grpcTemplates = []FileTemplate{
	{
		Path: "go.mod",
		Content: `module github.com/{{.ModuleName}}

go 1.21

require (
	google.golang.org/grpc v1.60.1
	google.golang.org/protobuf v1.31.0
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
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/{{.ModuleName}}/internal/infrastructure/config"
	"github.com/{{.ModuleName}}/internal/infrastructure/server"
	pb "github.com/{{.ModuleName}}/proto/{{.PackageName}}"
	"google.golang.org/grpc"
)

func main() {
	cfg := config.Load()

	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", cfg.ServerPort))
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	s := grpc.NewServer()
	grpcServer := server.NewGRPCServer()
	
	pb.Register{{.ProjectNamePascal}}ServiceServer(s, grpcServer)

	go func() {
		log.Printf("gRPC server started on port %d", cfg.ServerPort)
		if err := s.Serve(lis); err != nil {
			log.Fatalf("failed to serve: %v", err)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("shutting down server...")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	done := make(chan bool)
	go func() {
		s.GracefulStop()
		done <- true
	}()

	select {
	case <-done:
		log.Println("server stopped gracefully")
	case <-ctx.Done():
		log.Println("server forced to shutdown")
		s.Stop()
	}

	log.Println("server exited")
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
	Environment string
	ServerPort  int
}

func Load() *Config {
	port := 50051
	if portStr := os.Getenv("PORT"); portStr != "" {
		if p, err := strconv.Atoi(portStr); err == nil {
			port = p
		}
	}

	env := os.Getenv("ENVIRONMENT")
	if env == "" {
		env = "development"
	}

	return &Config{
		Environment: env,
		ServerPort:  port,
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
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type GRPCServer struct {
	pb.Unimplemented{{.ProjectNamePascal}}ServiceServer
	useCase *usecase.UseCase
}

func NewGRPCServer() *GRPCServer {
	repo := repository.NewRepository()
	useCase := usecase.NewUseCase(repo)

	return &GRPCServer{
		useCase: useCase,
	}
}

func (s *GRPCServer) HealthCheck(ctx context.Context, req *pb.HealthCheckRequest) (*pb.HealthCheckResponse, error) {
	return &pb.HealthCheckResponse{
		Status: "ok",
	}, nil
}

func (s *GRPCServer) GetEntity(ctx context.Context, req *pb.GetEntityRequest) (*pb.GetEntityResponse, error) {
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
	"time"

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

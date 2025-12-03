package scaffolder

var restTemplates = []FileTemplate{
	{
		Path: "go.mod",
		Content: `module github.com/{{.ModuleName}}

go 1.21

require (
	github.com/gin-gonic/gin v1.9.1
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
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/{{.ModuleName}}/internal/infrastructure/config"
	"github.com/{{.ModuleName}}/internal/infrastructure/router"
	"github.com/gin-gonic/gin"
)

func main() {
	cfg := config.Load()

	if cfg.Environment == "production" {
		gin.SetMode(gin.ReleaseMode)
	}

	r := router.New()

	srv := &http.Server{
		Addr:    fmt.Sprintf(":%d", cfg.ServerPort),
		Handler: r,
	}

	go func() {
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("failed to start server: %v", err)
		}
	}()

	log.Printf("server started on port %d", cfg.ServerPort)

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("shutting down server...")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		log.Fatalf("server forced to shutdown: %v", err)
	}

	log.Println("server exited")
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
	port := 8080
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
		Path: "internal/infrastructure/router/router.go",
		Content: `package router

import (
	"github.com/{{.ModuleName}}/internal/interface/http/handler"
	"github.com/gin-gonic/gin"
)

func New() *gin.Engine {
	r := gin.New()

	r.Use(gin.Logger())
	r.Use(gin.Recovery())

	api := r.Group("/api/v1")
	{
		healthHandler := handler.NewHealthHandler()
		api.GET("/health", healthHandler.Check)
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

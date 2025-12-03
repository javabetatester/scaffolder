package scaffolder

var cliTemplates = []FileTemplate{
	{
		Path: "go.mod",
		Content: `module github.com/{{.ModuleName}}

go 1.21

require (
	github.com/spf13/cobra v1.8.0
	github.com/spf13/viper v1.18.2
	go.uber.org/zap v1.26.0
	github.com/stretchr/testify v1.8.4
)
`,
		Permissions: 0644,
	},
	{
		Path: "cmd/{{.PackageName}}/main.go",
		Content: `package main

import (
	"os"

	"github.com/{{.ModuleName}}/internal/cmd"
)

func main() {
	if err := cmd.Execute(); err != nil {
		os.Exit(1)
	}
}
`,
		Permissions: 0644,
	},
	{
		Path: "internal/cmd/root.go",
		Content: `package cmd

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var cfgFile string

var rootCmd = &cobra.Command{
	Use:   "{{.PackageName}}",
	Short: "{{.ProjectName}} CLI application",
	Long: "{{.ProjectName}} is a CLI application built with Cobra.\n" +
		"It provides a set of commands to interact with the application.",
	Version: "1.0.0",
}

func Execute() error {
	return rootCmd.Execute()
}

func init() {
	cobra.OnInitialize(initConfig)
	rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "config file (default is $HOME/.{{.PackageName}}.yaml)")
	rootCmd.PersistentFlags().Bool("verbose", false, "verbose output")
}

func initConfig() {
	if cfgFile != "" {
		viper.SetConfigFile(cfgFile)
	} else {
		home, err := os.UserHomeDir()
		if err != nil {
			fmt.Println(err)
			os.Exit(1)
		}

		viper.AddConfigPath(home)
		viper.AddConfigPath(".")
		viper.SetConfigType("yaml")
		viper.SetConfigName(".{{.PackageName}}")
	}

	viper.AutomaticEnv()

	if err := viper.ReadInConfig(); err == nil {
		fmt.Println("Using config file:", viper.ConfigFileUsed())
	}
}
`,
		Permissions: 0644,
	},
	{
		Path: "internal/cmd/version.go",
		Content: `package cmd

import (
	"fmt"

	"github.com/spf13/cobra"
)

var versionCmd = &cobra.Command{
	Use:   "version",
	Short: "Print the version number",
	Long:  "Print the version number of {{.ProjectName}}",
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Println("{{.ProjectName}} version 1.0.0")
	},
}

func init() {
	rootCmd.AddCommand(versionCmd)
}
`,
		Permissions: 0644,
	},
	{
		Path: "internal/cmd/config.go",
		Content: `package cmd

import (
	"fmt"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var configCmd = &cobra.Command{
	Use:   "config",
	Short: "Manage configuration",
	Long:  "Manage configuration settings for {{.ProjectName}}",
}

var configShowCmd = &cobra.Command{
	Use:   "show",
	Short: "Show current configuration",
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Println("Configuration:")
		allSettings := viper.AllSettings()
		for key, value := range allSettings {
			fmt.Printf("  %s: %v\n", key, value)
		}
	},
}

var configSetCmd = &cobra.Command{
	Use:   "set [key] [value]",
	Short: "Set a configuration value",
	Args:  cobra.ExactArgs(2),
	Run: func(cmd *cobra.Command, args []string) {
		key := args[0]
		value := args[1]
		viper.Set(key, value)
		fmt.Printf("Set %s = %s\n", key, value)
	},
}

func init() {
	configCmd.AddCommand(configShowCmd)
	configCmd.AddCommand(configSetCmd)
	rootCmd.AddCommand(configCmd)
}
`,
		Permissions: 0644,
	},
	{
		Path: "internal/cmd/example.go",
		Content: `package cmd

import (
	"fmt"

	"github.com/{{.ModuleName}}/internal/pkg/logger"
	"github.com/spf13/cobra"
	"go.uber.org/zap"
)

var exampleCmd = &cobra.Command{
	Use:   "example [name]",
	Short: "Example command",
	Long:  "Example command that demonstrates basic functionality",
	Args:  cobra.MinimumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		name := args[0]
		
		if name == "" {
			return fmt.Errorf("name cannot be empty")
		}
		
		if len(name) > 100 {
			return fmt.Errorf("name must be less than 100 characters")
		}
		
		verbose, _ := cmd.Flags().GetBool("verbose")
		
		log := logger.New(verbose)
		defer log.Sync()
		
		log.Debug("Running example command with verbose mode")
		log.Info("Hello", zap.String("name", name))
		
		return nil
	},
}

func init() {
	exampleCmd.Flags().BoolP("verbose", "v", false, "verbose output")
	rootCmd.AddCommand(exampleCmd)
}
`,
		Permissions: 0644,
	},
	{
		Path: "internal/config/config.go",
		Content: `package config

import (
	"github.com/spf13/viper"
)

type Config struct {
	AppName  string
	LogLevel string
	Timeout  int
}

func Load() *Config {
	viper.SetDefault("app.name", "{{.ProjectName}}")
	viper.SetDefault("log.level", "info")
	viper.SetDefault("timeout", 30)

	return &Config{
		AppName:  viper.GetString("app.name"),
		LogLevel: viper.GetString("log.level"),
		Timeout:  viper.GetInt("timeout"),
	}
}
`,
		Permissions: 0644,
	},
	{
		Path: "internal/pkg/logger/logger.go",
		Content: `package logger

import (
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

func New(verbose bool) *zap.Logger {
	var level zapcore.Level
	if verbose {
		level = zapcore.DebugLevel
	} else {
		level = zapcore.InfoLevel
	}

	config := zap.NewDevelopmentConfig()
	config.Level = zap.NewAtomicLevelAt(level)
	config.EncoderConfig.EncodeLevel = zapcore.CapitalColorLevelEncoder

	logger, err := config.Build()
	if err != nil {
		panic(err)
	}

	return logger.With(
		zap.String("service", "{{.PackageName}}"),
	)
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

*.yaml
!*.example.yaml
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
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o {{.PackageName}} ./cmd/{{.PackageName}}

FROM alpine:latest

RUN apk --no-cache add ca-certificates
WORKDIR /root/

COPY --from=builder /app/{{.PackageName}} .

ENTRYPOINT ["./{{.PackageName}}"]
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
*.yaml
!*.example.yaml
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
      run: go build -o bin/{{.PackageName}} ./cmd/{{.PackageName}}

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
- CLI application with Cobra
- Configuration management with Viper
- Structured logging with Zap
- Modular command structure
- Docker setup
- CI/CD with GitHub Actions
`,
		Permissions: 0644,
	},
	{
		Path: "README.md",
		Content: "# {{.ProjectName}}\n\n" +
			"CLI application built with Go and Cobra framework.\n\n" +
			"## Structure\n\n" +
			"```\n" +
			".\n" +
			"cmd/\n" +
			"  {{.PackageName}}/     # Application entry point\n" +
			"internal/\n" +
			"  cmd/                 # Cobra commands\n" +
			"  config/              # Configuration management\n" +
			"  pkg/                 # Shared packages\n" +
			"    logger/           # Logging utilities\n" +
			"```\n\n" +
			"## Getting Started\n\n" +
			"### Build\n\n" +
			"```bash\n" +
			"go mod download\n" +
			"go build -o bin/{{.PackageName}} cmd/{{.PackageName}}/main.go\n" +
			"```\n\n" +
			"### Install\n\n" +
			"```bash\n" +
			"go install ./cmd/{{.PackageName}}\n" +
			"```\n\n" +
			"### Usage\n\n" +
			"```bash\n" +
			"./bin/{{.PackageName}} --help\n" +
			"./bin/{{.PackageName}} version\n" +
			"./bin/{{.PackageName}} example world\n" +
			"./bin/{{.PackageName}} config show\n" +
			"```\n\n" +
			"### Docker\n\n" +
			"```bash\n" +
			"docker build -t {{.PackageName}} .\n" +
			"docker run {{.PackageName}} version\n" +
			"```\n\n" +
			"## Commands\n\n" +
			"- `version`: Print the version number\n" +
			"- `example [name]`: Example command\n" +
			"- `config show`: Show current configuration\n" +
			"- `config set [key] [value]`: Set a configuration value\n\n" +
			"## Configuration\n\n" +
			"Configuration can be set via:\n\n" +
			"1. Command line flags\n" +
			"2. Environment variables\n" +
			"3. Config file (YAML) in home directory or current directory\n\n" +
			"Config file name: `.{{.PackageName}}.yaml`\n\n" +
			"## Architecture\n\n" +
			"This project follows Clean Architecture principles:\n\n" +
			"- **Command Layer**: Cobra commands and CLI interface\n" +
			"- **Config Layer**: Configuration management with Viper\n" +
			"- **Package Layer**: Shared utilities (logger, etc.)\n\n" +
			"## Features\n\n" +
			"- Structured logging with Zap\n" +
			"- Configuration management with Viper\n" +
			"- Modular command structure\n" +
			"- Persistent flags\n" +
			"- Help system\n\n" +
			"## Development\n\n" +
			"```bash\n" +
			"make test          # Run tests\n" +
			"make test-coverage # Run tests with coverage\n" +
			"make lint         # Run linter\n" +
			"make fmt          # Format code\n" +
			"```\n",
		Permissions: 0644,
	},
	{
		Path: "internal/pkg/logger/logger_test.go",
		Content: `package logger

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNew(t *testing.T) {
	tests := []struct {
		name    string
		verbose bool
	}{
		{
			name:    "verbose mode",
			verbose: true,
		},
		{
			name:    "non-verbose mode",
			verbose: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			log := New(tt.verbose)
			assert.NotNil(t, log)
		})
	}
}
`,
		Permissions: 0644,
	},
	{
		Path: "Makefile",
		Content: `BINARY_NAME={{.PackageName}}
CMD_PATH=cmd/{{.PackageName}}

.PHONY: build
build:
	go build -o bin/$(BINARY_NAME) $(CMD_PATH)/main.go

.PHONY: install
install:
	go install ./$(CMD_PATH)

.PHONY: run
run:
	go run $(CMD_PATH)/main.go

.PHONY: clean
clean:
	rm -rf bin/
	rm -f $(BINARY_NAME)

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

.PHONY: docker-build
docker-build:
	docker build -t {{.PackageName}}:latest .

.PHONY: docker-run
docker-run:
	docker run {{.PackageName}}:latest

.PHONY: lint
lint:
	golangci-lint run

.PHONY: fmt
fmt:
	go fmt ./...

.PHONY: vet
vet:
	go vet ./...
`,
		Permissions: 0644,
	},
}

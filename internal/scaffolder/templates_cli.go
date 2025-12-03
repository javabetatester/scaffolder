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
			"Config file name: `.{{.PackageName}}.yaml`\n",
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

.PHONY: lint
lint:
	golangci-lint run
`,
		Permissions: 0644,
	},
}

package main

import (
	"fmt"
	"os"

	"github.com/bernardo/api/internal/scaffolder"
	"github.com/spf13/cobra"
)

var rootCmd = &cobra.Command{
	Use:   "scaffold",
	Short: "Project scaffolder for Go applications",
	Long: `Scaffold generates complete project structures for Go applications
following Clean Architecture, SOLID principles, and best practices.

Supported project types:
  - rest: REST API with Gin framework
  - grpc: gRPC service with Protocol Buffers
  - cli: CLI application with Cobra`,
}

var scaffoldCmd = &cobra.Command{
	Use:   "scaffold [type] [project-name]",
	Short: "Generate a new project structure",
	Long: `Generate a complete project structure for the specified type.

Types:
  rest    - REST API project with Clean Architecture
  grpc    - gRPC service project with Protocol Buffers
  cli     - CLI application project with Cobra commands

Example:
  scaffold rest my-api
  scaffold grpc my-service
  scaffold cli my-tool`,
	ValidArgs: []string{"rest", "grpc", "cli"},
	Args:      cobra.MatchAll(cobra.ExactArgs(2), cobra.OnlyValidArgs),
	RunE: func(cmd *cobra.Command, args []string) error {
		projectType := args[0]
		projectName := args[1]

		sc := scaffolder.New(projectType, projectName)
		if err := sc.Generate(); err != nil {
			return fmt.Errorf("scaffold failed: %w", err)
		}

		fmt.Printf("Project scaffolded successfully: %s\n", projectName)
		return nil
	},
}

func init() {
	rootCmd.AddCommand(scaffoldCmd)
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

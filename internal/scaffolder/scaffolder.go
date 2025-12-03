package scaffolder

import (
	"fmt"
	"os"
	"path/filepath"
)

type Scaffolder struct {
	projectType string
	projectName string
	outputDir   string
}

func New(projectType, projectName string) *Scaffolder {
	return &Scaffolder{
		projectType: projectType,
		projectName: projectName,
		outputDir:   projectName,
	}
}

func (s *Scaffolder) Generate() error {
	if err := s.validate(); err != nil {
		return err
	}

	if err := s.createOutputDir(); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}

	return nil
}

func (s *Scaffolder) TemplateVars() *TemplateVars {
	return s.buildTemplateVars()
}

func (s *Scaffolder) validate() error {
	validTypes := map[string]bool{
		"rest": true,
		"grpc": true,
		"cli":  true,
	}

	if !validTypes[s.projectType] {
		return fmt.Errorf("invalid project type: %s. Valid types: rest, grpc, cli", s.projectType)
	}

	if s.projectName == "" {
		return fmt.Errorf("project name cannot be empty")
	}

	if len(s.projectName) > 50 {
		return fmt.Errorf("project name too long (max 50 characters)")
	}

	return nil
}

func (s *Scaffolder) createOutputDir() error {
	if _, err := os.Stat(s.outputDir); err == nil {
		return fmt.Errorf("directory %s already exists", s.outputDir)
	}

	return os.MkdirAll(s.outputDir, 0755)
}

func (s *Scaffolder) ProjectType() string {
	return s.projectType
}

func (s *Scaffolder) ProjectName() string {
	return s.projectName
}

func (s *Scaffolder) OutputDir() string {
	return s.outputDir
}

func (s *Scaffolder) joinPath(elem ...string) string {
	return filepath.Join(s.outputDir, filepath.Join(elem...))
}

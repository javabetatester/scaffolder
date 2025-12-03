package scaffolder

import (
	"fmt"
	"os"
	"path/filepath"
)

type FileTemplate struct {
	Path        string
	Content     string
	Permissions os.FileMode
}

func (s *Scaffolder) createFileFromTemplate(ft FileTemplate, vars *TemplateVars) error {
	processedPath, err := s.processPathTemplate(ft.Path, vars)
	if err != nil {
		return fmt.Errorf("failed to process path template: %w", err)
	}

	fullPath := s.joinPath(processedPath)

	dir := filepath.Dir(fullPath)
	if errs := os.MkdirAll(dir, 0755); errs != nil {
		return fmt.Errorf("failed to create directory: %w", errs)
	}

	processedContent, err := s.processTemplate(ft.Content, vars)
	if err != nil {
		return fmt.Errorf("failed to process content template: %w", err)
	}

	perms := ft.Permissions
	if perms == 0 {
		perms = 0644
	}

	if err := os.WriteFile(fullPath, []byte(processedContent), perms); err != nil {
		return fmt.Errorf("failed to write file: %w", err)
	}

	return nil
}

func (s *Scaffolder) createDirectory(path string, vars *TemplateVars) error {
	processedPath, err := s.processPathTemplate(path, vars)
	if err != nil {
		return fmt.Errorf("failed to process path template: %w", err)
	}

	fullPath := s.joinPath(processedPath)
	return os.MkdirAll(fullPath, 0755)
}

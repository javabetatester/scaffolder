package scaffolder

import (
	"bytes"
	"fmt"
	"strings"
	"text/template"
)

type TemplateVars struct {
	ProjectName       string
	ProjectType       string
	ModuleName        string
	PackageName       string
	ProjectNameCamel  string
	ProjectNamePascal string
}

func (s *Scaffolder) buildTemplateVars() *TemplateVars {
	moduleName := s.normalizeModuleName(s.projectName)
	packageName := s.normalizePackageName(s.projectName)

	return &TemplateVars{
		ProjectName:       s.projectName,
		ProjectType:       s.projectType,
		ModuleName:        moduleName,
		PackageName:       packageName,
		ProjectNameCamel:  s.toCamelCase(s.projectName),
		ProjectNamePascal: s.toPascalCase(s.projectName),
	}
}

func (s *Scaffolder) normalizeModuleName(name string) string {
	name = strings.ToLower(name)
	name = strings.ReplaceAll(name, " ", "-")
	name = strings.ReplaceAll(name, "_", "-")
	return name
}

func (s *Scaffolder) normalizePackageName(name string) string {
	name = strings.ToLower(name)
	name = strings.ReplaceAll(name, " ", "")
	name = strings.ReplaceAll(name, "-", "")
	name = strings.ReplaceAll(name, "_", "")
	return name
}

func (s *Scaffolder) toCamelCase(name string) string {
	parts := strings.FieldsFunc(name, func(r rune) bool {
		return r == '-' || r == '_' || r == ' '
	})

	if len(parts) == 0 {
		return ""
	}

	result := strings.ToLower(parts[0])
	for i := 1; i < len(parts); i++ {
		if len(parts[i]) > 0 {
			result += strings.ToUpper(string(parts[i][0])) + strings.ToLower(parts[i][1:])
		}
	}

	return result
}

func (s *Scaffolder) toPascalCase(name string) string {
	parts := strings.FieldsFunc(name, func(r rune) bool {
		return r == '-' || r == '_' || r == ' '
	})

	var result strings.Builder
	for _, part := range parts {
		if len(part) > 0 {
			result.WriteString(strings.ToUpper(string(part[0])))
			if len(part) > 1 {
				result.WriteString(strings.ToLower(part[1:]))
			}
		}
	}

	return result.String()
}

func (s *Scaffolder) processTemplate(content string, vars *TemplateVars) (string, error) {
	tmpl, err := template.New("scaffold").Parse(content)
	if err != nil {
		return "", fmt.Errorf("failed to parse template: %w", err)
	}

	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, vars); err != nil {
		return "", fmt.Errorf("failed to execute template: %w", err)
	}

	return buf.String(), nil
}

func (s *Scaffolder) processTemplateFile(templatePath string, vars *TemplateVars) (string, error) {
	tmpl, err := template.ParseFiles(templatePath)
	if err != nil {
		return "", fmt.Errorf("failed to parse template file: %w", err)
	}

	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, vars); err != nil {
		return "", fmt.Errorf("failed to execute template: %w", err)
	}

	return buf.String(), nil
}

func (s *Scaffolder) processPathTemplate(path string, vars *TemplateVars) (string, error) {
	tmpl, err := template.New("path").Parse(path)
	if err != nil {
		return "", fmt.Errorf("failed to parse path template: %w", err)
	}

	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, vars); err != nil {
		return "", fmt.Errorf("failed to execute path template: %w", err)
	}

	return buf.String(), nil
}

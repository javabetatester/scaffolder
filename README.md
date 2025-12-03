# Scaffold - Go Project Generator

Scaffold é uma ferramenta CLI para gerar estruturas completas de projetos Go seguindo Clean Architecture, princípios SOLID e melhores práticas de desenvolvimento.

## Características

- **Clean Architecture**: Estrutura organizada em camadas (domain, usecase, interface, infrastructure)
- **Templates Completos**: REST API, gRPC service e CLI application
- **Observability**: Logging estruturado (Zap), métricas (Prometheus) e tracing (OpenTelemetry)
- **Security**: Validação de input, rate limiting, security headers e CORS
- **Database & Cache**: Suporte a PostgreSQL (GORM) e Redis
- **Testing**: Estrutura completa de testes unitários e de integração
- **DevOps**: Docker, Kubernetes e CI/CD (GitHub Actions)
- **Documentação**: README, CONTRIBUTING, LICENSE e CHANGELOG

## Instalação

```bash
go install github.com/bernardo/api/cmd/cli@latest
```

Ou clone o repositório e compile localmente:

```bash
git clone https://github.com/bernardo/api.git
cd API
go build -o scaffold ./cmd/cli
```

## Uso

### Gerar projeto REST API

```bash
scaffold scaffold rest my-api
```

Gera uma estrutura completa de REST API com:
- Gin framework
- Clean Architecture
- PostgreSQL e Redis
- Observability completa
- Testes
- Docker e Kubernetes

### Gerar projeto gRPC

```bash
scaffold scaffold grpc my-service
```

Gera uma estrutura completa de gRPC service com:
- Protocol Buffers
- Clean Architecture
- PostgreSQL e Redis
- Observability completa
- Testes
- Docker e Kubernetes

### Gerar projeto CLI

```bash
scaffold scaffold cli my-tool
```

Gera uma estrutura completa de CLI application com:
- Cobra commands
- Viper configuration
- Structured logging
- Testes
- Docker

## Estrutura Gerada

### REST API

```
my-api/
├── cmd/server/          # Entry point
├── internal/
│   ├── domain/          # Entities e interfaces
│   ├── usecase/         # Business logic
│   ├── interface/       # HTTP handlers
│   └── infrastructure/  # DB, cache, config
├── migrations/          # Database migrations
├── k8s/                 # Kubernetes manifests
├── .github/workflows/   # CI/CD
├── Dockerfile
├── docker-compose.yml
└── Makefile
```

### gRPC Service

```
my-service/
├── cmd/server/          # Entry point
├── proto/               # Protocol Buffers
├── internal/
│   ├── domain/          # Entities e interfaces
│   ├── usecase/         # Business logic
│   └── infrastructure/ # DB, cache, config, server
├── migrations/          # Database migrations
├── k8s/                 # Kubernetes manifests
├── .github/workflows/   # CI/CD
├── Dockerfile
├── docker-compose.yml
└── Makefile
```

### CLI Application

```
my-tool/
├── cmd/my-tool/         # Entry point
├── internal/
│   ├── cmd/             # Cobra commands
│   ├── config/          # Configuration
│   └── pkg/             # Shared packages
├── .github/workflows/   # CI/CD
├── Dockerfile
└── Makefile
```

## Features Implementadas

### Observability
- Structured logging com Zap
- Métricas Prometheus
- Distributed tracing com OpenTelemetry/Jaeger

### Security
- Input validation (validator/v10)
- Rate limiting
- Security headers
- CORS configurável

### Database & Cache
- PostgreSQL com GORM
- Redis para cache
- Database migrations
- Connection pooling

### Testing
- Testes unitários
- Testes de integração
- Coverage reports
- Mocks com testify

### DevOps
- Docker multi-stage builds
- Docker Compose para desenvolvimento
- Kubernetes manifests
- GitHub Actions CI/CD

## Requisitos

- Go 1.21+
- Docker (opcional, para desenvolvimento)
- Kubernetes (opcional, para deployment)

## Desenvolvimento

```bash
# Build
go build -o scaffold ./cmd/cli

# Testes
go test ./...

# Lint
golangci-lint run
```

## Contribuindo

Contribuições são bem-vindas! Por favor, leia o guia de contribuição antes de submeter PRs.

## Licença

MIT License - veja LICENSE para detalhes.

## Autor

Bernardo - Engenheiro de Software Sênior

# Sounio Package Registry

The official package registry for Sounio packages.

## Overview

The Sounio Package Registry is the central repository for discovering, publishing, and managing Sounio packages. It provides:

- **Package Discovery**: Search and browse packages by name, keywords, or category
- **Version Management**: Semantic versioning with dependency resolution
- **Authentication**: Secure API tokens for publishing
- **Statistics**: Download counts and usage metrics

## API Documentation

The registry API is documented using OpenAPI 3.1. See `openapi.yaml` for the complete specification.

### Base URL

- Production: `https://registry.sounio-lang.org/api/v1`
- Staging: `https://staging-registry.sounio-lang.org/api/v1`

### Quick Start

#### Search for packages

```bash
curl "https://registry.sounio-lang.org/api/v1/search?q=json"
```

#### Get package details

```bash
curl "https://registry.sounio-lang.org/api/v1/packages/json"
```

#### Download a package

```bash
curl -O "https://registry.sounio-lang.org/api/v1/packages/json/versions/1.0.0/download"
```

### Authentication

Most write operations require authentication. Create an API token at https://registry.sounio-lang.org/settings/tokens

```bash
# Using the sounio CLI
sounio login

# Using curl
curl -H "Authorization: Bearer YOUR_TOKEN" \
  https://registry.sounio-lang.org/api/v1/me
```

## Publishing Packages

### 1. Create a Sounio.toml

```toml
[package]
name = "my-package"
version = "0.1.0"
authors = ["Your Name <you@example.com>"]
description = "A brief description of my package"
license = "MIT"
repository = "https://github.com/you/my-package"
keywords = ["keyword1", "keyword2"]
categories = ["science", "math"]

[dependencies]
stdlib = "^1.0"
```

### 2. Login to the registry

```bash
sounio login
```

### 3. Publish

```bash
sounio publish
```

### Publishing Guidelines

- **Unique names**: Package names must be unique and follow `^[a-z][a-z0-9_-]*$`
- **Semantic versioning**: Use SemVer for version numbers
- **Documentation**: Include a README.md
- **License**: Specify a valid SPDX license identifier
- **No secrets**: Never include API keys, passwords, or private data

## Database Schema

The registry uses PostgreSQL 14+. See `schema.sql` for the complete database schema.

### Key Tables

| Table | Description |
|-------|-------------|
| `users` | User accounts |
| `packages` | Package metadata |
| `versions` | Package versions |
| `dependencies` | Version dependencies |
| `api_tokens` | API authentication tokens |
| `downloads` | Download tracking |

## Deployment

### Requirements

- PostgreSQL 14+
- Redis 6+ (for caching and rate limiting)
- S3-compatible storage (for package tarballs)
- Node.js 18+ or Rust runtime

### Environment Variables

```bash
# Database
DATABASE_URL=postgres://user:pass@localhost:5432/registry

# Redis
REDIS_URL=redis://localhost:6379

# Storage
S3_BUCKET=sounio-packages
S3_REGION=us-east-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...

# Auth
JWT_SECRET=your-secret-key
SESSION_SECRET=your-session-secret

# Email (for verification)
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USER=...
SMTP_PASS=...

# Optional
SENTRY_DSN=...
```

### Docker Deployment

```bash
# Build the image
docker build -t sounio-registry .

# Run with docker-compose
docker-compose up -d
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sounio-registry
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sounio-registry
  template:
    spec:
      containers:
      - name: registry
        image: sounio-registry:latest
        ports:
        - containerPort: 3000
        envFrom:
        - secretRef:
            name: registry-secrets
```

## Architecture

```
                    ┌─────────────────┐
                    │   Load Balancer │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
        ┌─────┴─────┐  ┌─────┴─────┐  ┌─────┴─────┐
        │  API Pod  │  │  API Pod  │  │  API Pod  │
        └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
   ┌─────┴─────┐      ┌─────┴─────┐      ┌─────┴─────┐
   │ PostgreSQL│      │   Redis   │      │    S3     │
   │  Primary  │      │  Cluster  │      │  Storage  │
   └───────────┘      └───────────┘      └───────────┘
```

## Rate Limiting

| Endpoint | Anonymous | Authenticated |
|----------|-----------|---------------|
| Search | 100/hour | 1000/hour |
| Download | 500/hour | 5000/hour |
| Publish | N/A | 60/hour |
| Other | 100/hour | 1000/hour |

## Security

### Best Practices

1. **API Tokens**: Use scoped tokens with minimal permissions
2. **Rotation**: Rotate tokens regularly
3. **Secrets**: Never commit tokens to version control
4. **HTTPS**: All API traffic is encrypted

### Reporting Vulnerabilities

Please report security vulnerabilities to security@sounio-lang.org

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for contribution guidelines.

## License

MIT License - see [LICENSE](../../LICENSE) for details.

# Deployment Guide

This guide covers deploying the HCT Prediction System to production environments.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Deployment](#local-deployment)
3. [Production Deployment](#production-deployment)
4. [Cloud Platforms](#cloud-platforms)
5. [SSL/HTTPS Configuration](#sslhttps-configuration)
6. [Monitoring](#monitoring)
7. [Backup & Recovery](#backup--recovery)

---

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 4 GB | 8+ GB |
| Storage | 20 GB | 50+ GB SSD |
| Docker | 20.10+ | Latest |
| Docker Compose | 2.0+ | Latest |

### Network Requirements

| Port | Service | Access |
|------|---------|--------|
| 80 | Frontend (HTTP) | Public |
| 443 | Frontend (HTTPS) | Public |
| 8001 | Backend API | Internal |
| 8000 | AI Service | Internal |
| 5432 | PostgreSQL | Internal |
| 6379 | Redis | Internal |

---

## Local Deployment

### Quick Start

```bash
cd Project

# Copy environment file
cp .env.example .env

# Edit environment variables
nano .env

# Build and start
docker-compose up --build -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### Verify Deployment

```bash
# Health checks
curl http://localhost/health
curl http://localhost:8001/health
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"age_at_hct": 45, "karnofsky_score": 80, "comorbidity_score": 2}'
```

---

## Production Deployment

### Environment Variables

```bash
# .env.production

# Database (use strong passwords!)
POSTGRES_USER=hct_prod_user
POSTGRES_PASSWORD=<GENERATE_STRONG_PASSWORD>
POSTGRES_DB=hct_production
DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}

# Redis
REDIS_URL=redis://redis:6379/0

# Security (generate with: openssl rand -hex 32)
SECRET_KEY=<GENERATE_64_CHAR_HEX_STRING>
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS (restrict to your domain)
ALLOWED_ORIGINS=https://yourdomain.com

# AI Service
AI_SERVICE_URL=http://ai_service:8000
MODEL_PATH=/app/models/trained_pipeline.pkl

# Logging
LOG_LEVEL=WARNING
```

### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.prod
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/ssl:/etc/nginx/ssl:ro
    restart: always
    depends_on:
      - backend

  backend:
    build: ./backend
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - SECRET_KEY=${SECRET_KEY}
    restart: always
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy

  ai_service:
    build: ./ai_service
    environment:
      - DATABASE_URL=${DATABASE_URL}
    volumes:
      - model_data:/app/models
    restart: always
    deploy:
      resources:
        limits:
          memory: 2G

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_DB=${POSTGRES_DB}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/init.sql:/docker-entrypoint-initdb.d/init.sql
    restart: always
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    restart: always
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
  redis_data:
  model_data:
```

### Deploy Commands

```bash
# Pull latest code
git pull origin main

# Build production images
docker-compose -f docker-compose.prod.yml build

# Deploy with zero downtime
docker-compose -f docker-compose.prod.yml up -d --no-deps backend
docker-compose -f docker-compose.prod.yml up -d --no-deps ai_service
docker-compose -f docker-compose.prod.yml up -d --no-deps frontend

# Run database migrations (if any)
docker-compose exec backend python -m alembic upgrade head
```

---

## Cloud Platforms

### AWS (ECS/Fargate)

```mermaid
flowchart TB
    subgraph AWS["AWS Cloud"]
        ALB["Application Load Balancer"]
        
        subgraph ECS["ECS Cluster"]
            FE["Frontend Task"]
            BE["Backend Task"]
            AI["AI Service Task"]
        end
        
        subgraph Data["Data Layer"]
            RDS[("RDS PostgreSQL")]
            EC[("ElastiCache Redis")]
            S3["S3 (Models)"]
        end
    end
    
    Internet --> ALB
    ALB --> FE
    ALB --> BE
    BE --> AI
    BE --> RDS
    BE --> EC
    AI --> S3
```

**Deployment Steps:**

1. Create ECR repositories for each service
2. Push Docker images to ECR
3. Create ECS cluster with Fargate
4. Configure task definitions
5. Set up Application Load Balancer
6. Create RDS PostgreSQL instance
7. Create ElastiCache Redis cluster
8. Configure security groups
9. Deploy services

### Azure (Container Apps)

```bash
# Login to Azure
az login

# Create resource group
az group create --name hct-rg --location eastus

# Create Container Apps environment
az containerapp env create \
  --name hct-env \
  --resource-group hct-rg \
  --location eastus

# Deploy services
az containerapp create \
  --name hct-backend \
  --resource-group hct-rg \
  --environment hct-env \
  --image your-registry.azurecr.io/hct-backend:latest \
  --target-port 8001 \
  --ingress external
```

### Google Cloud (Cloud Run)

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT_ID/hct-backend

# Deploy to Cloud Run
gcloud run deploy hct-backend \
  --image gcr.io/PROJECT_ID/hct-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

## SSL/HTTPS Configuration

### Using Let's Encrypt (Certbot)

```bash
# Install certbot
apt-get update
apt-get install certbot python3-certbot-nginx

# Get certificate
certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Auto-renewal (add to crontab)
0 0 * * * /usr/bin/certbot renew --quiet
```

### Nginx SSL Configuration

```nginx
# nginx/ssl.conf
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /etc/nginx/ssl/fullchain.pem;
    ssl_certificate_key /etc/nginx/ssl/privkey.pem;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;
    
    # HSTS
    add_header Strict-Transport-Security "max-age=63072000" always;

    location / {
        root /usr/share/nginx/html;
        try_files $uri $uri/ /index.html;
    }

    location /api {
        proxy_pass http://backend:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## Monitoring

### Health Check Endpoints

| Endpoint | Expected Response |
|----------|-------------------|
| `GET /health` | `{"status": "healthy"}` |
| `GET /health/db` | `{"database": "connected"}` |
| `GET /health/ai` | `{"model": "loaded"}` |

### Prometheus Metrics

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'hct-backend'
    static_configs:
      - targets: ['backend:8001']
    metrics_path: '/metrics'

  - job_name: 'hct-ai'
    static_configs:
      - targets: ['ai_service:8000']
    metrics_path: '/metrics'
```

### Logging (ELK Stack)

```yaml
# docker-compose.logging.yml
services:
  elasticsearch:
    image: elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
    volumes:
      - es_data:/usr/share/elasticsearch/data

  logstash:
    image: logstash:8.11.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf

  kibana:
    image: kibana:8.11.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch
```

---

## Backup & Recovery

### Database Backup

```bash
# Manual backup
docker-compose exec postgres pg_dump -U $POSTGRES_USER $POSTGRES_DB > backup_$(date +%Y%m%d).sql

# Automated backup script
#!/bin/bash
BACKUP_DIR=/backups
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Backup database
docker-compose exec -T postgres pg_dump -U $POSTGRES_USER $POSTGRES_DB | gzip > $BACKUP_DIR/db_$TIMESTAMP.sql.gz

# Backup models
tar -czf $BACKUP_DIR/models_$TIMESTAMP.tar.gz ./ai_service/models/

# Keep only last 7 days
find $BACKUP_DIR -type f -mtime +7 -delete

# Upload to S3 (optional)
aws s3 sync $BACKUP_DIR s3://your-bucket/backups/
```

### Recovery

```bash
# Restore database
gunzip -c backup_20251212.sql.gz | docker-compose exec -T postgres psql -U $POSTGRES_USER $POSTGRES_DB

# Restore models
tar -xzf models_20251212.tar.gz -C ./ai_service/
docker-compose restart ai_service
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Container won't start | Check logs: `docker-compose logs service_name` |
| Database connection failed | Verify `DATABASE_URL` and network |
| Model not loading | Check volume mounts and permissions |
| High memory usage | Increase container limits or optimize model |

### Debug Commands

```bash
# Check container status
docker-compose ps

# View logs
docker-compose logs -f --tail=100 backend

# Execute command in container
docker-compose exec backend bash

# Check resource usage
docker stats
```

---

**Last updated:** December 2025

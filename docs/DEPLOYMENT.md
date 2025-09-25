# Deployment Guide

## Overview

This guide covers deploying the FIAE AI Content Factory in various environments, from development to production.

## Prerequisites

- Docker and Docker Compose
- Python 3.8+ (for local development)
- Google Cloud credentials
- n8n instance (optional)

## Quick Start with Docker

### 1. Clone and Setup

```bash
git clone https://github.com/Maanik23/AI-Content-Generation-.git
cd AI-Content-Generation-
```

### 2. Configure Environment

```bash
cp env.example .env
# Edit .env with your configuration
```

### 3. Start Services

```bash
docker-compose up -d
```

### 4. Verify Deployment

```bash
curl http://localhost:8000/health
```

## Production Deployment

### Docker Compose Production

```yaml
version: '3.8'
services:
  ai-content-factory:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GOOGLE_CREDENTIALS_PATH=/app/credentials/google-credentials.json
    volumes:
      - ./credentials:/app/credentials:ro
      - ./chroma_db:/app/chroma_db
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-content-factory
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-content-factory
  template:
    metadata:
      labels:
        app: ai-content-factory
    spec:
      containers:
      - name: ai-content-factory
        image: ai-content-factory:latest
        ports:
        - containerPort: 8000
        env:
        - name: GOOGLE_CREDENTIALS_PATH
          value: "/app/credentials/google-credentials.json"
        volumeMounts:
        - name: credentials
          mountPath: /app/credentials
          readOnly: true
        - name: chroma-db
          mountPath: /app/chroma_db
      volumes:
      - name: credentials
        secret:
          secretName: google-credentials
      - name: chroma-db
        persistentVolumeClaim:
          claimName: chroma-db-pvc
```

## Environment-Specific Configurations

### Development

```bash
# .env.development
DEBUG=true
LOG_LEVEL=DEBUG
ENABLE_MONITORING=false
MAX_CONCURRENT_JOBS=2
```

### Staging

```bash
# .env.staging
DEBUG=false
LOG_LEVEL=INFO
ENABLE_MONITORING=true
MAX_CONCURRENT_JOBS=5
BUDGET_LIMIT=50.0
```

### Production

```bash
# .env.production
DEBUG=false
LOG_LEVEL=WARNING
ENABLE_MONITORING=true
MAX_CONCURRENT_JOBS=10
BUDGET_LIMIT=1000.0
AUTO_STOP_SERVICES=true
```

## Security Considerations

### 1. Credentials Management

```bash
# Use Kubernetes secrets
kubectl create secret generic google-credentials \
  --from-file=google-credentials.json=./credentials/google-credentials.json

# Or use external secret management
# AWS Secrets Manager, Azure Key Vault, etc.
```

### 2. Network Security

```yaml
# nginx.conf
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://ai-content-factory:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 3. SSL/TLS Configuration

```yaml
# docker-compose.prod.yml
services:
  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - ai-content-factory
```

## Monitoring and Observability

### 1. Prometheus Metrics

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'ai-content-factory'
    static_configs:
      - targets: ['ai-content-factory:8000']
    metrics_path: '/monitoring/metrics'
    scrape_interval: 30s
```

### 2. Grafana Dashboard

```json
{
  "dashboard": {
    "title": "AI Content Factory",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])"
          }
        ]
      }
    ]
  }
}
```

### 3. Log Aggregation

```yaml
# docker-compose.logging.yml
services:
  ai-content-factory:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
  
  elasticsearch:
    image: elasticsearch:7.14.0
    environment:
      - discovery.type=single-node
  
  kibana:
    image: kibana:7.14.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch
```

## Scaling Strategies

### Horizontal Scaling

```yaml
# docker-compose.scale.yml
services:
  ai-content-factory:
    deploy:
      replicas: 5
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G
```

### Load Balancing

```yaml
# nginx-lb.conf
upstream ai_content_factory {
    server ai-content-factory-1:8000;
    server ai-content-factory-2:8000;
    server ai-content-factory-3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://ai_content_factory;
    }
}
```

## Database Management

### ChromaDB Persistence

```yaml
# docker-compose.db.yml
services:
  ai-content-factory:
    volumes:
      - chroma-data:/app/chroma_db
  
  chroma-backup:
    image: postgres:13
    volumes:
      - chroma-data:/backup
    command: |
      sh -c "
        while true; do
          tar -czf /backup/chroma-$(date +%Y%m%d-%H%M%S).tar.gz /app/chroma_db
          sleep 86400
        done
      "

volumes:
  chroma-data:
```

## Backup and Recovery

### Automated Backups

```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d-%H%M%S)
BACKUP_DIR="/backups"

# Backup ChromaDB
docker exec ai-content-factory tar -czf /tmp/chroma-${DATE}.tar.gz /app/chroma_db
docker cp ai-content-factory:/tmp/chroma-${DATE}.tar.gz ${BACKUP_DIR}/

# Backup configuration
cp .env ${BACKUP_DIR}/env-${DATE}.backup

# Cleanup old backups (keep 30 days)
find ${BACKUP_DIR} -name "*.tar.gz" -mtime +30 -delete
```

### Recovery Process

```bash
#!/bin/bash
# restore.sh
BACKUP_FILE=$1

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup-file.tar.gz>"
    exit 1
fi

# Stop services
docker-compose down

# Restore ChromaDB
docker run --rm -v ai-content-factory_chroma-data:/data -v $(pwd):/backup alpine tar -xzf /backup/${BACKUP_FILE} -C /data

# Start services
docker-compose up -d
```

## Performance Optimization

### 1. Resource Limits

```yaml
services:
  ai-content-factory:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

### 2. Caching Strategy

```python
# Redis caching
CACHE_CONFIG = {
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': 'redis://localhost:6379/0',
    'CACHE_DEFAULT_TIMEOUT': 300
}
```

### 3. Database Optimization

```python
# ChromaDB optimization
CHROMA_SETTINGS = {
    'collection_metadata': {
        'hnsw:space': 'cosine',
        'hnsw:construction_ef': 200,
        'hnsw:M': 16
    }
}
```

## Troubleshooting

### Common Issues

1. **Memory Issues**
   ```bash
   # Check memory usage
   docker stats ai-content-factory
   
   # Increase memory limits
   docker-compose up -d --scale ai-content-factory=2
   ```

2. **Database Connection Issues**
   ```bash
   # Check ChromaDB status
   docker exec ai-content-factory ls -la /app/chroma_db
   
   # Rebuild database
   docker exec ai-content-factory rm -rf /app/chroma_db/*
   ```

3. **API Timeout Issues**
   ```bash
   # Check logs
   docker logs ai-content-factory
   
   # Increase timeout
   export TIMEOUT_SECONDS=600
   ```

### Health Checks

```bash
# Basic health check
curl -f http://localhost:8000/health

# Detailed health check
curl -f http://localhost:8000/monitoring/health

# Metrics check
curl -f http://localhost:8000/monitoring/metrics
```

## Maintenance

### Regular Tasks

1. **Log Rotation**
   ```bash
   # Configure logrotate
   echo "/var/lib/docker/containers/*/*.log {
       daily
       rotate 7
       compress
       delaycompress
       missingok
       notifempty
   }" > /etc/logrotate.d/docker
   ```

2. **Database Cleanup**
   ```bash
   # Clean old ChromaDB data
   docker exec ai-content-factory find /app/chroma_db -name "*.tmp" -mtime +7 -delete
   ```

3. **Security Updates**
   ```bash
   # Update base images
   docker-compose pull
   docker-compose up -d
   ```

## Support

For deployment issues:
- Check the [troubleshooting section](#troubleshooting)
- Review application logs
- Create an [issue](https://github.com/Maanik23/AI-Content-Generation-/issues)
- Contact the development team

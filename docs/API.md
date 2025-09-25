# API Documentation

## Overview

The FIAE AI Content Factory provides a comprehensive REST API for AI-powered content generation and processing.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API uses API key authentication. Include your API key in the request headers:

```http
Authorization: Bearer your-api-key
```

## Endpoints

### Health & Monitoring

#### GET /health
Basic health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "timestamp": "2024-12-19T10:30:00Z",
  "dependencies": {
    "google_drive": "healthy",
    "google_sheets": "healthy",
    "elevenlabs": "healthy"
  }
}
```

#### GET /monitoring/health
Detailed health check with system metrics.

#### GET /monitoring/metrics
Get comprehensive system performance metrics.

### Document Processing

#### POST /process-document
Main document processing endpoint with RAG enhancement.

**Request Body:**
```json
{
  "file_path": "path/to/document.docx",
  "content": "Document content as string",
  "job_id": "optional-job-id"
}
```

**Response:**
```json
{
  "success": true,
  "job_id": "job_20241219_103000",
  "message": "Document processed successfully with RAG enhancement",
  "data": {
    "knowledge_analysis": "# Wissensanalyse\n...",
    "use_case_text": "# Praktische Anwendungsfälle\n...",
    "quiz_text": "# Bewertungsfragen\n...",
    "script_text": "# Video-Skript\n..."
  }
}
```

#### POST /process-document-rag
RAG-enhanced document processing.

#### POST /process-document-orchestrated
LangGraph orchestrated processing with intelligent workflow management.

#### POST /process-document-advanced
Advanced multi-modal document processing with semantic chunking.

### Human-in-the-Loop (HITL)

#### GET /hitl/pending-approvals
Get all pending approval requests.

#### POST /hitl/approve/{approval_id}
Approve generated content.

#### POST /hitl/reject/{approval_id}
Reject content with revision requests.

### Content Intelligence

#### GET /content-intelligence/patterns
Analyze content patterns and structures.

#### GET /content-intelligence/quality-prediction
Predict content quality using ML models.

#### GET /content-intelligence/analytics
Get performance analytics and insights.

### Production Monitoring

#### GET /production-monitor/status
Get comprehensive production system status.

#### GET /production-monitor/metrics
Get production metrics summary.

#### GET /production-monitor/alerts
Get production alerts with filtering options.

## Error Handling

The API uses standard HTTP status codes:

- `200` - Success
- `400` - Bad Request
- `401` - Unauthorized
- `404` - Not Found
- `500` - Internal Server Error
- `503` - Service Unavailable

**Error Response Format:**
```json
{
  "success": false,
  "error": "Error message",
  "details": "Additional error details"
}
```

## Rate Limiting

- Default: 60 requests per minute
- Configurable via environment variables
- Headers include rate limit information

## Examples

### Process a Document

```bash
curl -X POST "http://localhost:8000/process-document" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Your document content here",
    "job_id": "my-job-123"
  }'
```

### Get System Health

```bash
curl -X GET "http://localhost:8000/monitoring/health"
```

### Approve Content

```bash
curl -X POST "http://localhost:8000/hitl/approve/approval-123" \
  -H "Content-Type: application/json" \
  -d '{
    "approved_by": "reviewer@company.com",
    "notes": "Content looks great!"
  }'
```

## SDK Examples

### Python

```python
import requests

# Process document
response = requests.post(
    "http://localhost:8000/process-document",
    json={
        "content": "Your document content",
        "job_id": "python-job-123"
    }
)

result = response.json()
print(f"Job ID: {result['job_id']}")
```

### JavaScript

```javascript
// Process document
const response = await fetch('http://localhost:8000/process-document', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    content: 'Your document content',
    job_id: 'js-job-123'
  })
});

const result = await response.json();
console.log('Job ID:', result.job_id);
```

## WebSocket Support

Real-time updates are available via WebSocket connections for:
- Job progress updates
- System status changes
- Alert notifications

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Update:', data);
};
```

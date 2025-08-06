### File: docs/deployment_strategy.md

```markdown
# استراتژی استقرار سیستم
# Deployment Strategy

## استراتژی کلی استقرار

### محیط‌های استقرار:
1. **Development**: Local Docker Compose
2. **Staging**: Docker Swarm / Single VM
3. **Production**: Kubernetes cluster

## Phase 1: Development Environment

### Docker Compose Setup
```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    environment:
      - ENV=development
    volumes:
      - ./src:/app/src
    ports:
      - "8000:8000"
  
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=legal_ai
      - POSTGRES_USER=legal_ai_user
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  chromadb:
    image: chromadb/chroma:latest
    volumes:
      - chroma_data:/chroma/chroma
    ports:
      - "8001:8000"
  
  streamlit:
    build: ./streamlit_app
    ports:
      - "8501:8501"
    depends_on:
      - backend

volumes:
  postgres_data:
  chroma_data:
```

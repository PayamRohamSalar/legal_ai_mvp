# معماری سیستم دستیار هوشمند حقوقی

# System Architecture - Legal AI Assistant

## 1. نمای کلی و سبک معماری

### Overview & Architectural Style

این سیستم بر اساس معماری سرویس‌گرا (Service-Oriented Architecture) با رابط کاربری مجزا طراحی شده است. معماری کلی از الگوی Microservices الهام گرفته و برای مقیاس‌پذیری و نگهداری آسان بهینه شده است.

### نمودار معماری سطح بالا

```mermaid
graph TB
    subgraph "Client Layer"
        UI[Streamlit UI / React Frontend]
    end
  
    subgraph "API Gateway"
        API[FastAPI Backend]
    end
  
    subgraph "Core Services"
        RAG[RAG Core Service]
        QP[Query Processor]
        RG[Response Generator]
    end
  
    subgraph "Data Layer"
        VDB[(ChromaDB<br/>Vector Store)]
        PG[(PostgreSQL<br/>Metadata)]
        CACHE[(Redis Cache)]
    end
  
    subgraph "External Services"
        LLM[LLM Service<br/>Gemini/Local]
    end
  
    subgraph "Data Pipeline"
        ING[Document Ingestion Service]
        EMB[Embedding Service]
    end
  
    UI --> API
    API --> RAG
    API --> QP
    RAG --> RG
    RAG --> VDB
    RAG --> PG
    RAG --> LLM
    QP --> CACHE
    ING --> EMB
    EMB --> VDB
    ING --> PG
```

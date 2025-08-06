# دستیار هوشمند حقوقی برای حوزه پژوهش و فناوری ایران

# Legal AI Assistant for Iran's Research and Technology Domain

An intelligent system based on RAG (Retrieval-Augmented Generation) to provide accurate, citation-backed answers to questions about Iranian laws in the research and technology sector.

## 🎯 Project Goal

To create a specialized AI assistant that can accurately answer legal questions related to Iran's research and technology laws and regulations, providing precise citations to the source legal articles.

## Core Principles

This system operates under a strict set of principles to ensure accuracy and reliability:

- **Source-Bound:** Answers are based exclusively on the provided legal documents and do not use any external knowledge.
- **Citation-First:** Prioritizes direct quotes from legal texts with precise, granular citations to the source article, section, and clause.
- **Specialized Tool:** Acts as a legal text retrieval assistant designed for researchers and professionals, not as a legal advisor providing opinions or interpretations.

## 🛠️ Tech Stack

- **Backend**: Python 3.11, FastAPI
- **Data & AI**: LangChain, Transformers, Sentence-Transformers
- **Databases**:
  - **Relational**: PostgreSQL (for metadata and structured data)
  - **Vector**: ChromaDB (for semantic search)
- **Caching**: Redis
- **Frontend (MVP)**: Streamlit
- **Containerization & DevOps**: Docker, Docker Compose, GitHub Actions

## 🚀 Getting Started

Follow these steps to set up and run the complete development environment locally.

### Prerequisites

- Docker and Docker Compose

### 1. Clone the Repository

```bash
git clone [https://github.com/your-org/legal-ai-assistant.git](https://github.com/your-org/legal-ai-assistant.git)
cd legal-ai-assistant
```

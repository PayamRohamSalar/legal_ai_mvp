### File: docs/technology_stack.md

```markdown
# توجیه انتخاب پشته فناوری
# Technology Stack Justification

## زبان برنامه‌نویسی: Python 3.11+

### دلایل انتخاب:
- **اکوسیستم غنی NLP و AI**: کتابخانه‌های قدرتمند مانند LangChain، Transformers
- **پشتیبانی عالی از زبان فارسی**: کتابخانه‌های hazm و parsivar
- **سرعت توسعه**: syntax ساده و خوانا
- **جامعه فعال**: پشتیبانی و منابع آموزشی فراوان

### جایگزین‌های بررسی شده:
- **Node.js**: محدودیت در کتابخانه‌های NLP
- **Go**: عدم بلوغ اکوسیستم AI
- **Java**: پیچیدگی بیشتر، سرعت توسعه کمتر

## Backend Framework: FastAPI

### دلایل انتخاب:
- **Performance**: مبتنی بر Starlette و Pydantic، بسیار سریع
- **Type Safety**: پشتیبانی native از type hints
- **Auto Documentation**: تولید خودکار OpenAPI/Swagger
- **Async Support**: پشتیبانی کامل از async/await
- **Modern**: طراحی مدرن و استانداردهای جدید

### مقایسه با جایگزین‌ها:
| ویژگی | FastAPI | Flask | Django |
|-------|---------|-------|--------|
| سرعت | بسیار بالا | متوسط | متوسط |
| Async | Native | محدود | محدود |
| Type Checking | Native | Plugin | محدود |
| Learning Curve | متوسط | آسان | سخت |
| Auto Docs | بله | خیر | محدود |

## Vector Database: ChromaDB

### دلایل انتخاب:
- **سادگی**: راه‌اندازی آسان، مناسب MVP
- **Embedded Mode**: امکان اجرا بدون سرور جداگانه
- **Python Native**: طراحی شده برای Python
- **LangChain Integration**: یکپارچگی عالی
- **Metadata Filtering**: پشتیبانی قوی از فیلترینگ

### Roadmap:
1. **MVP**: ChromaDB (Embedded)
2. **Scale**: ChromaDB (Server Mode)
3. **Enterprise**: مهاجرت به Qdrant/Weaviate

## Relational Database: PostgreSQL

### دلایل انتخاب:
- **ACID Compliance**: تضمین یکپارچگی داده‌ها
- **JSON Support**: ذخیره metadata پیچیده
- **Full-text Search**: قابلیت جستجوی متنی (backup)
- **Scalability**: امکان scale vertical و horizontal
- **Extensions**: PostGIS, pgvector برای آینده

## LLM Strategy: Hybrid Approach

### Phase 1 - Google Gemini API:
- **سرعت راه‌اندازی**: بدون نیاز به GPU
- **کیفیت بالا**: مدل‌های state-of-the-art
- **پشتیبانی فارسی**: عملکرد خوب با زبان فارسی

### Phase 2 - Local Model Option:
- **حریم خصوصی**: داده‌ها سازمان را ترک نمی‌کنند
- **کنترل کامل**: امکان fine-tuning
- **هزینه ثابت**: بدون هزینه per-request

### معماری انتزاعی:
```python
# رابط یکسان برای هر دو approach
class LLMInterface:
    def generate(prompt: str) -> str:
        if config.use_api:
            return api_llm.generate(prompt)
        else:
            return local_llm.generate(prompt)
```

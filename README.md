# SQLFriend

SQLFriend — инструмент автоматического перевода запросов с естественного языка в валидный SQL-код (Text-to-SQL) на основе LLM с поддержкой связывания схем и автоматической коррекцией ошибок.

## Архитектура и методы

* **Schema Linking**: Динамическое извлечение структуры БД и типов данных для формирования контекста LLM.
* **Prompt Engineering**: Применение Few-Shot и Chain-of-Thought подходов для повышения точности генерации.
* **SQL Sandbox & Validation**: Модуль синтаксической проверки и безопасного исполнения запросов.
* **Self-Correction Loop**: Автоматический контур исправления невалидных SQL-запросов на основе сообщений об ошибках СУБД.

## Стек технологий

* Python
* FastAPI / Streamlit
* OpenAI API / LangChain
* SQLite / PostgreSQL
* Pydantic

## Установка и запуск

1. Клонирование репозитория:
```bash
git clone [https://github.com/ArtemCh101/SQLFriend.git](https://github.com/ArtemCh101/SQLFriend.git)
cd SQLFriend
```
2. Создание и активация виртуального окружения
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
3. Настройка конфигурации
```bash
OPENAI_API_KEY=your_api_key_here
DATABASE_URL=sqlite:///example.db
```
4. Запуск приложения
```bash
streamlit run app.py
```
## Real‑time NLP Triage (FastAPI + WebSockets + Streamlit)

A real‑time triage system that analyzes incoming text for **sentiment, intent, toxicity, and entities**, stores results in **SQLite**, and visualizes them in a **Streamlit dashboard**.  
It supports both **REST** and **WebSocket** clients and ships with **Docker Compose** for easy, reproducible runs.

---

## Features

- **FastAPI** backend
  - `/analyze` (REST): submit one message and get full NLP results
  - `/ws` (WebSocket): stream multiple messages, receive incremental updates + final results
  - `/client` and `/client-ws`: simple built‑in web UIs for quick testing
  - `/search` + `/metrics`: query + aggregate stored results
  - `/health` and redirect from `/` → `/docs` (OpenAPI UI)

- **Streamlit Dashboard** (`:8501`)
  - Live charts (sentiment & intent distributions, toxicity trend)
  - Filters (keyword, sentiment, intent, toxicity, language)
  - “Latest messages” table

- **Persistence**
  - SQLite database stored on a **Docker volume** 
  - Hugging Face model cache stored on a **Docker volume**

---




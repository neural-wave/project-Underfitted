# Swisscom Customer Support Chatbot

Built for the **NeuralWave 2024 Hackathon** in partnership with Swisscom — a RAG-based conversational assistant that answers customer support queries grounded in Swisscom's internal documentation.
---

## How it works

Standard single-stage similarity search degrades on domain-specific jargon — Swisscom's documentation uses Swiss telecom product names that embedding similarity handles poorly. The solution is a **two-stage retrieval pipeline**:

```
User query
    │
    ▼
VoyageAI voyage-3 embeddings → ChromaDB similarity search → top 50 candidates
    │
    ▼
VoyageAI rerank-2 → top 10 reranked documents
    │
    ▼
GPT-4o + conversation history → response with source links
```

Reranking at the second stage significantly improves precision: the first stage casts a wide net (top 50), the reranker selects the 10 most relevant before passing context to the LLM.

## Tech stack

| Layer | Technology |
|---|---|
| Embeddings | VoyageAI `voyage-3` |
| Vector store | ChromaDB |
| Reranking | VoyageAI `rerank-2` |
| LLM | OpenAI GPT-4o |
| Orchestration | LangChain |
| Interface | Flask |

## Setup

Requires Python 3.10+, an OpenAI API key, and a VoyageAI API key.

```bash
git clone https://github.com/neural-wave/project-Underfitted.git
cd project-Underfitted
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:
```
OPENAI_API_KEY=your-key
VOYAGE_API_KEY=your-key
```

Update the dataset paths in `main.py` to point to your document collection, then:
```bash
python main.py
# Open http://127.0.0.1:5000
```

## Project structure

```
project-Underfitted/
├── Swisscom_chatbot/src/
│   ├── data/        # document loading and preprocessing
│   ├── retrieve/    # ChromaDB vector store + two-stage retriever
│   └── llm/         # GPT-4o prompt builder and request handler
├── templates/       # Flask chat UI
├── docs/assignment/ # original hackathon brief
├── main.py          # Flask entrypoint
└── requirements.txt
```

## Team

Matteo Vitali · Luigi Tisci · Paolo Deidda · Diell Kryeziu  
*NeuralWave Hackathon 2024 — Team Underfitted*

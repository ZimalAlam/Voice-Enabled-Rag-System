# Agentic RAG System

A full-stack **Retrieval-Augmented Generation (RAG)** system with voice input, multimodal document processing, and multi-source search. Built with **FastAPI** and **Next.js**, it provides real-time AI answers with visual grounding and citations.

---

## Key Features

- **Voice Queries:** Streaming speech-to-text, real-time transcription, multi-language support.  
- **Multimodal Documents:** PDF text/images, OCR, charts, diagrams, context-aware chunking.  
- **Intelligent Search:** Local vector search (ChromaDB), web search (SERP API), Google Drive integration, parallel querying.  
- **Citations & Transparency:** Source tracking, visual references, confidence scores.  
- **Real-time Responses:** WebSocket communication, streaming answers, concurrent processing.  

---

## Architecture

**Frontend (Next.js) ↔ Backend (FastAPI) ↔ External Services**

| Frontend UI        | Backend Engine      | External Services           |
|-------------------|------------------|-----------------------------|
| Voice & Chat UI    | RAG Engine        | Gemini / Claude            |
| Citations Display  | STT Service       | Google Drive               |
| Image Viewer       | Document Processing | SERP API / ChromaDB       |

---

## Technology Stack

### Backend
- **Frameworks & APIs:** FastAPI, Google Gemini / Claude, ChromaDB  
- **Speech-to-Text:** Google Cloud Speech-to-Text  
- **Document Processing:** PyPDF2, pytesseract, Pillow  
- **Search APIs:** SERP API, Google Drive API  

### Frontend
- **Frameworks & Libraries:** Next.js, React 18, TypeScript, Tailwind CSS  
- **State Management:** Zustand  
- **Audio & Communication:** Web Audio API, WebSocket support  

### AI & ML
- Sentence-transformers embeddings  
- Vision models for image understanding  
- Context-aware text generation  
- Similarity search and confidence scoring  

---

## Quick Start

### Clone Repository
git clone <repo-url>
cd agentic-rag


### Backend Setup
cd backend
pip install -r requirements.txt
cp env.example .env
uvicorn main:app --reload

### Frontend Setup
cd frontend
npm install
npm run dev



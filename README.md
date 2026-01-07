# 🏗️ Construction RAG Assistant

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-green.svg)](https://faiss.ai/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![OpenRouter](https://img.shields.io/badge/LLM-OpenRouter-purple.svg)](https://openrouter.ai/)

An AI-powered Retrieval-Augmented Generation (RAG) system for construction industry documents. Implements semantic search using FAISS vector database with TF-IDF embeddings and OpenRouter LLM integration.

## 📋 Assignment Requirements Met

✅ **FAISS** for vector search (preferred method)  
✅ **Open-source embeddings** (TF-IDF from scikit-learn)  
✅ **Semantic retrieval** with similarity scoring  
✅ **LLM integration** via OpenRouter  
✅ **Transparent results** with source citations  

## Deployment Link
https://construction-rag-assistant-hksdxwpyeapeq9vdnlx4kz.streamlit.app/

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/construction-rag-assistant.git
cd construction-rag-assistant

### Create virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

## to upgrade pip
python.exe -m pip install --upgrade pip

##Install dependencies
pip install -r requirements.txt

## Test the RAG pipeline
python rag_pipeline.py

##Launch the web interface
streamlit run app.py

Open your browser and go to http://localhost:8501

🏗️ Features
🔍 Semantic Search
FAISS vector database for fast similarity search

TF-IDF embeddings (512 dimensions) for document representation

Cosine similarity scoring with relevance percentages

Top-k retrieval with configurable result count

🤖 AI-Powered Answers
OpenRouter integration with Mistral 7B (free tier)

Context-aware response generation grounded in retrieved documents

Source citation and transparency

Fallback to document extraction mode when LLM is unavailable

📊 Interactive Dashboard
Real-time metrics and performance statistics

Similarity score visualizations with Plotly charts

Expandable document previews with metadata

Query history and example queries

System status monitoring

🔧 Technical Implementation
Vector Database: FAISS
Type: IndexFlatL2 (exact search)

Dimension: 512 (TF-IDF features)

Distance Metric: L2 (Euclidean distance)

Search Method: Exact similarity search

Normalization: TF-IDF vector normalization

Embeddings: TF-IDF
Library: scikit-learn's TfidfVectorizer

Features: 512 dimensions

Processing: Stopword removal, n-grams (1-2)

Advantages: Lightweight, no external model dependencies

Performance: Fast training and inference

LLM Integration: OpenRouter
Model: Mistral 7B Instruct (free tier available)

API: OpenAI-compatible interface

Prompt Engineering: Strict context grounding

Temperature: 0.1 for consistent, factual responses

Fallback: Document extraction when API unavailable

Document Processing Pipeline
Chunking: Semantic paragraph splitting

Cleaning: Text normalization and preprocessing

Embedding: TF-IDF vector transformation

Indexing: FAISS vector storage

Retrieval: Similarity search with scoring

Generation: LLM answer synthesis

📚 Document Processing
The system processes three construction documents:

1. Company Overview (doc1_company.md)
Construction packages and pricing

Basic, Standard, and Premium options

Package durations and inclusions

Target audience for each package

2. Quality & Safety (doc2_quality.md)
Quality assurance processes

Safety protocols and standards

Material certification requirements

Warranty terms and conditions

3. Process & Journey (doc3_process.md)
Customer journey workflow

Consultation to handover timeline

Client approval processes

Post-construction support

🎯 Example Queries
Try these example queries to test the system:

"What construction packages are available?"

"How much does the basic package cost?"

"What quality assurance measures exist?"

"Describe the customer journey process"

"What safety protocols are in place?"

"How long does premium package construction take?"

"What is included in the standard package?"

"What warranty terms are offered?"

📊 Performance Metrics
Search Speed: < 100ms for typical queries

Accuracy: High relevance through semantic search

Scalability: FAISS supports millions of vectors

Transparency: Full similarity scores and sources shown

LLM Latency: ~2-3 seconds for answer generation

🖥️ User Interface
Dashboard Features
Real-time Search: Instant document retrieval

Visual Metrics: Performance statistics and charts

Result Preview: Expandable document chunks

Source Tracking: Document origin and relevance scores

System Status: Live monitoring of RAG components"# construction-rag-assistant" 

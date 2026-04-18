# ColPali Multimodal RAG — ESA & AI Research Papers QA System

> DSAI 413 Assignment 1 | Zewail City of Science and Technology

A **Multimodal Retrieval-Augmented Generation** system that treats PDF pages as images and uses ColPali's late-interaction matching to retrieve visually rich document pages, then feeds them to a Vision Language Model for grounded, citation-backed answers.

## Architecture

```
PDF corpus (ESA + AI Research Papers)
    │
    ▼  pdf2image
Page images (PIL)
    │
    ▼  ColPali encoder (vidore/colpali-v1.3)
Multi-vector patch embeddings
    │
    ▼  FAISS index (offline) + Text keyword index
─────────────────────────────────────────────────
    ▲  User query
    │  ColPali query encoder
    │  MaxSim late interaction + keyword boost (hybrid)
Top-K page images + modality metadata
    │
    ▼  VLM (Llama 4 Scout via Groq API)
Grounded answer + page citations
    │
    ▼  Streamlit chat UI
```

## Why ColPali?

Traditional PDF RAG pipelines require: OCR → layout detection → chunking → text embedding → retrieval.
ColPali skips all of that by **treating every page as an image** and encoding it directly via a VLM (PaliGemma 3B). It produces multi-vector (patch-level) embeddings that capture both text and visual structure simultaneously.

Key advantages:
- No OCR needed — works on scanned PDFs, tables, charts, figures
- Late interaction (ColBERT-style) MaxSim scoring preserves fine-grained token-patch alignment
- End-to-end trainable, SOTA on ViDoRe benchmark
- Hybrid retrieval: ColPali visual scores + text keyword boost

## Dataset

**ESA Space Environment Report + AI Research Papers** — 5 publicly available PDFs, 95 pages total.

| File | Source | Content |
|------|--------|---------|
| `esa_space_environment.pdf` | ESA / sdo.esoc.esa.int | Debris charts, launch statistics, compliance tables, orbital maps |
| `attention_paper.pdf` | arXiv:1706.03762 | Transformer architecture figures, BLEU score tables, attention diagrams |
| `gpt4_paper.pdf` | arXiv:2303.08774 | Exam benchmark tables, safety evaluation charts |
| `mistral_paper.pdf` | arXiv:2310.06825 | Sliding window attention figures, benchmark comparison tables |
| `colpali_paper.pdf` | arXiv:2407.01449 | ViDoRe benchmark results, retrieval pipeline diagrams |

This corpus was chosen for its **modality diversity**: every document contains a different mix of text, tables, figures, charts, and equations — ideal for stress-testing multimodal retrieval. It is also distinct from commonly used datasets (SEC filings, WHO reports) used by other students.

## Setup

```bash
# 1. Clone / navigate to project
cd DSAI-413-A1

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set API key (free at console.groq.com)
cp .env.example .env
# Edit .env: add GROQ_API_KEY=gsk_...

# 5. Download PDFs
python src/download_dataset.py

# 6. Build the index (GPU recommended; CPU works but slower)
python src/build_index.py

# 7. Extract modality metadata (fast, CPU only)
python src/metadata_extractor.py

# 8. Launch the app
streamlit run app.py
```

## Project Structure

```
DSAI-413-A1/
├── app.py                        # Streamlit chat interface
├── requirements.txt
├── .env.example
├── README.md
├── src/
│   ├── ingestion.py              # PDF → page images (pdf2image, no OCR)
│   ├── indexer.py                # ColPali encoding + FAISS multi-vector index
│   ├── retriever.py              # Hybrid retrieval: MaxSim + keyword boost
│   ├── generator.py              # VLM answer generation (Groq / Llama 4 Scout)
│   ├── metadata_extractor.py     # Table/figure/chart detection per page
│   ├── build_index.py            # Offline indexing entry point
│   └── download_dataset.py       # Dataset downloader
├── data/
│   ├── pdfs/                     # Source PDFs
│   └── index/
│       ├── colpali.faiss         # FAISS patch embedding index
│       ├── index_meta.pkl        # Page map + patch counts
│       ├── metadata.json         # Page records (doc_id, pdf_name, page_number)
│       ├── page_metadata.json    # Modality tags per page (table/figure/chart)
│       ├── text_index.json       # Keyword inverted index
│       └── thumbnails/           # Page image thumbnails for UI
├── evaluation/
│   ├── benchmark_queries.json    # 15 annotated queries with ground truth
│   └── run_eval.py               # Recall@K, nDCG@5, latency, modality breakdown
└── notebooks/
    ├── exploration.ipynb         # Ad-hoc retrieval + visualization
    └── build_index_colab.ipynb   # Colab notebook for GPU indexing
```

## Evaluation

```bash
python evaluation/run_eval.py
```

Reports the following metrics across 15 annotated benchmark queries:

| Metric | Description |
|--------|-------------|
| Recall@1 | Is the top result relevant? |
| Recall@5 | Is any of the top-5 results relevant? |
| nDCG@5 | Ranked quality of top-5 results |
| Mean latency | End-to-end retrieval time (ms) |
| By modality | nDCG@5 broken down by text / table / chart / figure |

## Multi-modal Coverage

Each retrieved page is tagged with detected modalities:

| Badge | Meaning |
|-------|---------|
| 📝 text | Primarily text content |
| 📊 table | Page contains tabular data |
| 🖼 figure | Page contains figures or diagrams |
| 📈 chart | Page contains charts or plots |

Detection uses `pypdf` text extraction + regex patterns for table captions (`Table X`), figure captions (`Figure X`, `Fig. X`), chart keywords, and equation markers.

## Grading Checklist

| Criterion | Implementation | File |
|-----------|----------------|------|
| Multi-modal ingestion | pdf2image, no OCR, page-as-image | `src/ingestion.py` |
| Vector index | FAISS IndexFlatIP, ColPali multi-vector patches | `src/indexer.py` |
| Smart chunking | Page-level + patch-level (ColPali internal) | `src/indexer.py` |
| Multi-modal coverage | Table/figure/chart detection + modality tags | `src/metadata_extractor.py` |
| QA chatbot | Streamlit + Llama 4 Scout (Groq) vision | `app.py` |
| Source attribution | Page number + PDF filename + modality badge | `app.py` |
| Evaluation suite | Recall@K, nDCG@5, latency, modality breakdown | `evaluation/run_eval.py` |
| Hybrid retrieval | ColPali visual + keyword boost | `src/retriever.py` |
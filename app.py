import os
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent / "src"))

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ColPali RAG | WHO Health Reports",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.source-card {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 8px 12px;
    margin: 4px 0;
    background: #f9f9f9;
    font-size: 13px;
}
.score-badge {
    background: #e8f4fd;
    color: #1565c0;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
}
.modality-badge {
    background: #f3e5f5;
    color: #6a1b9a;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
}
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────────────────────

def init_session():
    defaults = {
        "retriever": None,
        "generator": None,
        "messages": [],
        "index_loaded": False,
        "error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


# ── Load pipeline (cached) ────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading ColPali model and index...")
def load_pipeline():
    """Load retriever + generator once, cache for the session."""
    try:
        from retriever import Retriever
        from generator import RAGGenerator

        retriever = Retriever(
            index_dir=os.getenv("INDEX_DIR", "data/index"),
            pdf_dir=os.getenv("PDF_DIR", "data/pdfs"),
            top_k=int(os.getenv("TOP_K", "3")),
        )
        retriever.load()

        generator = RAGGenerator(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            max_tokens=1024,
            temperature=0.1,
        )

        return retriever, generator, None
    except FileNotFoundError as e:
        return None, None, str(e)
    except Exception as e:
        return None, None, f"Pipeline error: {e}"


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔬 ColPali RAG")
    st.markdown("**WHO Global Health Reports**")
    st.markdown("---")

    st.markdown("### About")
    st.markdown("""
This system uses **ColPali** (PaliGemma + late interaction) to retrieve 
relevant document pages purely from their **visual features** — no OCR needed.

Retrieved pages are sent to **Claude** (vision) for grounded, 
citation-backed answers.
    """)

    st.markdown("### Pipeline")
    st.markdown("""
1. `pdf2image` → page images  
2. **ColPali encoder** → patch embeddings  
3. **FAISS index** → multi-vector store  
4. **MaxSim retrieval** → top-K pages  
5. **Claude vision** → grounded answer  
    """)

    st.markdown("### Configuration")
    top_k = st.slider("Pages to retrieve (top-K)", 1, 6, 3)
    show_thumbnails = st.toggle("Show source page images", value=True)
    show_scores = st.toggle("Show relevance scores", value=True)

    st.markdown("---")
    if st.button("🔄 Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("DSAI 413 — Assignment 1 | Zewail City")


# ── Main content ──────────────────────────────────────────────────────────────

st.title("🔬 Multimodal RAG — WHO Health Reports")
st.caption("Ask questions about WHO global health data. The system retrieves relevant PDF pages visually and answers with citations.")

# Load pipeline
retriever, generator, load_error = load_pipeline()

if load_error:
    st.error(f"**Could not load pipeline:** {load_error}")
    st.info("**Setup instructions:**")
    st.code("""
# 1. Download WHO PDFs
python src/download_dataset.py

# 2. Build the ColPali index
python src/build_index.py

# 3. Set your Anthropic API key in .env
ANTHROPIC_API_KEY=sk-ant-...

# 4. Relaunch the app
streamlit run app.py
    """)
    st.stop()

# Pipeline loaded
st.success(f"✓ Index loaded | {len(retriever._pages)} pages across {len(set(p.pdf_name for p in retriever._pages))} documents", icon="📚")

# ── Chat history ───────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander(f"📄 Sources ({len(msg['sources'])} pages retrieved)", expanded=False):
                cols = st.columns(min(len(msg["sources"]), 3))
                for i, src in enumerate(msg["sources"]):
                    with cols[i % len(cols)]:
                        if show_scores:
                            st.markdown(
                                f'<div class="source-card">'
                                f'<b>{src["citation"]}</b><br>'
                                f'<span class="score-badge">score: {src["score"]:.3f}</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.caption(src["citation"])

                        if show_thumbnails and src.get("image"):
                            st.image(src["image"], use_container_width=True)

# ── Chat input ────────────────────────────────────────────────────────────────

# Example queries
if not st.session_state.messages:
    st.markdown("**Try these example queries:**")
    example_queries = [
        "What is the global tuberculosis incidence rate and which regions are most affected?",
        "How has malaria mortality changed between 2015 and 2022?",
        "What are the key statistics on mental health treatment gap worldwide?",
        "Compare cancer incidence rates between different WHO regions.",
        "What percentage of countries have universal health coverage?",
    ]
    cols = st.columns(2)
    for i, q in enumerate(example_queries):
        with cols[i % 2]:
            if st.button(q, key=f"eg_{i}", use_container_width=True):
                st.session_state.pending_query = q

if query := st.chat_input("Ask a question about WHO health reports...") or st.session_state.pop("pending_query", None):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate response
    with st.chat_message("assistant"):
        sources_container = st.empty()
        response_container = st.empty()

        with st.spinner("Retrieving relevant pages..."):
            t_retrieve = time.time()
            retrieved = retriever.retrieve(query, top_k=top_k, load_images=True)
            retrieve_ms = (time.time() - t_retrieve) * 1000

        # Show sources immediately
        sources_info = [
            {
                "citation": r.citation,
                "score": r.score,
                "image": r.image,
                "modality": r.modality,
                "badges": r.modality_badges,
                "table_captions": r.table_captions,
                "figure_captions": r.figure_captions,
                "text_snippet": r.text_snippet,
            }
            for r in retrieved
        ]

        with st.expander(f"📄 Retrieved {len(retrieved)} pages in {retrieve_ms:.0f}ms", expanded=True):
            cols = st.columns(min(len(retrieved), 3))
            for i, src in enumerate(sources_info):
                with cols[i % len(cols)]:
                    badges_html = " ".join(
                        f'<span class="modality-badge">{b}</span>'
                        for b in src["badges"]
                    )
                    score_html = (
                        f'<span class="score-badge">score: {src["score"]:.3f}</span>'
                        if show_scores else ""
                    )
                    st.markdown(
                        f'<div class="source-card">'
                        f'<b>{src["citation"]}</b><br>'
                        f'{score_html} {badges_html}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    # Show detected captions
                    if src["table_captions"]:
                        st.caption("Tables: " + " | ".join(src["table_captions"][:2]))
                    if src["figure_captions"]:
                        st.caption("Figures: " + " | ".join(src["figure_captions"][:2]))
                    if show_thumbnails and src.get("image"):
                        st.image(src["image"], use_container_width=True)

        # Stream answer
        st.markdown("**Answer:**")
        answer_placeholder = st.empty()
        full_answer = ""

        t_gen = time.time()
        for chunk in generator.generate_stream(query, retrieved):
            full_answer += chunk
            answer_placeholder.markdown(full_answer + "▌")
        answer_placeholder.markdown(full_answer)
        gen_ms = (time.time() - t_gen) * 1000

        st.caption(f"Generation: {gen_ms:.0f}ms | Model: llama-4-scout (Groq) | Pages used: {len(retrieved)}")

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer,
        "sources": sources_info,
    })
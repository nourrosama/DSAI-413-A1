import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from ingestion import PDFIngestionPipeline
from indexer import ColPaliIndexer


def main():
    pdf_dir = Path(os.getenv("PDF_DIR", "data/pdfs"))
    index_dir = Path(os.getenv("INDEX_DIR", "data/index"))

    # ── Ingestion ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 1: PDF Ingestion")
    print("=" * 60)

    pipeline = PDFIngestionPipeline(
        pdf_dir=pdf_dir,
        thumbnail_dir=index_dir / "thumbnails",
        dpi=96,
        max_pages_per_pdf=15,
    )

    t0 = time.time()
    pages, images = pipeline.ingest_all(
        save_metadata=True,
        metadata_path=index_dir / "metadata.json",
    )
    print(f"Ingestion done in {time.time() - t0:.1f}s | {len(pages)} pages")

    # ── Encoding + Indexing ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: ColPali Encoding + FAISS Index")
    print("=" * 60)

    indexer = ColPaliIndexer(
        model_name="vidore/colpali-v1.3",
        index_dir=index_dir,
        batch_size=4,
    )

    t1 = time.time()
    indexer.build_index(images)
    indexer.save()
    elapsed = time.time() - t1
    print(f"\nIndexing done in {elapsed:.1f}s ({elapsed/len(pages)*1000:.0f}ms/page)")

    # ── Sanity check ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3: Sanity Check — sample retrieval")
    print("=" * 60)

    indexer.load()
    test_query = "What is the global tuberculosis incidence rate?"
    print(f"Query: {test_query}")

    results = indexer.retrieve(test_query, top_k=3)
    for rank, (page_id, score) in enumerate(results):
        p = pages[page_id]
        print(f"  #{rank+1} score={score:.3f} | {p.citation}")

    print("\n✓ Index built and verified successfully")
    print(f"  Index saved to: {index_dir.resolve()}")
    print("  Run: streamlit run app.py")


if __name__ == "__main__":
    main()
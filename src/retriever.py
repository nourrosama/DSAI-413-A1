from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image

from ingestion import PageRecord, load_metadata
from indexer import ColPaliIndexer


@dataclass
class RetrievedPage:
    """A retrieved page with metadata, image, and modality info."""
    rank: int
    score: float
    record: PageRecord
    image: Optional[Image.Image] = None

    # Modality metadata (from MetadataExtractor)
    modality: str = "text"
    has_table: bool = False
    has_figure: bool = False
    has_chart: bool = False
    table_captions: list = None
    figure_captions: list = None
    text_snippet: str = ""

    def __post_init__(self):
        if self.table_captions is None:
            self.table_captions = []
        if self.figure_captions is None:
            self.figure_captions = []

    @property
    def citation(self) -> str:
        return self.record.citation

    @property
    def modality_badges(self) -> list[str]:
        badges = []
        if self.has_table:  badges.append("📊 table")
        if self.has_figure: badges.append("🖼 figure")
        if self.has_chart:  badges.append("📈 chart")
        if not badges:      badges.append("📝 text")
        return badges


STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","were","be","been","have","has","had","do","does","did",
    "will","would","could","should","may","might","this","that","these",
    "those","it","its","we","our","they","their","from","by","as","not",
    "can","also","which","who","when",
}


class Retriever:
    def __init__(
        self,
        index_dir: str | Path = "data/index",
        pdf_dir: str | Path = "data/pdfs",
        top_k: int = 3,
        text_boost_weight: float = 0.15,
    ):
        self.index_dir = Path(index_dir)
        self.pdf_dir = Path(pdf_dir)
        self.top_k = top_k
        self.text_boost_weight = text_boost_weight
        self._indexer = None
        self._pages = []
        self._page_metadata = {}
        self._text_index = {}
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        self._pages = load_metadata(self.index_dir / "metadata.json")
        self._indexer = ColPaliIndexer(index_dir=self.index_dir)
        self._indexer.load()
        self._indexer.load_model()

        meta_path = self.index_dir / "page_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                for item in json.load(f):
                    self._page_metadata[item["doc_id"]] = item
            print(f"Modality metadata loaded: {len(self._page_metadata)} pages")
        else:
            print("Note: run metadata_extractor.py for richer modality metadata")

        text_idx_path = self.index_dir / "text_index.json"
        if text_idx_path.exists():
            with open(text_idx_path) as f:
                data = json.load(f)
            self._text_index = data.get("inverted_index", {})
            print(f"Text index loaded: {len(self._text_index)} terms")

        self._loaded = True
        print(f"Retriever ready: {len(self._pages)} pages indexed")

    def _keyword_scores(self, query: str) -> dict:
        if not self._text_index:
            return {}
        words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())) - STOPWORDS
        if not words:
            return {}
        doc_hits = {}
        for word in words:
            for doc_id in self._text_index.get(word, []):
                doc_hits[doc_id] = doc_hits.get(doc_id, 0) + 1
        return {doc_id: hits / len(words) for doc_id, hits in doc_hits.items()}

    def retrieve(self, query, top_k=None, load_images=True):
        if not self._loaded:
            self.load()
        k = top_k or self.top_k
        t0 = time.time()

        ranked = self._indexer.retrieve(query, top_k=k * 3)
        visual_scores = {page_id: score for page_id, score in ranked}

        if visual_scores:
            max_v = max(visual_scores.values())
            min_v = min(visual_scores.values())
            rng = max_v - min_v or 1.0
            visual_scores = {k: (v - min_v) / rng for k, v in visual_scores.items()}

        keyword_scores = self._keyword_scores(query)
        all_ids = set(visual_scores) | set(keyword_scores)
        combined = {}
        for doc_id in all_ids:
            v = visual_scores.get(doc_id, 0.0)
            t = keyword_scores.get(doc_id, 0.0)
            combined[doc_id] = (1 - self.text_boost_weight) * v + self.text_boost_weight * t

        top_ids = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
        latency_ms = (time.time() - t0) * 1000

        results = []
        for rank, (page_id, score) in enumerate(top_ids):
            record = self._pages[page_id]
            image = self._load_image(record) if load_images else None
            meta = self._page_metadata.get(page_id, {})

            results.append(RetrievedPage(
                rank=rank + 1,
                score=score,
                record=record,
                image=image,
                modality=meta.get("modality", "text"),
                has_table=meta.get("has_table", False),
                has_figure=meta.get("has_figure", False),
                has_chart=meta.get("has_chart", False),
                table_captions=meta.get("table_captions", []),
                figure_captions=meta.get("figure_captions", []),
                text_snippet=meta.get("text_chunk", "")[:200],
            ))

        print(f"Retrieved {len(results)} pages in {latency_ms:.0f}ms")
        return results

    def _load_image(self, record):
        try:
            if record.image_path and Path(record.image_path).exists():
                return Image.open(record.image_path).convert("RGB")
            from pdf2image import convert_from_path
            pdf_path = self.pdf_dir / f"{record.pdf_name}.pdf"
            images = convert_from_path(
                str(pdf_path), dpi=120,
                first_page=record.page_number,
                last_page=record.page_number,
            )
            return images[0] if images else None
        except Exception as e:
            print(f"Warning: could not load image for {record.citation}: {e}")
            return None
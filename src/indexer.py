from __future__ import annotations

import gc
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import faiss
from PIL import Image
from tqdm import tqdm

# ColPali via byaldi wrapper
try:
    from byaldi import RAGMultiModalModel
    BYALDI_AVAILABLE = True
except ImportError:
    BYALDI_AVAILABLE = False

# Direct colpali-engine import as fallback
try:
    from colpali_engine.models import ColPali, ColPaliProcessor
    from torch.utils.data import DataLoader
    COLPALI_DIRECT = True
except ImportError:
    COLPALI_DIRECT = False


MODEL_NAME = "vidore/colpali-v1.3"
EMBEDDING_DIM = 128  # ColPali output dimension
INDEX_DIR = Path("data/index")


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ColPaliIndexer:
    """
    Builds and persists a FAISS multi-vector index from page images.

    The index stores patch-level embeddings from ColPali. During retrieval,
    MaxSim aggregation scores each page by summing the best patch match
    for each query token.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        index_dir: str | Path = INDEX_DIR,
        batch_size: int = 1,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.device = device or get_device()

        self._model = None
        self._processor = None
        self._index: faiss.Index | None = None
        self._page_map: list[int] = []    # flat_idx → page_id
        self._patch_counts: list[int] = []  # page_id → num patches

        print(f"ColPaliIndexer | device={self.device} | model={model_name}")

    # ── Model loading ────────────────────────────────────────────────────────

    def load_model(self):
        """Load ColPali model and processor."""
        if self._model is not None:
            return

        print(f"Loading ColPali model: {self.model_name}")
        dtype = torch.float16 if self.device == "cuda" else torch.float16

        self._model = ColPali.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=self.device,
            low_cpu_mem_usage=True,
        ).eval()
        self._processor = ColPaliProcessor.from_pretrained(self.model_name)
        print("Model loaded ✓")

    # ── Encoding ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_pages(self, images: list[Image.Image]) -> list[np.ndarray]:
        """
        Encode a list of PIL Images → list of patch embedding arrays.

        Each array has shape (num_patches, embedding_dim).
        """
        self.load_model()
        all_embeddings: list[np.ndarray] = []

        for i in tqdm(range(0, len(images), self.batch_size), desc="Encoding pages"):
            batch_imgs = images[i : i + self.batch_size]
            inputs = self._processor.process_images(batch_imgs)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # ColPali output: (batch, num_patches, embed_dim)
            embeddings = self._model(**inputs)  # tensor

            for emb in embeddings:
                # emb: (num_patches, embed_dim)
                all_embeddings.append(emb.cpu().float().numpy())

            # Free memory between batches (important on CPU)
            del inputs, embeddings
            gc.collect()

        return all_embeddings

    @torch.no_grad()
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a text query → token embedding array.
        Shape: (num_tokens, embedding_dim)
        """
        self.load_model()
        inputs = self._processor.process_queries([query])
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        embeddings = self._model(**inputs)  # (1, num_tokens, embed_dim)
        return embeddings[0].cpu().float().numpy()

    # ── Index building ────────────────────────────────────────────────────────

    def build_index(self, images: list[Image.Image]) -> None:
        """
        Encode all pages and build a FAISS flat L2 index.

        Index layout: each patch embedding is stored at a unique flat index.
        self._page_map[flat_idx] = page_id tells us which page owns that patch.
        """
        print(f"\nBuilding index for {len(images)} pages...")
        page_embeddings = self.encode_pages(images)

        all_vectors = []
        self._page_map = []
        self._patch_counts = []

        for page_id, patch_embs in enumerate(page_embeddings):
            # patch_embs: (num_patches, embed_dim)
            # L2-normalize for cosine similarity
            norms = np.linalg.norm(patch_embs, axis=1, keepdims=True) + 1e-8
            patch_embs = patch_embs / norms

            self._patch_counts.append(len(patch_embs))
            self._page_map.extend([page_id] * len(patch_embs))
            all_vectors.append(patch_embs)

        matrix = np.vstack(all_vectors).astype(np.float32)  # (total_patches, embed_dim)
        dim = matrix.shape[1]

        print(f"Total patch vectors: {matrix.shape[0]} | dim: {dim}")

        # Flat (exact) index — use IVF for large corpora (>100k vectors)
        self._index = faiss.IndexFlatIP(dim)  # Inner product (L2-normalized = cosine)
        self._index.add(matrix)

        print(f"FAISS index size: {self._index.ntotal}")

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist index and mappings to disk."""
        assert self._index is not None, "Build index first"

        faiss.write_index(self._index, str(self.index_dir / "colpali.faiss"))

        meta = {
            "page_map": self._page_map,
            "patch_counts": self._patch_counts,
            "model_name": self.model_name,
            "embedding_dim": self._index.d,
        }
        with open(self.index_dir / "index_meta.pkl", "wb") as f:
            pickle.dump(meta, f)

        print(f"Index saved to {self.index_dir}")

    def load(self) -> None:
        """Load persisted index and mappings from disk."""
        index_path = self.index_dir / "colpali.faiss"
        meta_path = self.index_dir / "index_meta.pkl"

        if not index_path.exists():
            raise FileNotFoundError(
                f"No index found at {index_path}. "
                "Run `python src/build_index.py` first."
            )

        self._index = faiss.read_index(str(index_path))

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        self._page_map = meta["page_map"]
        self._patch_counts = meta["patch_counts"]
        print(f"Index loaded: {self._index.ntotal} patch vectors, {len(self._patch_counts)} pages")

    # ── MaxSim Retrieval ──────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        candidate_factor: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Retrieve top-k pages using MaxSim late interaction scoring.

        Algorithm:
          1. Encode query → token embeddings Q (num_tokens, embed_dim)
          2. For each query token, find its nearest patch via FAISS
          3. Accumulate MaxSim score per page: sum over tokens of max patch sim
          4. Return top-k pages ranked by score

        Returns: list of (page_id, score) sorted descending
        """
        assert self._index is not None, "Load or build index first"

        query_emb = self.encode_query(query)  # (num_tokens, embed_dim)

        # L2-normalize query tokens
        norms = np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-8
        query_emb = (query_emb / norms).astype(np.float32)

        # Retrieve many candidates per query token
        num_candidates = min(top_k * candidate_factor, self._index.ntotal)
        similarities, flat_indices = self._index.search(query_emb, num_candidates)
        # similarities: (num_tokens, num_candidates)
        # flat_indices:  (num_tokens, num_candidates)

        # MaxSim: for each page, sum the best token-patch similarities
        page_scores: dict[int, float] = {}
        num_pages = len(self._patch_counts)

        for token_idx in range(len(query_emb)):
            # Per-page maximum similarity for this token
            token_best: dict[int, float] = {}
            for rank in range(num_candidates):
                fi = flat_indices[token_idx, rank]
                if fi < 0:
                    continue
                sim = float(similarities[token_idx, rank])
                page_id = self._page_map[fi]
                if page_id not in token_best or sim > token_best[page_id]:
                    token_best[page_id] = sim

            # Accumulate into global page scores
            for page_id, best_sim in token_best.items():
                page_scores[page_id] = page_scores.get(page_id, 0.0) + best_sim

        # Sort and return top-k
        ranked = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
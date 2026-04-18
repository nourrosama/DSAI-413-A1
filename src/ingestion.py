"""
Ingestion pipeline: PDF → PIL page images + metadata.

ColPali does NOT need OCR or text extraction — each page is encoded
directly as an image. This module handles:
  - PDF → list of PIL Images (one per page)
  - Metadata extraction (filename, page number, total pages)
  - Optional page thumbnail saving for UI display
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Generator

from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm


@dataclass
class PageRecord:
    """Metadata for a single PDF page."""
    doc_id: int          # Index in the corpus
    pdf_name: str        # Original filename (no extension)
    page_number: int     # 1-indexed page number
    total_pages: int     # Total pages in the PDF
    image_path: str      # Path to saved PNG thumbnail (optional)

    @property
    def citation(self) -> str:
        """Human-readable citation string."""
        return f"{self.pdf_name}, p. {self.page_number}"


class PDFIngestionPipeline:
    """
    Converts a directory of PDFs into page images ready for ColPali encoding.

    Usage:
        pipeline = PDFIngestionPipeline("data/pdfs", "data/index/thumbnails")
        pages, images = pipeline.ingest_all()
    """

    def __init__(
        self,
        pdf_dir: str | Path,
        thumbnail_dir: str | Path | None = None,
        dpi: int = 150,
        max_pages_per_pdf: int | None = None,
    ):
        self.pdf_dir = Path(pdf_dir)
        self.thumbnail_dir = Path(thumbnail_dir) if thumbnail_dir else None
        self.dpi = dpi
        self.max_pages_per_pdf = max_pages_per_pdf

        if self.thumbnail_dir:
            self.thumbnail_dir.mkdir(parents=True, exist_ok=True)

    def iter_pdfs(self) -> Generator[Path, None, None]:
        """Yield all PDF paths in the directory."""
        for path in sorted(self.pdf_dir.glob("*.pdf")):
            yield path

    def pdf_to_images(self, pdf_path: Path) -> list[Image.Image]:
        """Convert a single PDF to a list of PIL Images."""
        images = convert_from_path(
            str(pdf_path),
            dpi=self.dpi,
            fmt="RGB",
        )
        if self.max_pages_per_pdf:
            images = images[: self.max_pages_per_pdf]
        return images

    def ingest_all(
        self,
        save_metadata: bool = True,
        metadata_path: str | Path = "data/index/metadata.json",
    ) -> tuple[list[PageRecord], list[Image.Image]]:
        """
        Ingest all PDFs in pdf_dir.

        Returns:
            pages: list of PageRecord metadata (one per page)
            images: list of PIL Images in same order as pages
        """
        pages: list[PageRecord] = []
        images: list[Image.Image] = []
        doc_id = 0

        pdfs = list(self.iter_pdfs())
        if not pdfs:
            raise FileNotFoundError(
                f"No PDFs found in {self.pdf_dir}. "
                "Run `python src/download_dataset.py` first."
            )

        print(f"Found {len(pdfs)} PDF(s) in {self.pdf_dir}")

        for pdf_path in tqdm(pdfs, desc="Ingesting PDFs"):
            pdf_name = pdf_path.stem
            print(f"\n  Processing: {pdf_path.name}")

            try:
                pdf_images = self.pdf_to_images(pdf_path)
            except Exception as e:
                print(f"  ✗ Failed to convert {pdf_path.name}: {e}")
                continue

            total_pages = len(pdf_images)
            print(f"  → {total_pages} pages")

            for page_idx, img in enumerate(pdf_images):
                page_number = page_idx + 1
                image_path = ""

                # Optionally save thumbnail for UI
                if self.thumbnail_dir:
                    thumb_name = f"{pdf_name}_p{page_number:04d}.jpg"
                    thumb_path = self.thumbnail_dir / thumb_name
                    if not thumb_path.exists():
                        # Resize for faster display, keep aspect ratio
                        thumb = img.copy()
                        thumb.thumbnail((800, 1200), Image.LANCZOS)
                        thumb.save(str(thumb_path), "JPEG", quality=85)
                    image_path = str(thumb_path)

                record = PageRecord(
                    doc_id=doc_id,
                    pdf_name=pdf_name,
                    page_number=page_number,
                    total_pages=total_pages,
                    image_path=image_path,
                )
                pages.append(record)
                images.append(img)
                doc_id += 1

        print(f"\nTotal pages ingested: {len(pages)}")

        if save_metadata:
            meta_path = Path(metadata_path)
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            with open(meta_path, "w") as f:
                json.dump([asdict(p) for p in pages], f, indent=2)
            print(f"Metadata saved to: {meta_path}")

        return pages, images


def load_metadata(metadata_path: str | Path = "data/index/metadata.json") -> list[PageRecord]:
    """Load previously saved page metadata."""
    with open(metadata_path) as f:
        data = json.load(f)
    return [PageRecord(**d) for d in data]


if __name__ == "__main__":
    pipeline = PDFIngestionPipeline(
        pdf_dir="data/pdfs",
        thumbnail_dir="data/index/thumbnails",
        dpi=150,
        max_pages_per_pdf=50,  # Limit for dev; remove for production
    )
    pages, images = pipeline.ingest_all()
    print(f"\nReady: {len(pages)} pages from {len(set(p.pdf_name for p in pages))} PDFs")
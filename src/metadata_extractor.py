from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pypdf import PdfReader
from tqdm import tqdm


@dataclass
class PageMetadata:
    """Rich metadata for a single PDF page."""
    doc_id: int
    pdf_name: str
    page_number: int

    # Text content
    text_chunk: str = ""          # Raw extracted text (first 1000 chars)
    word_count: int = 0

    # Modality flags
    has_table: bool = False        # Page likely contains a table
    has_figure: bool = False       # Page likely contains a figure/image
    has_chart: bool = False        # Page likely contains a chart/plot
    has_equation: bool = False     # Page contains mathematical equations

    # Detected captions
    table_captions: list[str] = field(default_factory=list)
    figure_captions: list[str] = field(default_factory=list)

    # Primary modality label
    modality: str = "text"         # text / table / figure / chart / mixed

    @property
    def modality_tags(self) -> list[str]:
        tags = []
        if self.has_table:   tags.append("table")
        if self.has_figure:  tags.append("figure")
        if self.has_chart:   tags.append("chart")
        if self.has_equation: tags.append("equation")
        return tags or ["text"]


# ── Regex patterns for detecting content types ────────────────────────────────

# Table captions: "Table 1", "Table I", "TABLE 1"
TABLE_CAPTION_RE = re.compile(
    r'\b(Table\s+[\dIVXivx]+[\.\:]?\s*.{0,80})',
    re.IGNORECASE
)

# Figure captions: "Figure 1", "Fig. 1", "Fig 1"
FIGURE_CAPTION_RE = re.compile(
    r'\b(Fig(?:ure|\.?)?\s+[\dIVXivx]+[\.\:]?\s*.{0,80})',
    re.IGNORECASE
)

# Chart indicators in text
CHART_KEYWORDS_RE = re.compile(
    r'\b(bar chart|line chart|pie chart|histogram|scatter plot|'
    r'boxplot|heatmap|confusion matrix|roc curve|precision.recall|'
    r'learning curve|training curve|loss curve|accuracy curve)\b',
    re.IGNORECASE
)

# Table structure indicators (pipe chars, dashes forming borders, aligned columns)
TABLE_STRUCTURE_RE = re.compile(
    r'(\|.+\|)|(-{3,}[\s-]*-{3,})|((?:\d+\.?\d*\s+){4,})',
)

# Math/equation indicators
EQUATION_RE = re.compile(
    r'(\\frac|\\sum|\\int|\\alpha|\\beta|\\theta|∑|∫|→|≈|≤|≥|'
    r'\$.*?\$|[A-Z]\s*=\s*[A-Z].*[A-Z])',
)


class MetadataExtractor:
    """
    Extracts structural metadata from PDFs to complement ColPali visual retrieval.

    Usage:
        extractor = MetadataExtractor("data/pdfs", "data/index")
        metadata = extractor.extract_all(pages)  # pages from ingestion.py
        extractor.save(metadata)
    """

    def __init__(
        self,
        pdf_dir: str | Path,
        index_dir: str | Path,
        max_text_chars: int = 1500,
    ):
        self.pdf_dir = Path(pdf_dir)
        self.index_dir = Path(index_dir)
        self.max_text_chars = max_text_chars

    def extract_page_text(self, pdf_path: Path, page_number: int) -> str:
        """Extract raw text from a specific page using pypdf."""
        try:
            reader = PdfReader(str(pdf_path))
            # page_number is 1-indexed
            page = reader.pages[page_number - 1]
            text = page.extract_text() or ""
            return text[:self.max_text_chars]
        except Exception:
            return ""

    def analyze_text(self, text: str) -> dict:
        """
        Analyze extracted text for content type indicators.
        Returns a dict of detected features.
        """
        features = {
            "word_count": len(text.split()),
            "has_table": False,
            "has_figure": False,
            "has_chart": False,
            "has_equation": False,
            "table_captions": [],
            "figure_captions": [],
        }

        if not text:
            return features

        # Table detection
        table_caps = TABLE_CAPTION_RE.findall(text)
        if table_caps or TABLE_STRUCTURE_RE.search(text):
            features["has_table"] = True
            features["table_captions"] = [c.strip()[:100] for c in table_caps[:3]]

        # Figure detection
        fig_caps = FIGURE_CAPTION_RE.findall(text)
        if fig_caps:
            features["has_figure"] = True
            features["figure_captions"] = [c.strip()[:100] for c in fig_caps[:3]]

        # Chart detection (keyword-based)
        if CHART_KEYWORDS_RE.search(text):
            features["has_chart"] = True
            features["has_figure"] = True  # charts are also figures

        # Equation detection
        if EQUATION_RE.search(text):
            features["has_equation"] = True

        return features

    def determine_modality(self, features: dict) -> str:
        """Determine primary modality label for a page."""
        flags = []
        if features["has_table"]:  flags.append("table")
        if features["has_chart"]:  flags.append("chart")
        if features["has_figure"]: flags.append("figure")

        if len(flags) == 0:
            return "text"
        elif len(flags) == 1:
            return flags[0]
        else:
            return "mixed"

    def extract_all(self, pages) -> list[PageMetadata]:
        """
        Extract metadata for all pages.

        Args:
            pages: list of PageRecord from ingestion.py

        Returns:
            list of PageMetadata
        """
        metadata_list: list[PageMetadata] = []

        # Group pages by PDF to avoid re-opening the same file repeatedly
        from collections import defaultdict
        by_pdf: dict[str, list] = defaultdict(list)
        for page in pages:
            by_pdf[page.pdf_name].append(page)

        print(f"Extracting metadata from {len(by_pdf)} PDFs...")

        for pdf_name, pdf_pages in tqdm(by_pdf.items(), desc="Extracting metadata"):
            pdf_path = self.pdf_dir / f"{pdf_name}.pdf"

            if not pdf_path.exists():
                print(f"  Warning: {pdf_path} not found, skipping")
                continue

            for page_rec in pdf_pages:
                # Extract text
                text = self.extract_page_text(pdf_path, page_rec.page_number)

                # Analyze for content types
                features = self.analyze_text(text)

                # Determine modality
                modality = self.determine_modality(features)

                metadata = PageMetadata(
                    doc_id=page_rec.doc_id,
                    pdf_name=pdf_name,
                    page_number=page_rec.page_number,
                    text_chunk=text,
                    word_count=features["word_count"],
                    has_table=features["has_table"],
                    has_figure=features["has_figure"],
                    has_chart=features["has_chart"],
                    has_equation=features["has_equation"],
                    table_captions=features["table_captions"],
                    figure_captions=features["figure_captions"],
                    modality=modality,
                )
                metadata_list.append(metadata)

        # Print modality distribution
        from collections import Counter
        dist = Counter(m.modality for m in metadata_list)
        print(f"\nModality distribution across {len(metadata_list)} pages:")
        for mod, count in sorted(dist.items()):
            pct = count / len(metadata_list) * 100
            print(f"  {mod:8s}: {count:3d} pages ({pct:.0f}%)")

        return metadata_list

    def save(
        self,
        metadata_list: list[PageMetadata],
        path: str | Path | None = None,
    ) -> Path:
        """Save metadata to JSON."""
        if path is None:
            path = self.index_dir / "page_metadata.json"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump([asdict(m) for m in metadata_list], f, indent=2)

        print(f"Metadata saved → {path}")
        return path

    @staticmethod
    def load(path: str | Path = "data/index/page_metadata.json") -> list[PageMetadata]:
        """Load saved metadata."""
        with open(path) as f:
            data = json.load(f)
        return [PageMetadata(**d) for d in data]


def build_text_search_index(
    metadata_list: list[PageMetadata],
    output_path: str | Path = "data/index/text_index.json",
) -> dict[int, str]:
    """
    Build a simple keyword → [doc_id] inverted index from text chunks.
    Used for hybrid retrieval (text keyword fallback when ColPali score is low).
    """
    from collections import defaultdict

    # Simple word-level inverted index
    inverted: dict[str, list[int]] = defaultdict(list)
    doc_texts: dict[int, str] = {}

    # Common stopwords to skip
    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "this", "that", "these",
        "those", "it", "its", "we", "our", "they", "their", "from",
        "by", "as", "not", "can", "also", "which", "who", "when",
    }

    for meta in metadata_list:
        doc_texts[meta.doc_id] = meta.text_chunk
        words = re.findall(r'\b[a-zA-Z]{3,}\b', meta.text_chunk.lower())
        for word in set(words):
            if word not in STOPWORDS:
                inverted[word].append(meta.doc_id)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "inverted_index": dict(inverted),
            "doc_texts": {str(k): v for k, v in doc_texts.items()},
        }, f)

    print(f"Text index saved → {output_path} ({len(inverted)} terms)")
    return doc_texts


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from ingestion import load_metadata

    pages = load_metadata("data/index/metadata.json")
    extractor = MetadataExtractor("data/pdfs", "data/index")
    metadata = extractor.extract_all(pages)
    extractor.save(metadata)
    build_text_search_index(metadata)
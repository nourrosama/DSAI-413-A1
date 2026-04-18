"""
Dataset downloader for the ESA Space Environment + AI Research Papers corpus.

Documents:
  - ESA Annual Space Environment Report (ESA, visually rich with charts/maps)
  - Attention Is All You Need (Vaswani et al., 2017)
  - GPT-4 Technical Report (OpenAI, 2023)
  - Mistral 7B (Jiang et al., 2023)
  - ColPali: Efficient Document Retrieval with VLMs (Faysse et al., 2024)

These PDFs are publicly available and contain diverse modalities:
text, tables, figures, charts, and equations — ideal for multimodal RAG.
"""

import os
import urllib.request
from pathlib import Path

OUTPUT_DIR = Path("data/pdfs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PDFS = [
    {
        "name": "esa_space_environment.pdf",
        "url": "https://www.sdo.esoc.esa.int/environment_report/Space_Environment_Report_latest.pdf",
        "description": "ESA Annual Space Environment Report — debris charts, launch statistics, compliance tables",
    },
    {
        "name": "attention_paper.pdf",
        "url": "https://arxiv.org/pdf/1706.03762",
        "description": "Attention Is All You Need — Transformer architecture, BLEU score tables, attention diagrams",
    },
    {
        "name": "gpt4_paper.pdf",
        "url": "https://arxiv.org/pdf/2303.08774",
        "description": "GPT-4 Technical Report — exam benchmark tables, safety evaluation charts",
    },
    {
        "name": "mistral_paper.pdf",
        "url": "https://arxiv.org/pdf/2310.06825",
        "description": "Mistral 7B — sliding window attention figures, benchmark comparison tables",
    },
    {
        "name": "colpali_paper.pdf",
        "url": "https://arxiv.org/pdf/2407.01449",
        "description": "ColPali: Efficient Document Retrieval with VLMs — ViDoRe benchmark, retrieval pipeline figures",
    },
]


def download_pdf(url: str, output_path: Path) -> bool:
    """Download a PDF using urllib. Returns True on success."""
    if output_path.exists():
        size_mb = output_path.stat().st_size / 1e6
        if size_mb > 0.1:
            print(f"  ✓ Already exists: {output_path.name} ({size_mb:.1f} MB)")
            return True
        else:
            print(f"  ⚠ Found but too small ({size_mb:.2f} MB), re-downloading...")
            output_path.unlink()

    try:
        print(f"  Downloading {output_path.name}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as response, \
             open(output_path, "wb") as f:
            f.write(response.read())

        size_mb = output_path.stat().st_size / 1e6
        print(f"  ✓ Done: {size_mb:.1f} MB")
        return True

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def main():
    print("=" * 60)
    print("ESA & AI Research Papers Downloader")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR.resolve()}\n")

    success, failed = [], []
    for pdf in PDFS:
        path = OUTPUT_DIR / pdf["name"]
        print(f"\n→ {pdf['name']}")
        print(f"  {pdf['description']}")
        if download_pdf(pdf["url"], path):
            success.append(pdf["name"])
        else:
            failed.append(pdf["name"])

    print("\n" + "=" * 60)
    print(f"Downloaded: {len(success)}/{len(PDFS)}")
    if failed:
        print(f"\nFailed: {failed}")
        for pdf in PDFS:
            if pdf["name"] in failed:
                print(f"  {pdf['url']}")
    print("=" * 60)
    print("\nNext step: python src/build_index.py")


if __name__ == "__main__":
    main()
from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass

from groq import Groq
from PIL import Image

from retriever import RetrievedPage


@dataclass
class GeneratedAnswer:
    """Structured output from the VLM."""
    answer: str
    citations: list[str]
    num_pages_used: int
    model: str


# Groq vision model — free tier, supports multiple images
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def image_to_base64(img: Image.Image, max_size: tuple = (1024, 1400)) -> str:
    """Resize and encode a PIL image as base64 JPEG."""
    img = img.copy()
    img.thumbnail(max_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


SYSTEM_PROMPT = """You are a precise document QA assistant. You receive images of document pages from WHO health reports and answer questions based solely on their content.

Rules:
1. Answer ONLY from the provided page images. Do not use external knowledge.
2. If the answer spans multiple pages, synthesize across them.
3. Be specific: quote statistics, figures, and table values when relevant.
4. At the end of your answer, list which pages informed your response in a "Sources:" section.
5. If the provided pages do not contain sufficient information, say so clearly.
6. For tables and charts, describe the key data points in plain language."""


class RAGGenerator:
    """
    Generates grounded answers using Groq vision + retrieved page images.

    Usage:
        generator = RAGGenerator()
        answer = generator.generate(query, retrieved_pages)
        print(answer.answer)
        print(answer.citations)
    """

    def __init__(
        self,
        model: str = GROQ_MODEL,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not set.\n"
                "1. Go to https://console.groq.com and sign up (free)\n"
                "2. Create an API key\n"
                "3. Add GROQ_API_KEY=your_key to your .env file"
            )

        self.client = Groq(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _build_content(self, query, retrieved_pages, include_scores=True):
        content = []

        # Only use the top 1 page image (Scout free tier limitation)
        best_page = retrieved_pages[0]

        content.append({
            "type": "text",
            "text": (
                f"You are a document QA assistant. Answer based only on the provided page image.\n\n"
                f"Query: {query}\n\n"
                f"Source: {best_page.citation}"
            )
        })

        if best_page.image is not None:
            b64 = image_to_base64(best_page.image)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })

        content.append({
            "type": "text",
            "text": "Answer the query based on this page. Be specific and cite the source at the end."
        })

        return content

    def generate(
        self,
        query: str,
        retrieved_pages: list[RetrievedPage],
    ) -> GeneratedAnswer:
        """
        Generate a grounded answer from the query + retrieved page images.

        Args:
            query: User's natural language question
            retrieved_pages: Top-K pages from the retriever (with images)

        Returns:
            GeneratedAnswer with answer text and citations
        """
        if not retrieved_pages:
            return GeneratedAnswer(
                answer="No relevant pages were retrieved. Please try a different query.",
                citations=[],
                num_pages_used=0,
                model=self.model,
            )

        content = self._build_content(query, retrieved_pages)

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
        )

        answer_text = response.choices[0].message.content
        citations = [p.citation for p in retrieved_pages]

        return GeneratedAnswer(
            answer=answer_text,
            citations=citations,
            num_pages_used=len(retrieved_pages),
            model=self.model,
        )

    def generate_stream(
        self,
        query: str,
        retrieved_pages: list[RetrievedPage],
    ):
        """
        Generator that yields text chunks for streaming in Streamlit.

        Usage:
            for chunk in generator.generate_stream(query, pages):
                print(chunk, end="", flush=True)
        """
        if not retrieved_pages:
            yield "No relevant pages were retrieved. Please try a different query."
            return

        content = self._build_content(query, retrieved_pages, include_scores=False)

        stream = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=True,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
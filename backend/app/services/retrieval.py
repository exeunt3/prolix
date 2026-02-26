from __future__ import annotations

import json
from pathlib import Path

from app.models import RetrievalBundle, RetrievalSnippet, VectorDomain


class RetrievalService:
    def __init__(self, corpus_dir: str = "corpus") -> None:
        self.corpus_path = Path(corpus_dir)
        self.docs = self._load_corpus()

    def _load_corpus(self) -> list[dict]:
        docs: list[dict] = []
        if not self.corpus_path.exists():
            return docs
        for item in self.corpus_path.glob("*.json"):
            payload = json.loads(item.read_text())
            docs.extend(payload)
        return docs

    def retrieve(self, object_label: str, domain: VectorDomain, k: int = 12) -> list[RetrievalSnippet]:
        scored: list[tuple[int, dict]] = []
        terms = {object_label.lower(), domain.value.lower().replace("_", " ")}
        for doc in self.docs:
            text = f"{doc.get('title','')} {doc.get('excerpt','')} {doc.get('domain_tag','')}".lower()
            score = sum(3 for term in terms if term in text)
            if domain.value == doc.get("domain_tag"):
                score += 5
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [entry for _, entry in scored[:k]]
        if not top:
            top = [
                {
                    "title": "Entropy and prediction",
                    "excerpt": "Information processing is paid for in heat, delay, and substrate.",
                    "domain_tag": "THERMODYNAMICS_ENTROPY",
                    "source_id": "core-entropy",
                }
            ]
        return [RetrievalSnippet(**doc) for doc in top]


    def build_bundle(self, snippets: list[RetrievalSnippet]) -> RetrievalBundle:
        return RetrievalBundle(
            fragment_ids=[snippet.source_id for snippet in snippets],
            fragments=[f"{snippet.title}: {snippet.excerpt}" for snippet in snippets],
        )

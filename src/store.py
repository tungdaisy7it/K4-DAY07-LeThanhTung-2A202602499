from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        # Chi dung in-memory. Bo han nhanh Chroma: khong test nao can no,
        # requirements.txt khong cai no, va neu may cham bai tinh co co chromadb
        # thi moi method se re vao nhanh chua cai dat -> 14 test store sap.
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

    def _make_record(self, doc: Document) -> dict[str, Any]:
        # Copy metadata thay vi dung truc tiep object cua nguoi goi: neu khong,
        # setdefault("doc_id") ben duoi se sua nguoc dict cua caller.
        metadata = dict(doc.metadata or {})
        # delete_document() loc theo metadata['doc_id'] nen record luon phai co khoa nay.
        # Khi chunk mot file thanh nhieu Document ("file#0", "file#1"), bench.py da gan
        # san doc_id tro ve FILE GOC; setdefault giu nguyen gia tri do.
        metadata.setdefault("doc_id", doc.id)

        record = {
            "id": doc.id,
            "content": doc.content,
            "metadata": metadata,
            "embedding": self._embedding_fn(doc.content),
            "index": self._next_index,
        }
        self._next_index += 1
        return record

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not records or top_k <= 0:
            return []

        query_embedding = self._embedding_fn(query)
        scored = [
            {
                "id": record["id"],
                "content": record["content"],
                "metadata": record["metadata"],
                # Embedding da duoc chuan hoa (||v|| = 1) nen dot product = cosine.
                "score": _dot(query_embedding, record["embedding"]),
            }
            for record in records
        ]
        # Bo khoa "embedding" khoi ket qua: vector 64-1536 chieu lam ban output terminal.
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if not docs:
            return
        # 1 Document = 1 record. Store khong tu chunk — viec chunk nam o tang ngoai
        # (bench.py), moi chunk duoc dong goi thanh mot Document rieng.
        for doc in docs:
            self._store.append(self._make_record(doc))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        # Loc TRUOC roi moi search. Neu lay top-k xong moi bo cai khong khop thi
        # k slot co the da bi tai lieu sai chiem het -> tra ve 0 ket qua du store
        # van con tai lieu hop le.
        if not metadata_filter:
            candidates = self._store
        else:
            candidates = [
                record
                for record in self._store
                if all(record["metadata"].get(key) == value for key, value in metadata_filter.items())
            ]

        # Ca hai duong deu di qua _search_records nen ket qua khong the lech nhau.
        return self._search_records(query, candidates, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        size_before = len(self._store)
        self._store = [record for record in self._store if record["metadata"].get("doc_id") != doc_id]
        return len(self._store) < size_before

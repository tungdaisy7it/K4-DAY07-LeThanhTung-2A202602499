from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    SYSTEM_RULES = (
        "Ban la tro ly tra cuu quy dinh/dich vu dai hoc.\n"
        "Quy tac bat buoc:\n"
        "1. Chi tra loi dua tren NGU CANH duoc cung cap ben duoi.\n"
        "2. Moi y trong cau tra loi phai trich dan so nguon dang [1], [2], [3].\n"
        "3. Neu ngu canh khong chua thong tin, tra loi dung mot cau: "
        "'Khong tim thay thong tin nay trong tai lieu duoc cung cap.' — khong suy doan."
    )

    NO_CONTEXT_ANSWER = "Khong tim thay thong tin nay trong tai lieu duoc cung cap."

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def _format_context(self, results: list[dict]) -> str:
        blocks = []
        for index, result in enumerate(results, start=1):
            metadata = result.get("metadata") or {}
            source = metadata.get("source_url") or metadata.get("source") or metadata.get("doc_id") or result.get("id")
            # Danh so tung chunk kem nguon -> cau tra loi truy vet duoc ve dung
            # chunk va dung file (tieu chi Source Traceability trong EVALUATION.md).
            blocks.append(
                f"[{index}] (doc_id={metadata.get('doc_id', result.get('id'))}"
                f" | audience={metadata.get('audience', 'n/a')}"
                f" | nguon={source}"
                f" | score={result.get('score', 0.0):.4f})\n{result.get('content', '')}"
            )
        return "\n\n".join(blocks)

    def build_prompt(self, question: str, results: list[dict]) -> str:
        return (
            f"{self.SYSTEM_RULES}\n\n"
            f"=== NGU CANH TRUY XUAT DUOC ===\n"
            f"{self._format_context(results)}\n\n"
            f"=== CAU HOI ===\n{question}\n\n"
            f"=== CAU TRA LOI (kem trich dan [n]) ==="
        )

    def answer(self, question: str, top_k: int = 3, metadata_filter: dict | None = None) -> str:
        # 1. Retrieve — dung search_with_filter de agent cung ho tro loc metadata.
        results = self.store.search_with_filter(question, top_k=top_k, metadata_filter=metadata_filter)

        # Store rong / khong truy xuat duoc gi: tra ve thong bao, khong crash
        # va khong goi LLM vo ich.
        if not results:
            return self.NO_CONTEXT_ANSWER

        # 2. Build prompt co ngu canh danh so. 3. Goi LLM.
        return self.llm_fn(self.build_prompt(question, results))

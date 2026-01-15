from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from fastapi import UploadFile
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)

from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException

# -------------------------------------------------------------------
# FIX 1: Logger was never initialized → NameError at runtime
# -------------------------------------------------------------------
log = CustomLogger(__name__).get_logger()

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


# ============================================================
# Document Loading
# ============================================================
def load_documents(paths: Iterable[Path]) -> List[Document]:
    """
    Load documents using appropriate LangChain loaders based on file extension.

    WHY:
    - Centralized loader logic
    - Easy to extend for CSV, HTML, XLSX later
    """
    docs: List[Document] = []

    try:
        for p in paths:
            # FIX:
            # Ensure path is a Path object (defensive programming)
            p = Path(p)

            ext = p.suffix.lower()

            if ext not in SUPPORTED_EXTENSIONS:
                log.warning("Unsupported extension skipped", path=str(p))
                continue

            if ext == ".pdf":
                loader = PyPDFLoader(str(p))
            elif ext == ".docx":
                loader = Docx2txtLoader(str(p))
            elif ext == ".txt":
                loader = TextLoader(str(p), encoding="utf-8")
            else:
                # This should never happen due to SUPPORTED_EXTENSIONS check
                continue

            loaded = loader.load()
            docs.extend(loaded)

            log.info(
                "Loaded document",
                file=str(p),
                pages=len(loaded),
            )

        log.info("Documents loaded successfully", count=len(docs))
        return docs

    except Exception as e:
        log.error("Failed loading documents", error=str(e))
        raise DocumentPortalException(
            "Error loading documents", e
        ) from e


# ============================================================
# Concatenation Helpers (LLM-friendly formatting)
# ============================================================
def concat_for_analysis(docs: List[Document]) -> str:
    """
    Combine documents into a single analysis-friendly string.

    Output format preserves source attribution for citations.
    """
    parts: List[str] = []

    for d in docs:
        # FIX:
        # metadata may be None → avoid AttributeError
        md = d.metadata or {}

        src = (
            md.get("source")
            or md.get("file_path")
            or md.get("filename")
            or "unknown"
        )

        parts.append(
            f"\n--- SOURCE: {src} ---\n{d.page_content}"
        )

    return "\n".join(parts)


def concat_for_comparison(
    ref_docs: List[Document],
    act_docs: List[Document],
) -> str:
    """
    Combine reference and actual documents in a deterministic format.

    Designed for:
    - contract diffing
    - policy comparison
    - compliance review
    """
    left = concat_for_analysis(ref_docs)
    right = concat_for_analysis(act_docs)

    return (
        "<<REFERENCE_DOCUMENTS>>\n"
        f"{left}\n\n"
        "<<ACTUAL_DOCUMENTS>>\n"
        f"{right}"
    )


# ============================================================
# FastAPI Helpers
# ============================================================
class FastAPIFileAdapter:
    """
    Adapter to normalize FastAPI UploadFile to a
    `.name + .getbuffer()` interface.

    WHY:
    - Reuse existing file-saving logic
    - Avoid branching everywhere for FastAPI vs non-FastAPI uploads
    """

    def __init__(self, uf: UploadFile):
        self._uf = uf

        # FIX:
        # UploadFile exposes `filename`, not `name`
        self.name = uf.filename

    def getbuffer(self) -> bytes:
        """
        Return full file contents as bytes.

        FIX:
        - Always seek to beginning before reading
        """
        self._uf.file.seek(0)
        return self._uf.file.read()


# ============================================================
# DocHandler Compatibility Helper
# ============================================================
def read_pdf_via_handler(handler, path: str) -> str:
    """
    Read PDF via handler while remaining backward-compatible.

    WHY:
    - Supports multiple handler implementations
    - Avoids tight coupling
    """
    if hasattr(handler, "read_pdf"):
        return handler.read_pdf(path)  # type: ignore

    if hasattr(handler, "read_"):
        return handler.read_(path)  # type: ignore

    raise RuntimeError(
        "DocHandler has neither read_pdf nor read_ method."
    )

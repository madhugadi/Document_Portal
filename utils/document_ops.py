from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, TYPE_CHECKING

from fastapi import UploadFile
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)

from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException

# 🔒 TYPE-ONLY import to avoid circular dependency
if TYPE_CHECKING:
    from src.document_ingestion.data_ingestion import DocHandler


log = CustomLogger(__name__).get_logger()

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


# ============================================================
# Document Loading
# ============================================================
def load_documents(paths: Iterable[Path]) -> List[Document]:
    """
    Load documents using appropriate LangChain loaders based on file extension.
    """
    docs: List[Document] = []

    try:
        for p in paths:
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
        raise DocumentPortalException("Error loading documents", e) from e


# ============================================================
# Concatenation Helpers
# ============================================================
def concat_for_analysis(docs: List[Document]) -> str:
    parts: List[str] = []

    for d in docs:
        md = d.metadata or {}
        src = (
            md.get("source")
            or md.get("file_path")
            or md.get("filename")
            or "unknown"
        )
        parts.append(f"\n--- SOURCE: {src} ---\n{d.page_content}")

    return "\n".join(parts)


def concat_for_comparison(
    ref_docs: List[Document],
    act_docs: List[Document],
) -> str:
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
    def __init__(self, file: UploadFile):
        self.uf = file
        self.name = file.filename

    def getbuffer(self) -> bytes:
        self.uf.file.seek(0)
        return self.uf.file.read()

    def read(self, size: int = -1) -> bytes:
        return self.uf.file.read(size)


# ============================================================
# DocHandler Compatibility Helper
# ============================================================
def _read_pdf_via_handler(handler: "DocHandler", path: str) -> str:
    try:
        return handler.read_pdf(path)
    except Exception as e:
        log.error("Failed to read PDF via handler", error=str(e))
        raise DocumentPortalException(
            "Failed to read PDF via handler", e
        ) from e

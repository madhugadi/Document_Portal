import sys
from pathlib import Path
import fitz  # PyMuPDF
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException


class DocumentIngestion:
    """
    Handles PDF ingestion for document comparison.

    Folder rules (trainer-aligned):
    - Root PDFs live in: data/document_compare/
    - Each run gets its own: data/document_compare/session_<id>/
    - Root PDFs are NEVER deleted
    - Only session folders are mutable
    """

    def __init__(
        self,
        base_dir: str = r"data/document_compare",
        session_id: str | None = None,
    ):
        self.log = CustomLogger().get_logger(__name__)

        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = session_id
        self.session_dir = None

        # Create session directory ONLY if session_id is provided
        if self.session_id:
            self.session_dir = self.base_dir / self.session_id
            self.session_dir.mkdir(parents=True, exist_ok=True)

        self.log.info(
            "DocumentIngestion initialized",
            base_dir=str(self.base_dir),
            session_id=self.session_id,
        )

    # --------------------------------------------------
    # Session-safe cleanup (NEVER touches root PDFs)
    # --------------------------------------------------
    def delete_existing_files(self):
        """
        Deletes files ONLY inside the current session directory.
        Root PDFs are never deleted.
        """
        if not self.session_dir:
            self.log.info(
                "No session directory provided; skipping cleanup"
            )
            return

        try:
            for file in self.session_dir.iterdir():
                if file.is_file():
                    file.unlink()
                    self.log.info("Session file deleted", file=str(file))

            self.log.info(
                "Session directory cleaned",
                session_dir=str(self.session_dir),
            )

        except Exception as e:
            self.log.error("delete_existing_files failed", error=str(e))
            raise DocumentPortalException(
                "Error deleting session files", sys
            ) from e

    # --------------------------------------------------
    # Save uploaded PDFs (session-scoped)
    # --------------------------------------------------
    def save_uploaded_files(self, reference_file, actual_file):
        """
        Saves uploaded PDFs into the session directory.
        """
        if not self.session_dir:
            raise DocumentPortalException(
                "session_id is required to save uploaded files",
                sys,
            )

        try:
            self.delete_existing_files()

            if not reference_file.name.endswith(".pdf") or not actual_file.name.endswith(".pdf"):
                raise DocumentPortalException(
                    "Only PDF files are supported",
                    sys,
                )

            ref_path = self.session_dir / reference_file.name
            act_path = self.session_dir / actual_file.name

            with open(ref_path, "wb") as f:
                f.write(reference_file.getbuffer())

            with open(act_path, "wb") as f:
                f.write(actual_file.getbuffer())

            self.log.info(
                "Uploaded files saved to session",
                reference=ref_path.name,
                actual=act_path.name,
                session_id=self.session_id,
            )

        except Exception as e:
            self.log.error("save_uploaded_files failed", error=str(e))
            raise DocumentPortalException(
                "Error saving uploaded files", sys
            ) from e

    # --------------------------------------------------
    # Read PDF
    # --------------------------------------------------
    def read_pdf(self, pdf_path: Path) -> str:
        """
        Reads a PDF file and extracts page-wise text.
        """
        try:
            with fitz.open(pdf_path) as doc:
                if doc.is_encrypted:
                    raise ValueError("Encrypted PDF is not supported")

                pages = []

                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text = page.get_text()

                    if text.strip():
                        pages.append(
                            f"\n-- Page {page_num + 1} --\n{text}"
                        )

                self.log.info(
                    "PDF read successfully",
                    file=pdf_path.name,
                    pages=len(pages),
                )

                return "\n".join(pages)

        except Exception as e:
            self.log.error(
                "read_pdf failed",
                file=str(pdf_path),
                error=str(e),
            )
            raise DocumentPortalException(
                "Error reading PDF file", sys
            ) from e

    # --------------------------------------------------
    # Combine documents
    # --------------------------------------------------
    def combine_documents(self) -> str:
        """
        Combines PDFs either from:
        - session directory (if session_id provided)
        - root directory (read-only mode)
        """
        try:
            source_dir = self.session_dir or self.base_dir

            pdf_files = sorted(
                f for f in source_dir.iterdir()
                if f.is_file() and f.suffix == ".pdf"
            )

            if len(pdf_files) < 2:
                raise ValueError(
                    "At least two PDF files are required for comparison"
                )

            doc_parts = []

            for idx, pdf in enumerate(pdf_files):
                role = "REFERENCE DOCUMENT" if idx == 0 else "ACTUAL DOCUMENT"
                content = self.read_pdf(pdf)

                doc_parts.append(
                    f"===== {role} ({pdf.name}) =====\n{content}\n"
                )

            combined_text = "\n\n".join(doc_parts)

            self.log.info(
                "Documents combined successfully",
                source=str(source_dir),
                document_count=len(pdf_files),
            )

            return combined_text

        except Exception as e:
            self.log.error("combine_documents failed", error=str(e))
            raise DocumentPortalException(
                "Error combining documents", sys
            ) from e

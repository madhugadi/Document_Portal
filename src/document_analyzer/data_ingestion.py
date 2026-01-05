import os
import fitz  # PyMuPDF
import uuid
import sys
from datetime import datetime
from pathlib import Path
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException


class DocumentHandler:
    """
    Class to handle document ingestion and processing.
    Automatically logs all actions and supports session-based organization.
    """

    def __init__(self, data_dir=None, session_id=None):
        try:
            # Logger
            self.log = CustomLogger().get_logger(__name__)

            # Base data directory
            self.data_dir = data_dir or os.getenv(
                "DATA_STORAGE_PATH",
                os.path.join(os.getcwd(), "data", "document_analysis")
            )

            # Session ID
            self.session_id = session_id or (
                f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_"
                f"{uuid.uuid4().hex[:8]}"
            )

            # Create session directory
            self.session_path = os.path.join(self.data_dir, self.session_id)
            os.makedirs(self.session_path, exist_ok=True)

            self.log.info(
                "PDFHandler initialized",
                session_id=self.session_id,
                session_path=self.session_path
            )

        except Exception as e:
            raise DocumentPortalException(e, sys) from e

    # ------------------------------------------------------------------
    # Save PDF
    # ------------------------------------------------------------------
    def save_pdf(self, uploaded_file):
        try:
            filename = os.path.basename(uploaded_file.name)

            if not filename.lower().endswith(".pdf"):
                raise ValueError("Uploaded file is not a PDF")

            save_path = os.path.join(self.session_path, filename)

            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            self.log.info(
                "PDF saved successfully",
                file_path=save_path,
                session_id=self.session_id
            )

            return save_path

        except Exception as e:
            self.log.error("Error saving PDF", error=str(e))
            raise DocumentPortalException(e, sys) from e

    # ------------------------------------------------------------------
    # Read PDF
    # ------------------------------------------------------------------
    def read_pdf(self, pdf_path: str) -> str:
        try:
            text_chunks = []

            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc, start=1):
                    text_chunks.append(
                        f"\n-- Page {page_num} --\n{page.get_text()}"
                    )

            text = "\n".join(text_chunks)

            self.log.info(
                "PDF read successfully",
                file_path=pdf_path,
                session_id=self.session_id
            )

            return text

        except Exception as e:
            self.log.error("Error reading PDF", error=str(e))
            raise DocumentPortalException(e, sys) from e


# ----------------------------------------------------------------------
# Dummy uploaded file (for local testing)
# ----------------------------------------------------------------------
class DummyUploadedFile:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.name = Path(file_path).name

    def getbuffer(self):
        with open(self.file_path, "rb") as f:
            return f.read()


# ----------------------------------------------------------------------
# Local test runner
# ----------------------------------------------------------------------
if __name__ == "__main__":
    handler = DocumentHandler(session_id="test_session_001")

    pdf_path = (
        "C:\\Users\\Madhu Kumar\\Desktop\\My Learnings\\LLMops\\Document_Portal\\"
        "data\\document_analysis\\NIPS-2017-attention-is-all-you-need-Paper.pdf"
    )

    dummy_pdf = DummyUploadedFile(pdf_path)

    try:
        saved_path = handler.save_pdf(dummy_pdf)
        print(f"PDF saved at: {saved_path}")

        content = handler.read_pdf(saved_path)
        print(f"PDF content length: {len(content)} characters")
        print(content[:500])  # preview first 500 chars

    except DocumentPortalException as e:
        print(f"An error occurred:\n{e}")

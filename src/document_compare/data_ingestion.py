import sys
from pathlib import Path
import fitz  # PyMuPDF: used to open PDFs and extract text page-wise

from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException


class DocumentIngestion:
    """
    DocumentIngestion = "File + Text Preparation" class.

    Job:
    1) Create / manage the working folder (root + session folder)
    2) Save PDFs into the session folder
    3) Read PDFs and extract text
    4) Combine both PDF texts into one single string (input for LLM comparison)
    """

    def __init__(self, base_dir: str = r"data/document_compare", session_id: str | None = None):
        # Logger for this class (so we can see logs from ingestion flow)
        self.log = CustomLogger().get_logger(__name__)

        # Root directory where your PDFs live (trainer structure)
        # Example: data/document_compare/
        self.base_dir = Path(base_dir)

        # Ensure root directory exists (creates it if missing)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Session id is optional:
        # - If provided -> create a session folder inside base_dir
        # - If not provided -> we use base_dir as read-only source
        self.session_id = session_id
        self.session_dir = None

        # If session_id exists, create session folder:
        # Example: data/document_compare/session_20260105_123000_abcd1234/
        if self.session_id:
            self.session_dir = self.base_dir / self.session_id
            self.session_dir.mkdir(parents=True, exist_ok=True)

        # Log initialization details (helps debugging path issues)
        self.log.info(
            "DocumentIngestion initialized",
            base_dir=str(self.base_dir),
            session_id=self.session_id,
            session_dir=str(self.session_dir) if self.session_dir else None,
        )

    def save_uploaded_files(self, reference_file, actual_file):
        """
        Saves uploaded PDFs into the session directory.

        Why session directory?
        - Because you always create a NEW session for each test
        - This keeps each run isolated and avoids overwriting root PDFs
        """

        # If session_dir is missing, we don't know where to save
        # (Saving to root is not allowed in your design because root is source-of-truth)
        if not self.session_dir:
            raise DocumentPortalException("session_id is required to save uploaded files", sys)

        try:
            # Safety check: allow only PDFs
            if not reference_file.name.endswith(".pdf") or not actual_file.name.endswith(".pdf"):
                raise DocumentPortalException("Only PDF files are supported", sys)

            # Construct where the files will be saved inside the session folder
            ref_path = self.session_dir / reference_file.name
            act_path = self.session_dir / actual_file.name

            # Write the "reference" PDF bytes into the session folder
            with open(ref_path, "wb") as f:
                f.write(reference_file.getbuffer())

            # Write the "actual" PDF bytes into the session folder
            with open(act_path, "wb") as f:
                f.write(actual_file.getbuffer())

            # Log success: helpful to confirm correct save location
            self.log.info(
                "Uploaded files saved to session",
                session_id=self.session_id,
                reference=ref_path.name,
                actual=act_path.name,
                session_dir=str(self.session_dir),
            )

        except Exception as e:
            # Log the error and raise custom exception
            self.log.error("save_uploaded_files failed", error=str(e))
            raise DocumentPortalException("Error saving uploaded files", sys) from e

    def read_pdf(self, pdf_path: Path) -> str:
        """
        Reads a single PDF and returns extracted text.

        Output format includes page markers:
        -- Page 1 --
        text...
        -- Page 2 --
        text...
        """

        try:
            # Open the PDF using PyMuPDF
            with fitz.open(pdf_path) as doc:

                # Block encrypted PDFs (cannot extract without password)
                if doc.is_encrypted:
                    raise ValueError("Encrypted PDF is not supported")

                pages = []

                # Loop through pages (0-indexed in PyMuPDF)
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)

                    # Extract plain text from the page
                    text = page.get_text()

                    # Only keep meaningful pages (skip empty pages)
                    if text.strip():
                        pages.append(f"\n-- Page {page_num + 1} --\n{text}")

                # Log how many pages were extracted (debugging extraction issues)
                self.log.info(
                    "PDF read successfully",
                    file=pdf_path.name,
                    extracted_pages=len(pages),
                )

                # Return combined text for this PDF
                return "\n".join(pages)

        except Exception as e:
            # If reading fails, wrap and raise custom exception
            self.log.error("read_pdf failed", file=str(pdf_path), error=str(e))
            raise DocumentPortalException("Error reading PDF file", sys) from e

    def combine_documents(self) -> str:
        """
        Combines two PDFs into one single string that the LLM can compare.

        Behavior:
        - If session exists -> read PDFs from session folder
        - Else -> read PDFs from base_dir (root)

        Output structure:
        ===== REFERENCE DOCUMENT (file1.pdf) =====
        text...
        ===== ACTUAL DOCUMENT (file2.pdf) =====
        text...
        """

        try:
            # Choose where to read PDFs from
            # - session_dir when running tests (recommended)
            # - base_dir only if session_id not provided
            source_dir = self.session_dir or self.base_dir

            # Find only PDF files in the chosen directory
            pdf_files = sorted(
                f for f in source_dir.iterdir()
                if f.is_file() and f.suffix.lower() == ".pdf"
            )

            # Need at least 2 PDFs to compare
            if len(pdf_files) < 2:
                raise ValueError("At least two PDF files are required for comparison")

            doc_parts = []

            # For LLM clarity:
            # - First PDF = reference
            # - Second PDF = actual
            for idx, pdf in enumerate(pdf_files):
                role = "REFERENCE DOCUMENT" if idx == 0 else "ACTUAL DOCUMENT"

                # Convert each PDF to text (page-wise)
                content = self.read_pdf(pdf)

                # Add headers so LLM knows which doc is which
                doc_parts.append(f"===== {role} ({pdf.name}) =====\n{content}\n")

            # Final combined input for LLM
            combined_text = "\n\n".join(doc_parts)

            # Log success and where we read from
            self.log.info(
                "Documents combined successfully",
                source=str(source_dir),
                document_count=len(pdf_files),
            )

            return combined_text

        except Exception as e:
            self.log.error("combine_documents failed", error=str(e))
            raise DocumentPortalException("Error combining documents", sys) from e

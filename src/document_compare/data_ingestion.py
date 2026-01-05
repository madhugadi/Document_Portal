import sys
from pathlib import Path
import fitz  # PyMuPDF for reading PDF files
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException


class DocumentIngestion:
    """
    Handles PDF ingestion for document comparison.
    Responsible for:
    - Cleaning the working directory
    - Saving uploaded PDFs
    - Reading PDF text
    - Combining PDFs into a single structured string
    """

    def __init__(self, base_dir: str = r"data\document_compare"):
        # Initialize logger for this class
        self.log = CustomLogger().get_logger(__name__)

        # Base directory where PDFs will be stored temporarily
        self.base_dir = Path(base_dir)

        # Ensure the directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def delete_existing_files(self):
        """
        Deletes all files inside the base directory.
        This ensures only the latest uploaded PDFs are processed.
        """
        try:
            # Check that the directory exists and is valid
            if self.base_dir.exists() and self.base_dir.is_dir():
                # Iterate through all items in the directory
                for file in self.base_dir.iterdir():
                    # Delete only files (not subdirectories)
                    if file.is_file():
                        file.unlink()
                        self.log.info(f"File deleted successfully: {file}")

            self.log.info(f"Directory cleaned successfully: {self.base_dir}")

        except Exception as e:
            # Log and raise a custom exception if deletion fails
            self.log.error(f"Error in delete_existing_files: {e}")
            raise DocumentPortalException(
                "An Error occurred while deleting existing files.", sys
            ) from e

    def save_uploaded_files(self, reference_file, actual_file):
        """
        Saves the uploaded reference and actual PDF files
        into the base directory after clearing old files.
        """
        try:
            # Remove previously stored PDFs
            self.delete_existing_files()
            self.log.info("Existing files deleted successfully.")

            # Build full file paths
            ref_path = self.base_dir / reference_file.name
            act_path = self.base_dir / actual_file.name

            # Validate file types
            if not reference_file.name.endswith(".pdf") or not actual_file.name.endswith(".pdf"):
                raise DocumentPortalException("Only PDF files are supported.", sys)

            # Write reference PDF to disk
            with open(ref_path, "wb") as f:
                f.write(reference_file.getbuffer())

            # Write actual PDF to disk
            with open(act_path, "wb") as f:
                f.write(actual_file.getbuffer())

            self.log.info("Uploaded files saved successfully.")

        except Exception as e:
            # Log and raise a custom exception if saving fails
            self.log.error(f"Error in save_uploaded_files: {e}")
            raise DocumentPortalException(
                "An Error occurred while saving uploaded files.", sys
            ) from e

    def read_pdf(self, pdf_path: Path) -> str:
        """
        Reads a PDF file and extracts text page by page.
        Returns the extracted text as a single string.
        """
        try:
            # Open the PDF using PyMuPDF
            with fitz.open(pdf_path) as doc:
                # Block encrypted PDFs
                if doc.is_encrypted:
                    raise ValueError("The PDF document is encrypted and cannot be read.")

                all_text = []

                # Iterate through each page in the PDF
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text = page.get_text()

                    # Append non-empty text with page markers
                    if text.strip():
                        all_text.append(
                            f"\n -- Page {page_num + 1} -- \n{text}"
                        )

                self.log.info(
                    f"PDF read successfully: {pdf_path} "
                    f"(pages extracted: {len(all_text)})"
                )

                return "\n".join(all_text)

        except Exception as e:
            # Log and raise a custom exception if reading fails
            self.log.error(f"Error in read_pdf: {e}")
            raise DocumentPortalException(
                "An Error occurred while reading PDF file.", sys
            ) from e

    def combine_documents(self) -> str:
        """
        Combines all PDFs in the base directory into a single string.
        The first document is treated as the reference document,
        and the second as the actual document for comparison.
        """
        try:
            content_dict = {}

            # Read all PDF files in sorted order
            for filename in sorted(self.base_dir.iterdir()):
                if filename.is_file() and filename.suffix == ".pdf":
                    content_dict[filename.name] = self.read_pdf(filename)

            doc_parts = []

            # Assign roles to documents for clear LLM comparison
            for idx, (filename, content) in enumerate(content_dict.items()):
                role = "REFERENCE DOCUMENT" if idx == 0 else "ACTUAL DOCUMENT"
                doc_parts.append(
                    f"===== {role} ({filename}) =====\n{content}\n"
                )

            # Combine all document parts into one string
            combined_text = "\n\n".join(doc_parts)

            self.log.info("Documents combined successfully.", count=len(doc_parts))
            return combined_text

        except Exception as e:
            # Log and raise a custom exception if combining fails
            self.log.error(f"Error in combine_documents: {e}")
            raise DocumentPortalException(
                "An Error occurred while combining documents.", sys
            ) from e

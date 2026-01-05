import sys
from pathlib import Path
import fitz
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException


class DocumentIngestion:

    def __init__(self, base_dir: str = r"data\document_compare"):
        self.log = CustomLogger().get_logger(__name__)
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def delete_existing_files(self):
        """Deletes existing files in the base directory."""
        try:
            if self.base_dir.exists() and self.base_dir.is_dir():
                for file in self.base_dir.iterdir():
                    if file.is_file():
                        file.unlink()
                        self.log.info(f"File deleted successfully: {file}")

            self.log.info(f"Directory cleaned successfully: {self.base_dir}")

        except Exception as e:
            self.log.error(f"Error in delete_existing_files: {e}")
            raise DocumentPortalException(
                "An Error occurred while deleting existing files.", sys
            ) from e

    def save_uploaded_files(self, reference_file, actual_file):
        """Saves uploaded files to a designated directory."""
        try:
            self.delete_existing_files()
            self.log.info("Existing files deleted successfully.")

            ref_path = self.base_dir / reference_file.name
            act_path = self.base_dir / actual_file.name

            if not reference_file.name.endswith(".pdf") or not actual_file.name.endswith(".pdf"):
                raise DocumentPortalException("Only PDF files are supported.", sys)

            with open(ref_path, "wb") as f:
                f.write(reference_file.getbuffer())

            with open(act_path, "wb") as f:
                f.write(actual_file.getbuffer())

            self.log.info("Uploaded files saved successfully.")

        except Exception as e:
            self.log.error(f"Error in save_uploaded_files: {e}")
            raise DocumentPortalException(
                "An Error occurred while saving uploaded files.", sys
            ) from e

    def read_pdf(self, pdf_path: Path) -> str:
        try:
            with fitz.open(pdf_path) as doc:
                if doc.is_encrypted:
                    raise ValueError("The PDF document is encrypted and cannot be read.")

                all_text = []
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text = page.get_text()

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
            self.log.error(f"Error in read_pdf: {e}")
            raise DocumentPortalException(
                "An Error occurred while reading PDF file.", sys
            ) from e

    def combine_documents(self) -> str:
        """
        Combines PDFs into a single string with explicit document roles
        so the LLM can compare them correctly.
        """
        try:
            content_dict = {}

            for filename in sorted(self.base_dir.iterdir()):
                if filename.is_file() and filename.suffix == ".pdf":
                    content_dict[filename.name] = self.read_pdf(filename)

            doc_parts = []
            for idx, (filename, content) in enumerate(content_dict.items()):
                role = "REFERENCE DOCUMENT" if idx == 0 else "ACTUAL DOCUMENT"
                doc_parts.append(
                    f"===== {role} ({filename}) =====\n{content}\n"
                )

            combined_text = "\n\n".join(doc_parts)
            self.log.info("Documents combined successfully.", count=len(doc_parts))
            return combined_text

        except Exception as e:
            self.log.error(f"Error in combine_documents: {e}")
            raise DocumentPortalException(
                "An Error occurred while combining documents.", sys
            ) from e

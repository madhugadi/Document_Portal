from pathlib import Path
from datetime import datetime
import uuid

from src.document_compare.data_ingestion import DocumentIngestion
from src.document_compare.document_comparator import DocumentComparatorLLM


class FakeUpload:
    def __init__(self, file_path: Path):
        self.name = file_path.name
        self._bytes = file_path.read_bytes()

        if not self._bytes:
            raise ValueError(f"Empty file: {file_path}")

    def getbuffer(self):
        return self._bytes


def test_compare_documents():
    # ----------------------------------
    # Create a unique session
    # ----------------------------------
    session_id = (
        f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
        f"{uuid.uuid4().hex[:8]}"
    )

    # ----------------------------------
    # Resolve base directory safely
    # ----------------------------------
    PROJECT_ROOT = Path(__file__).resolve().parent
    base_dir = PROJECT_ROOT / "data" / "document_compare"

    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    # ----------------------------------
    # Dynamically discover PDFs (NO hardcoding)
    # ----------------------------------
    pdf_files = sorted(
        f for f in base_dir.iterdir()
        if f.is_file() and f.suffix.lower() == ".pdf"
    )

    if len(pdf_files) < 2:
        raise RuntimeError(
            f"Need at least 2 PDFs in {base_dir}, found {len(pdf_files)}"
        )

    ref_path, act_path = pdf_files[:2]

    print("\nUsing PDFs:")
    print("Reference:", ref_path.name)
    print("Actual   :", act_path.name)

    # ----------------------------------
    # Fake uploads
    # ----------------------------------
    ref_upload = FakeUpload(ref_path)
    act_upload = FakeUpload(act_path)

    # ----------------------------------
    # Ingestion (session-based)
    # ----------------------------------
    ingestion = DocumentIngestion(
        base_dir=str(base_dir),
        session_id=session_id,
    )

    ingestion.save_uploaded_files(ref_upload, act_upload)
    combined_text = ingestion.combine_documents()

    # ----------------------------------
    # LLM comparison
    # ----------------------------------
    comparator = DocumentComparatorLLM()
    df = comparator.compare_documents(combined_text)

    print("\nComparison Result:\n")
    print(df)


if __name__ == "__main__":
    test_compare_documents()

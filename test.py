from pathlib import Path
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
    ref_path = Path("data/document_compare/policy_notice.pdf")
    act_path = Path("data/document_compare/product_announcement.pdf")

    ref_upload = FakeUpload(ref_path)
    act_upload = FakeUpload(act_path)

    ingestion = DocumentIngestion()
    ingestion.save_uploaded_files(ref_upload, act_upload)

    combined_text = ingestion.combine_documents()

    comparator = DocumentComparatorLLM()
    result = comparator.compare_documents(combined_text)

    print(result)


if __name__ == "__main__":
    test_compare_documents()

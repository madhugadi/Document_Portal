from pathlib import Path
from datetime import datetime
import uuid

from src.document_compare.data_ingestion import DocumentIngestion
from src.document_compare.document_comparator import DocumentComparatorLLM


# class FakeUpload:
#     def __init__(self, file_path: Path):
#         self.name = file_path.name
#         self._bytes = file_path.read_bytes()

#         if not self._bytes:
#             raise ValueError(f"Empty file: {file_path}")

#     def getbuffer(self):
#         return self._bytes


# def test_compare_documents():
#     # ----------------------------------
#     # Create a unique session
#     # ----------------------------------
#     session_id = (
#         f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
#         f"{uuid.uuid4().hex[:8]}"
#     )

    # ----------------------------------
    # Resolve base directory safely
    # # ----------------------------------
    # PROJECT_ROOT = Path(__file__).resolve().parent
    # base_dir = PROJECT_ROOT / "data" / "document_compare"

    # if not base_dir.exists():
    #     raise FileNotFoundError(f"Base directory not found: {base_dir}")

    # # ----------------------------------
    # Dynamically discover PDFs (NO hardcoding)
    # ----------------------------------
#     pdf_files = sorted(
#         f for f in base_dir.iterdir()
#         if f.is_file() and f.suffix.lower() == ".pdf"
#     )

#     if len(pdf_files) < 2:
#         raise RuntimeError(
#             f"Need at least 2 PDFs in {base_dir}, found {len(pdf_files)}"
#         )

#     ref_path, act_path = pdf_files[:2]

#     print("\nUsing PDFs:")
#     print("Reference:", ref_path.name)
#     print("Actual   :", act_path.name)

#     # ----------------------------------
#     # Fake uploads
#     # ----------------------------------
#     ref_upload = FakeUpload(ref_path)
#     act_upload = FakeUpload(act_path)

#     # ----------------------------------
#     # Ingestion (session-based)
#     # ----------------------------------
#     ingestion = DocumentIngestion(
#         base_dir=str(base_dir),
#         session_id=session_id,
#     )

#     ingestion.save_uploaded_files(ref_upload, act_upload)
#     combined_text = ingestion.combine_documents()

#     # ----------------------------------
#     # LLM comparison
#     # ----------------------------------
#     comparator = DocumentComparatorLLM()
#     df = comparator.compare_documents(combined_text)

#     print("\nComparison Result:\n")
#     print(df)


# if __name__ == "__main__":
#     test_compare_documents()



# # Testing code for document chat functionality

# Testing code for document chat functionality

import sys
from pathlib import Path

from langchain_community.vectorstores import FAISS

from src.single_document_chat.data_ingestion import SingleDocIngestor
from src.single_document_chat.retrieval import ConversationalRAG
from utils.model_loader import ModelLoader


FAISS_INDEX_PATH = Path("faiss_index")
FAISS_INDEX_FILE = FAISS_INDEX_PATH / "index.faiss"


def test_conversational_rag_on_pdf(pdf_path: str, question: str):
    try:
        model_loader = ModelLoader()

        # ---------------------------------------------------------
        # Load existing FAISS index IF it actually exists
        # ---------------------------------------------------------
        if FAISS_INDEX_FILE.exists():
            print("Loading existing FAISS index...")

            embeddings = model_loader.load_embedding_model()

            vectorstore = FAISS.load_local(
                folder_path=str(FAISS_INDEX_PATH),
                embeddings=embeddings,
                allow_dangerous_deserialization=True,
            )

            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5},
            )

        # ---------------------------------------------------------
        # Otherwise ingest PDF and create FAISS index
        # ---------------------------------------------------------
        else:
            print("FAISS index not found. Ingesting PDF and creating index...")

            with open(pdf_path, "rb") as f:
                uploaded_files = [f]
                ingestor = SingleDocIngestor()
                retriever = ingestor.ingest_files(uploaded_files)

        # ---------------------------------------------------------
        # Run Conversational RAG
        # ---------------------------------------------------------
        print("Running Conversational RAG...")

        session_id = "test_conversational_rag"
        rag = ConversationalRAG(
            retriever=retriever,
            session_id=session_id,
        )

        response = rag.invoke(question)

        print("\n================ RESULT ================\n")
        print(f"Question:\n{question}\n")
        print(f"Answer:\n{response}")
        print("\n=======================================\n")

    except Exception as e:
        print(f"Test failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    pdf_path = (
        "C:\\Users\\Madhu Kumar\\Desktop\\My Learnings\\LLMops\\"
        "Document_Portal\\data\\single_document_chat\\sample.pdf"
    )

    question = (
        "What is the significance of the attention mechanism? "
        "Can you explain it in simple terms?"
    )

    if not Path(pdf_path).exists():
        print(f"PDF file does not exist at: {pdf_path}")
        sys.exit(1)

    test_conversational_rag_on_pdf(pdf_path, question)

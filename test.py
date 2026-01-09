import sys
from pathlib import Path

from src.multi_document_chat.data_ingestion import DocumentIngestor
from src.multi_document_chat.retrieval import ConversationalRAG


DATA_DIR = Path(
    "C:/Users/Madhu Kumar/Desktop/My Learnings/LLMops/"
    "Document_Portal/data/multi_doc_chat"
)


def test_multi_doc_chat(question: str):
    try:
        # ---------------------------------------------------------
        # Collect all PDFs
        # ---------------------------------------------------------
        pdf_paths = list(DATA_DIR.glob("*.pdf"))

        if not pdf_paths:
            print("No PDF files found in multi_doc_chat directory")
            sys.exit(1)

        print(f"Found {len(pdf_paths)} documents:")
        for pdf in pdf_paths:
            print(" -", pdf.name)

        # ---------------------------------------------------------
        # Ingest documents (paths, not file handles)
        # ---------------------------------------------------------
        ingestor = DocumentIngestor()
        retriever = ingestor.ingest_files(pdf_paths)

        session_id = ingestor.session_id
        print(f"\nSession created: {session_id}")

        # ---------------------------------------------------------
        # Run Conversational RAG
        # ---------------------------------------------------------
        rag = ConversationalRAG(
            session_id=session_id,
            retriever=retriever,
        )

        print("\nRunning Multi-Document Chat...\n")

        answer = rag.invoke(question)

        print("\n================ ANSWER ================\n")
        print(answer)
        print("\n=======================================\n")

    except Exception as e:
        print(f"Test failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    question = (
        "Summarize the key ideas across all documents "
        "and explain how they relate to each other."
    )

    test_multi_doc_chat(question)

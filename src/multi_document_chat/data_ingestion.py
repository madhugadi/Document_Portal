import sys
import uuid
from pathlib import Path
from datetime import datetime, timezone

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from utils.model_loader import ModelLoader


class DocumentIngestor:
    """
    Multi-document ingestion for chat sessions.
    Each session has its own document set and FAISS index.
    """

    SUPPORTED_FILE_TYPES = {".pdf", ".docx", ".txt", ".md"}

    def __init__(
        self,
        temp_dir: str = "data/multi_doc_chat",
        faiss_dir: str = "faiss_index",
        session_id: str | None = None,
    ):
        try:
            self.log = CustomLogger.get_logger(__name__)

            # Base dirs
            self.temp_dir = Path(temp_dir).resolve()
            self.faiss_dir = Path(faiss_dir).resolve()
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            self.faiss_dir.mkdir(parents=True, exist_ok=True)

            # Session
            self.session_id = session_id or (
                f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_"
                f"{uuid.uuid4().hex[:8]}"
            )

            self.session_temp_dir = self.temp_dir / self.session_id
            self.session_faiss_dir = self.faiss_dir / self.session_id
            self.session_temp_dir.mkdir(parents=True, exist_ok=True)
            self.session_faiss_dir.mkdir(parents=True, exist_ok=True)

            self.model_loader = ModelLoader()

            self.log.info(
                "DocumentIngestor initialized",
                session_id=self.session_id,
                session_temp_path=str(self.session_temp_dir),
                session_faiss_path=str(self.session_faiss_dir),
            )

        except Exception as e:
            self.log.error("Error initializing DocumentIngestor", error=str(e))
            raise DocumentPortalException("Error initializing DocumentIngestor", sys)

    def ingest_files(self, uploaded_files):
        try:
            documents = []

            for uploaded_file in uploaded_files:

                # Case 1: Local file path (Path)
                if isinstance(uploaded_file, Path):
                    ext = uploaded_file.suffix.lower()
                    source_name = uploaded_file.name
                    temp_path = self.session_temp_dir / source_name
                    temp_path.write_bytes(uploaded_file.read_bytes())

                # Case 2: File-like object (Streamlit / Flask)
                else:
                    ext = Path(uploaded_file.name).suffix.lower()
                    source_name = uploaded_file.name
                    temp_path = self.session_temp_dir / source_name
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.read())

                # Validate extension
                if ext not in self.SUPPORTED_FILE_TYPES:
                    self.log.warning(
                        "Unsupported file type",
                        filename=source_name,
                        session_id=self.session_id,
                    )
                    continue

                self.log.info(
                    "File saved",
                    filename=source_name,
                    saved_path=str(temp_path),
                    session_id=self.session_id,
                )

                # Load document
                if ext == ".pdf":
                    loader = PyPDFLoader(str(temp_path))
                elif ext == ".docx":
                    loader = Docx2txtLoader(str(temp_path))
                elif ext in {".txt", ".md"}:
                    loader = TextLoader(str(temp_path))
                else:
                    continue

                documents.extend(loader.load())

            if not documents:
                raise DocumentPortalException(
                    "No valid documents found for ingestion",
                    sys,
                )

            self.log.info(
                "Documents loaded",
                total_docs=len(documents),
                session_id=self.session_id,
            )

            return self._create_retriever(documents)

        except Exception as e:
            self.log.error(
                "Error in ingest_files",
                error=str(e),
                session_id=self.session_id,
            )
            raise DocumentPortalException("Error in ingest_files", sys)

    def _create_retriever(self, documents):
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=300,
            )

            chunks = splitter.split_documents(documents)

            self.log.info(
                "Documents split",
                total_chunks=len(chunks),
                session_id=self.session_id,
            )

            embeddings = self.model_loader.load_embedding_model()

            vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=embeddings,
            )

            vectorstore.save_local(str(self.session_faiss_dir))

            self.log.info(
                "FAISS index saved",
                session_id=self.session_id,
                path=str(self.session_faiss_dir),
            )

            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5},
            )

            self.log.info(
                "Retriever created",
                session_id=self.session_id,
            )

            return retriever

        except Exception as e:
            self.log.error("Error creating retriever", error=str(e))
            raise DocumentPortalException("Error creating retriever", sys)

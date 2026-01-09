import sys
from pathlib import Path
from datetime import datetime, timezone

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from utils.model_loader import ModelLoader


class SingleDocIngestor:
    """
    Ingestion logic for single document chat.

    STRUCTURE (same as document_compare):

    data/single_document_chat/
    ├── sample.pdf                # input PDFs (immutable, root-level)
    ├── another.pdf
    ├── session_YYYYMMDD_HHMMSS/  # session folders (runtime artifacts)
    ├── session_YYYYMMDD_HHMMSS/
    """

    def __init__(
        self,
        data_dir: str = "data/single_document_chat",
        faiss_dir: str = "faiss_index"
    ):
        try:
            self.log = CustomLogger.get_logger(__name__)

            # Canonical, normalized path (prevents Windows duplicates)
            self.data_dir = Path(data_dir).resolve()
            self.data_dir.mkdir(parents=True, exist_ok=True)

            self.faiss_dir = Path(faiss_dir).resolve()
            self.faiss_dir.mkdir(parents=True, exist_ok=True)

            self.model_loader = ModelLoader()

            self.log.info(
                "SingleDocIngestor initialized",
                data_dir=str(self.data_dir),
                faiss_dir=str(self.faiss_dir)
            )

        except Exception as e:
            self.log.error(
                "Failed to initialize SingleDocIngestor",
                error=str(e)
            )
            raise DocumentPortalException(
                "Initialization error in SingleDocIngestor",
                sys
            )

    def ingest_files(self, pdf_filenames: list[str]):
        """
        pdf_filenames:
            List of PDF filenames that already exist in `data/single_document_chat`.

        IMPORTANT:
        - PDFs are NOT copied
        - PDFs are NOT renamed
        - Sessions are directories only
        """
        try:
            documents = []

            # 1️⃣ Create session folder (exactly like document_compare)
            session_id = f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            session_dir = self.data_dir / session_id
            session_dir.mkdir(parents=True, exist_ok=True)

            self.log.info(
                "Session created",
                session_id=session_id,
                session_dir=str(session_dir)
            )

            # 2️⃣ Load PDFs from ROOT (do NOT move or copy)
            for filename in pdf_filenames:
                pdf_path = self.data_dir / filename

                if not pdf_path.exists():
                    raise FileNotFoundError(
                        f"PDF not found in single_document_chat root: {pdf_path}"
                    )

                self.log.info(
                    "Loading PDF for ingestion",
                    session_id=session_id,
                    pdf=str(pdf_path)
                )

                loader = PyPDFLoader(str(pdf_path))
                documents.extend(loader.load())

            if not documents:
                raise ValueError("No documents loaded from provided PDFs")

            return self._create_retriever(documents)

        except Exception as e:
            self.log.error(
                "Document ingestion failed",
                error=str(e)
            )
            raise DocumentPortalException(
                "Document ingestion failed",
                sys
            )

    def _create_retriever(self, documents):
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=300
            )

            chunks = splitter.split_documents(documents)

            self.log.info(
                "Documents split into chunks",
                chunk_count=len(chunks)
            )

            embeddings = self.model_loader.load_embedding_model()

            vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=embeddings
            )

            vectorstore.save_local(str(self.faiss_dir))

            self.log.info(
                "FAISS index saved",
                faiss_path=str(self.faiss_dir)
            )

            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )

            self.log.info(
                "Retriever created successfully",
                retriever_type=str(type(retriever))
            )

            return retriever

        except Exception as e:
            self.log.error(
                "Retriever creation failed",
                error=str(e)
            )
            raise DocumentPortalException(
                "Retriever creation failed",
                sys
            )

import uuid
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
    def __init__(
        self,
        data_dir: str = "data/single/document_chat",
        faiss_dir: str = "faiss_index"
    ):
        try:
            self.log = CustomLogger.get_logger(__name__)

            self.data_dir = Path(data_dir)
            self.data_dir.mkdir(parents=True, exist_ok=True)

            self.faiss_dir = Path(faiss_dir)
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

    def ingest_files(self, uploaded_files):
        """
        uploaded_files: iterable of file-like objects (e.g. Streamlit uploads)
        """
        try:
            documents = []

            for uploaded_file in uploaded_files:
                unique_filename = (
                    f"session_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_"
                    f"{uuid.uuid4().hex}.pdf"
                )
                temp_path = self.data_dir / unique_filename

                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())

                self.log.info(
                    "File saved",
                    filename=uploaded_file.name,
                    path=str(temp_path)
                )

                loader = PyPDFLoader(str(temp_path))
                docs = loader.load()
                documents.extend(docs)

            if not documents:
                raise ValueError("No documents loaded from uploaded files")

            self.log.info(
                "PDF files loaded",
                count=len(documents)
            )

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
                "Retriever created",
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

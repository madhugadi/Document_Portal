from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    Request,
)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


from typing import Any, Dict, Optional
import os

from src.document_ingestion.data_ingestion import (
    DocHandler,
    DocumentComparator,
    ChatIngestor,
    FaissManager,
)
from src.document_analyzer.data_analysis import DocumentAnalyzer
from src.document_compare.document_comparator import DocumentComparatorLLM
from src.document_chat.retrieval import ConversationalRAG

from utils.document_ops import FastAPIFileAdapter,_read_pdf_via_handler

from logger.custom_logger import CustomLogger
from pathlib import Path

UPLOAD_BASE = Path("data")
FAISS_BASE = Path("faiss_index")



# ------------------ #
# App Initialization #
# ------------------ #

app = FastAPI(title="Document Portal API", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ #
# Static & Templates #
# ------------------ #

BASE_DIR = Path(__file__).resolve().parent.parent

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)

templates = Jinja2Templates(directory=BASE_DIR / "templates")



# ------------------ #
# Routes             #
# ------------------ #

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "document-portal"}


@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)) -> Any:
    try:
        handler = DocHandler()
        save_path = handler.save_pdf(FastAPIFileAdapter(file))
        text = _read_pdf_via_handler(handler, save_path)

        analyzer = DocumentAnalyzer()
        result = analyzer.analyze_document(text)

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare")
async def compare_documents(
    reference: UploadFile = File(...),
    actual: UploadFile = File(...)
) -> Any:
    try:
        dc = DocumentComparator()

        ref_path, act_path = dc.save_pdfs(
            FastAPIFileAdapter(reference),
            FastAPIFileAdapter(actual),
        )

        combined_text = dc.combine_documents()

        comp = DocumentComparatorLLM()
        df = comp.compare_documents(combined_text)

        return {
            "rows": df.to_dict(orient="records"),
            "session_id": dc.session_id,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/index")
async def chat_build_index(
    files: list[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    use_session__dirs: bool = Form(True),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    k: int = Form(5),
) -> Any:
    try:
        wrapped_files = [FastAPIFileAdapter(f) for f in files]

        ci = ChatIngestor(
            temp_base=UPLOAD_BASE,
            faiss_base=FAISS_BASE,
            use_session__dirs=use_session__dirs,
            session_id=session_id,
        )

        ci.build_retriever(
            wrapped_files,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            k=k,
        )

        return {
            "session_id": ci.session_id,
            "k": k,
            "use_session_dirs": use_session__dirs,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/query")
async def chat_query(
    question: str = Form(...),
    session_id: Optional[str] = Form(None),
    use_session__dirs: bool = Form(True),
    k: int = Form(5),
) -> Any:
    try:
        if use_session__dirs and not session_id:
            raise ValueError(
                "session_id must be provided if use_session__dirs is True"
            )

        index_dir = (
            os.path.join(FAISS_BASE, session_id)
            if use_session__dirs
            else FAISS_BASE
        )

        if not os.path.isdir(index_dir):
            raise ValueError(
                f"FAISS index directory {index_dir} does not exist"
            )

        rag = ConversationalRAG(session_id=session_id)
        rag.load_retriever_from_faiss(index_dir)

        response = rag.invoke(question, chat_history=[])

        return {
            "answer": response,
            "session_id": session_id,
            "k": k,
            "engine": "LCEL-RAG",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

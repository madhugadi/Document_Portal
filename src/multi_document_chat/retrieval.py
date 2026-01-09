import sys
import os
from typing import List, Optional
from operator import itemgetter

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

from utils.model_loader import ModelLoader
from exception.custom_exception import DocumentPortalException
from logger.custom_logger import CustomLogger
from prompt.prompt_library import PROMPT_REGISTRY
from models.models import PromptType


class ConversationalRAG:
    """
    Conversational RAG for multi-document chat.
    Expects a retriever already built for the session.
    """

    def __init__(self, session_id: str, retriever):
        try:
            self.log = CustomLogger.get_logger(__name__)
            self.session_id = session_id

            if retriever is None:
                raise ValueError("Retriever cannot be None")

            self.retriever = retriever
            self.llm = self._load_llm()

            # Prompts
            self.contextualize_prompt = PROMPT_REGISTRY[
                PromptType.CONTEXTUALIZE_QUESTION.value
            ]
            self.qa_prompt = PROMPT_REGISTRY[
                PromptType.CONTEXT_QA.value
            ]

            # Build LCEL chain
            self._build_lcel_chain()

            self.log.info(
                "ConversationalRAG initialized",
                session_id=self.session_id,
            )

        except Exception as e:
            self.log.error(
                "Failed to initialize ConversationalRAG",
                error=str(e),
                session_id=self.session_id,
            )
            raise DocumentPortalException(
                "Initialization error in ConversationalRAG",
                sys,
            )

    # ---------------------------------------------------------
    # LLM
    # ---------------------------------------------------------
    def _load_llm(self):
        try:
            llm = ModelLoader().load_llm()
            self.log.info(
                "LLM loaded",
                class_name=llm.__class__.__name__,
                session_id=self.session_id,
            )
            return llm
        except Exception as e:
            self.log.error(
                "Failed to load LLM",
                error=str(e),
                session_id=self.session_id,
            )
            raise DocumentPortalException(
                "Failed to load LLM",
                sys,
            )

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    @staticmethod
    def _format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # ---------------------------------------------------------
    # LCEL Chain
    # ---------------------------------------------------------
    def _build_lcel_chain(self):
        """
        Correct LCEL flow:
        user_input → question_rewriter (str)
                   → retriever (str)
                   → context (str)
                   → QA prompt → LLM
        """
        try:
            # Rewrite follow-up questions → STRING
            question_rewriter = (
                {
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                }
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
            )

            # Retriever MUST receive string
            retrieve_docs = (
                question_rewriter
                | self.retriever
                | self._format_docs
            )

            # Final QA chain
            self.chain = (
                {
                    "context": retrieve_docs,
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                }
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            )

            self.log.info(
                "LCEL chain built successfully",
                session_id=self.session_id,
            )

        except Exception as e:
            self.log.error(
                "Failed to build LCEL chain",
                error=str(e),
                session_id=self.session_id,
            )
            raise DocumentPortalException(
                "Failed to build LCEL chain",
                sys,
            )

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def invoke(
        self,
        user_input: str,
        chat_history: Optional[List[BaseMessage]] = None,
    ) -> str:
        try:
            chat_history = chat_history or []

            response = self.chain.invoke(
                {
                    "input": user_input,
                    "chat_history": chat_history,
                }
            )

            if not response:
                self.log.warning(
                    "No answer generated",
                    session_id=self.session_id,
                )
                return "No Answer"

            self.log.info(
                "Query answered",
                session_id=self.session_id,
                question=user_input,
                answer_preview=response[:150],
            )

            return response

        except Exception as e:
            self.log.error(
                "Failed to invoke ConversationalRAG",
                error=str(e),
                session_id=self.session_id,
            )
            raise DocumentPortalException(
                "Failed to invoke ConversationalRAG",
                sys,
            )

    # ---------------------------------------------------------
    # Optional: Load retriever from FAISS (session reuse)
    # ---------------------------------------------------------
    def load_retriever_from_faiss(self, index_path: str):
        try:
            embeddings = ModelLoader().load_embedding_model()

            if not os.path.isdir(index_path):
                raise FileNotFoundError(
                    f"FAISS index directory not found: {index_path}"
                )

            vectorstore = FAISS.load_local(
                index_path,
                embeddings,
                allow_dangerous_deserialization=True,
            )

            self.retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5},
            )

            self._build_lcel_chain()

            self.log.info(
                "Retriever loaded from FAISS",
                session_id=self.session_id,
                index_path=index_path,
            )

            return self.retriever

        except Exception as e:
            self.log.error(
                "Failed to load retriever from FAISS",
                error=str(e),
                session_id=self.session_id,
            )
            raise DocumentPortalException(
                "Failed to load retriever from FAISS",
                sys,
            )

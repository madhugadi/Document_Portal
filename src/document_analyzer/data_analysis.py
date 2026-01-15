from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from models.models import Metadata

from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from prompt.prompt_library import PROMPT_REGISTRY


class DocumentAnalyzer:
    """
    Handles document analysis using an LLM and extracts structured metadata.
    """

    def __init__(self):
        try:
            self.log = CustomLogger(__name__).get_logger()


            self.loader = ModelLoader()
            self.llm = self.loader.load_llm()

            self.parser = JsonOutputParser(pydantic_object=Metadata)
            self.fixing_parser = OutputFixingParser.from_llm(
                llm=self.llm,
                parser=self.parser
            )

            self.prompt = PROMPT_REGISTRY["document_analysis"]

            self.log.info("DocumentAnalyzer initialized successfully")

        except Exception as e:
            self.log.error("Failed to initialize DocumentAnalyzer", exc_info=True)
            raise DocumentPortalException(
                "Error initializing DocumentAnalyzer", e
            ) from e

    def analyze_document(self, document_text: str) -> dict:
        """
        Analyzes document content and returns structured metadata.
        """
        try:
            chain = self.prompt | self.llm | self.fixing_parser
            self.log.info("Document analysis chain created")

            response = chain.invoke({
                "document_text": document_text,
                "format_instructions": self.parser.get_format_instructions()
            })

            self.log.info(
                "Document metadata extracted successfully",
                extra={"fields": list(response.keys())}
            )

            return response

        except Exception as e:
            self.log.error("Error during document analysis", exc_info=True)
            raise DocumentPortalException(
                "Error analyzing document",
                e) from e
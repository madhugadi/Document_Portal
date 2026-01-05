import sys
import pandas as pd

from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException

from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import JsonOutputParser

from models.models import SummaryResponse
from prompt.prompt_library import PROMPT_REGISTRY


class DocumentComparatorLLM:

    def __init__(self):
        try:
            self.log = CustomLogger().get_logger(__name__)

            self.loader = ModelLoader()
            self.llm = self.loader.load_llm()

            self.parser = JsonOutputParser(pydantic_object=SummaryResponse)
            self.fixing_parser = OutputFixingParser.from_llm(
                llm=self.llm,
                parser=self.parser
            )

            self.prompt = PROMPT_REGISTRY["document_comparison"]
            self.chain = self.prompt | self.llm | self.fixing_parser

            self.log.info("DocumentComparatorLLM initialized")

        except Exception as e:
            raise DocumentPortalException(
                "Failed to initialize DocumentComparatorLLM", e
            ) from e

    def compare_documents(self, document_text: str) -> pd.DataFrame:
        try:
            inputs = {
                "combined_docs": document_text,
                "format_instruction": self.parser.get_format_instructions()
            }

            self.log.info("Starting document comparison")
            response = self.chain.invoke(inputs)
            self.log.info("Document comparison completed")

            return self._format_response(response)

        except Exception as e:
            raise DocumentPortalException(
                "Error during document comparison", e
            ) from e

    def _format_response(self, response_parsed: list[dict]) -> pd.DataFrame:
        try:
            df = pd.DataFrame(response_parsed)
            self.log.info("Response formatted into DataFrame")
            return df

        except Exception as e:
            self.log.error("Error formatting response", error=str(e))
            raise DocumentPortalException(
                "Error formatting response", sys
            ) from e

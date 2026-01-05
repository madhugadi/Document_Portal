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
    """
    Uses an LLM to compare two documents and return structured differences.
    The output is converted into a pandas DataFrame for easy downstream use.
    """

    def __init__(self):
        try:
            # Initialize logger for this class
            self.log = CustomLogger().get_logger(__name__)

            # Load the language model using the centralized model loader
            self.loader = ModelLoader()
            self.llm = self.loader.load_llm()

            # Initialize JSON output parser based on the expected response schema
            self.parser = JsonOutputParser(pydantic_object=SummaryResponse)

            # OutputFixingParser retries or corrects malformed LLM output
            self.fixing_parser = OutputFixingParser.from_llm(
                llm=self.llm,
                parser=self.parser
            )

            # Load the document comparison prompt from the prompt registry
            self.prompt = PROMPT_REGISTRY["document_comparison"]

            # Build the LangChain pipeline
            # Prompt → LLM → Output parser
            self.chain = self.prompt | self.llm | self.fixing_parser

            self.log.info("DocumentComparatorLLM initialized")

        except Exception as e:
            # Wrap initialization failures in a custom exception
            raise DocumentPortalException(
                "Failed to initialize DocumentComparatorLLM", e
            ) from e

    def compare_documents(self, document_text: str) -> pd.DataFrame:
        """
        Sends combined document text to the LLM and returns comparison results
        as a pandas DataFrame.
        """
        try:
            # Inputs passed to the prompt template
            inputs = {
                "combined_docs": document_text,
                "format_instruction": self.parser.get_format_instructions()
            }

            self.log.info("Starting document comparison")

            # Invoke the LLM comparison chain
            response = self.chain.invoke(inputs)

            self.log.info("Document comparison completed")

            # Convert structured LLM output into a DataFrame
            return self._format_response(response)

        except Exception as e:
            # Wrap runtime errors in a custom exception
            raise DocumentPortalException(
                "Error during document comparison", e
            ) from e

    def _format_response(self, response_parsed: list[dict]) -> pd.DataFrame:
        """
        Converts the parsed LLM response into a pandas DataFrame.
        """
        try:
            # Create DataFrame from list of comparison results
            df = pd.DataFrame(response_parsed)

            self.log.info("Response formatted into DataFrame")
            return df

        except Exception as e:
            # Log and raise error if formatting fails
            self.log.error("Error formatting response", error=str(e))
            raise DocumentPortalException(
                "Error formatting response", sys
            ) from e

import os
import sys
from dotenv import load_dotenv

from utils.config_loader import load_config

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# Optional (keep commented unless used)
# from langchain_openai import ChatOpenAI

from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException


log = CustomLogger(__name__).get_logger()


class ModelLoader:
    def __init__(self):
        load_dotenv()
        self.config = load_config()
        log.info("Configuration loaded successfully.")

        # Validate only required env vars
        self._validate_env_vars()

    def _validate_env_vars(self) -> None:
        """
        Validate environment variables based on ACTIVE providers only.
        """
        required_vars: list[str] = []

        # --- Active LLM validation ---
        llm_block = self.config.get("llm", {})
        active_llm = os.getenv("LLM_PROVIDER", "groq")

        if active_llm in llm_block:
            provider = llm_block[active_llm].get("provider")

            if provider == "groq":
                required_vars.append("GROQ_API_KEY")
            elif provider == "google":
                required_vars.append("GOOGLE_API_KEY")

        # --- Active embedding validation ---
        emb_cfg = self.config.get("embedding_model", {})
        if emb_cfg.get("active_provider") == "google":
            required_vars.append("GOOGLE_API_KEY")

        # Deduplicate
        required_vars = list(set(required_vars))

        self.api_keys = {key: os.getenv(key) for key in required_vars}
        missing = [k for k, v in self.api_keys.items() if not v]

        if missing:
            log.error(
                "Missing required environment variables",
                missing_vars=missing,
            )
            raise DocumentPortalException(
                f"Missing required environment variables: {missing}",
                sys,
            )

        log.info(
            "Environment variables validated",
            active_llm=active_llm,
            validated_keys=required_vars,
        )

    def load_embedding_model(self):
        """
        Load embedding model based on active provider in config.
        """
        try:
            log.info("Loading embedding model...")

            emb_cfg = self.config.get("embedding_model", {})
            active_provider = emb_cfg.get("active_provider", "huggingface")
            providers = emb_cfg.get("providers", {})

            if active_provider == "huggingface":
                model_name = providers["huggingface"]["model_name"]
                embeddings = HuggingFaceEmbeddings(model_name=model_name)

            elif active_provider == "google":
                model_name = providers["google"]["model_name"]
                embeddings = GoogleGenerativeAIEmbeddings(model=model_name)

            else:
                raise ValueError(f"Unsupported embedding provider: {active_provider}")

            log.info(
                "Embedding model loaded successfully",
                provider=active_provider,
                model_name=model_name,
            )
            return embeddings

        except Exception:
            log.exception("Failed to load embedding model")
            raise DocumentPortalException(
                "Failed to load embedding model",
                sys,
            )

    def load_llm(self):
        """
        Load and return the configured LLM model.
        """
        try:
            llm_block = self.config.get("llm", {})
            provider_key = os.getenv("LLM_PROVIDER", "groq")

            if provider_key not in llm_block:
                log.error(
                    "LLM provider not found in config",
                    provider=provider_key,
                )
                raise ValueError(f"LLM provider '{provider_key}' not found in config")

            llm_config = llm_block[provider_key]
            provider = llm_config.get("provider")
            model_name = llm_config.get("model_name")
            temperature = llm_config.get("temperature", 0)
            max_tokens = llm_config.get("max_output_tokens", 2048)

            log.info(
                "Loading LLM",
                provider=provider,
                model=model_name,
            )

            if provider == "google":
                return ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )

            if provider == "groq":
                return ChatGroq(
                    model=model_name,
                    temperature=temperature,
                )

            raise ValueError(f"Unsupported LLM provider: {provider}")

        except Exception:
            log.exception("Failed to load LLM")
            raise DocumentPortalException(
                "Failed to load LLM",
                sys,
            )


if __name__ == "__main__":
    loader = ModelLoader()

    embeddings = loader.load_embedding_model()
    print(f"Embedding Model Loaded: {embeddings}")
    print(f"Embedding Vector Length: {len(embeddings.embed_query('Hello, how are you?'))}")

    llm = loader.load_llm()
    print(f"LLM Loaded: {llm}")

    result = llm.invoke("Hello, how are you?")
    print(result.content if hasattr(result, "content") else result)

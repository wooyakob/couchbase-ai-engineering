"""LLM + embeddings, provider-agnostic through the OpenAI protocol (Chapter 8)."""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from . import config


def make_llm() -> ChatOpenAI:
    kwargs = {"model": config.LLM_MODEL, "temperature": 0, "api_key": config.LLM_API_KEY}
    if config.LLM_BASE_URL:
        kwargs["base_url"] = config.LLM_BASE_URL
    return ChatOpenAI(**kwargs)


def make_embeddings() -> OpenAIEmbeddings:
    kwargs = {"model": config.EMBEDDING_MODEL, "api_key": config.LLM_API_KEY}
    if config.LLM_BASE_URL:  # Capella Model Service specifics (Ch. 8 §8.2)
        kwargs.update(base_url=config.LLM_BASE_URL,
                      check_embedding_ctx_length=False,
                      tiktoken_enabled=False)
    return OpenAIEmbeddings(**kwargs)


def embed_query(embeddings: OpenAIEmbeddings, text: str) -> list[float]:
    # e5-family models expect an instruction prefix on queries (Ch. 4 §4.5)
    if config.EMBEDDING_MODEL.startswith("intfloat/e5"):
        text = f"query: {text}"
    return embeddings.embed_query(text)

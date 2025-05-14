import json
import os

from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.llms.openai import OpenAI


def get_config():
    config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "./config/key.json")
    with open(config_file) as f:
        return json.load(f)

def write_config(config):
    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
    os.makedirs(config_dir, exist_ok=True)
    config_file = os.path.join(config_dir, "key.json")
    with open(config_file, "w") as f:
        json.dump(config, f, indent=4)
    return config_file



def get_llm():
    config = get_config()
    if config["llm_model_type"] == "openai":
        return OpenAI(model=config["llm_model"], api_key=config["llm_model_api_key"])
    elif config["llm_model_type"] == "gemini":
        return Gemini(model=config["llm_model"], api_key=config["llm_model_api_key"])
    else:
        raise ValueError(f"Unsupported LLM: {config['llm']}")


def get_embedding():
    config = get_config()
    if config["embedding_model_type"] == "openai":
        return OpenAIEmbedding(model=config["embedding_model"], api_key=config["embedding_model_api_key"])
    elif config["embedding_model_type"] == "gemini":
        return GeminiEmbedding(model_name=config["embedding_model"], api_key=config["embedding_model_api_key"])
    else:
        raise ValueError(f"Unsupported embedding model: {config['embedding_model_type']}")

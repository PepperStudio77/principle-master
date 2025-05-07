import json
import os

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.deepseek import DeepSeek
from llama_index.llms.gemini import Gemini
from llama_index.llms.openai import OpenAI


def get_config():
    config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "./config/key.json")
    with open(config_file) as f:
        return json.load(f)


def get_openai_llm():
    config = get_config()
    return OpenAI(model="o1", api_key=config["openai_api_key"])

def get_gemini_llm():
    config = get_config()
    return Gemini(model="gemini-2.0-flash", api_key=config["gemini_api_key"])


def get_embedding():
    config = get_config()
    return OpenAIEmbedding(api_key=config["openai_api_key"])

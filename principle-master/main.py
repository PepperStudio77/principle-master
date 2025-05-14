import asyncio
import os

import click
from llama_index.core import Settings

from core.index import create_and_persist_index_from_path
from core.workflow import run_customise_workflow
from utils.llm import get_embedding, write_config


@click.group()
def consult():
    click.echo("Principle Master is ready!")
    pass


@click.command()
@click.option('--verbose', is_flag=True)
@click.option("--dynamic", is_flag=True)
def principle_master(verbose, dynamic):
    asyncio.run(run_customise_workflow(verbose=verbose, is_dynamic_advice_flow=dynamic))


@click.command()
def config_llm():
    # Prompt user for configuration details
    llm_model_type = input("Enter LLM model type (only support 'gemini' or 'openai'): ")
    llm_model = input("Enter LLM model (e.g., o1): ")
    embedding_model_type = input("Enter embedding model type (only support 'gemini' or 'openai'): ")
    embedding_model = input("Enter embedding model (e.g., text-embedding-ada-002): ")
    llm_model_api_key = input("Enter LLM model API key: ")
    embedding_model_api_key = input("Enter embedding model API key: ")

    # Create the configuration dictionary
    config = {
        "llm_model_type": llm_model_type,
        "llm_model": llm_model,
        "embedding_model": embedding_model,
        "embedding_model_type": embedding_model_type,
        "llm_model_api_key": llm_model_api_key,
        "embedding_model_api_key": embedding_model_api_key
    }
    write_config(config)
    print(f"Configuration saved. ")



@click.command()
@click.argument("pdf_path")
@click.option("--verbose", is_flag=True)
def index_content(pdf_path, verbose):
    print(f"Indexing pdf under this path {pdf_path}")
    embed_model = get_embedding()
    Settings.embed_model = embed_model
    create_and_persist_index_from_path(pdf_path)
    print(f" Index completed. ")


consult.add_command(principle_master)
consult.add_command(index_content)
consult.add_command(config_llm)
if __name__ == '__main__':
    consult()

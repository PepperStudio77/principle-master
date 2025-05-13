import asyncio

import click
from llama_index.core import Settings

from core.index import create_and_persist_index_from_path
from core.workflow import run_customise_workflow
from utils.llm import get_embedding


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
# consult.add_command(chat)
if __name__ == '__main__':
    consult()

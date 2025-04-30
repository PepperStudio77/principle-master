import asyncio

import click
from singleagent.full_context import chat
from multiagent.workflow import run


@click.group()
def consult():
    click.echo("Principle Master is ready!")
    pass


@click.command()
@click.argument("enquiry")
def single_agent(enquiry):
    response = chat(enquiry)
    click.echo(response.message)


@click.command()
@click.argument("enquiry")
@click.option('--verbose', is_flag=True, help='Enable verbose mode')
def multi_agent(enquiry, verbose):
    asyncio.run(run(enquiry, verbose=verbose))


# @click.command()
# def chat():
#     start_chat()


consult.add_command(single_agent)
consult.add_command(multi_agent)
# consult.add_command(chat)
if __name__ == '__main__':
    consult()

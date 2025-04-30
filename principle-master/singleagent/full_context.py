import time
from openai import OpenAI

from utils.journal_keeper import write_response_to_mark_down, Metric, Response
from utils.llm import get_config
from utils.pdf_file import load_principle_book_summary_to_string



def prompt_for_enquiry_with_full_context():
    system_prompt = ("\n "
                     "What attached below is the summary of book Principle by Ray Dalio. \n "
                     "Summary: \n {book_content}"
                     "You need to provide the answer to question consult to without seeking further clarification.\n"
                     "You should answer the question solly based on the summary of the book I attached above \n"
                     "Try to make the response concise and easy to understand.\n")

    book_content = load_principle_book_summary_to_string()
    prompt = system_prompt.format(book_content=book_content)
    return prompt

def chat(user_prompt):
    system_prompt = prompt_for_enquiry_with_full_context()
    config = get_config()
    client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
    s = time.perf_counter()
    response = client.chat.completions.create(
        model=config['model'],
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
    stream = False,
    )
    e = time.perf_counter()
    metric = Metric(response.usage.completion_tokens, response.usage.prompt_tokens, response.usage.total_tokens, e - s)
    write_response_to_mark_down(user_prompt, response, "full-context")
    return Response(response.choices[0].message.content, metric)




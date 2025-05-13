from utils.llm import get_openai_llm, get_gemini_llm
from utils.pdf_file import load_book_summary

system_prompt_template = """You are an AI assistant that provides thoughtful, practical, and *deeply personalized* suggestions by combining:
- The user's personal profile and principles
- Insights retrieved from *Principles* by Ray Dalio
Book Content: 
```
{book_content}
```
User profile:
```
{user_profile}
```
User's question:
```
{user_question}
```
"""


def get_system_prompt(book_content: str, user_profile: str, user_question: str):
    system_prompt = system_prompt_template.format(
        book_content=book_content,
        user_profile=user_profile,
        user_question=user_question
    )
    return system_prompt


def chat():
    llm = get_gemini_llm()
    user_profile = input(">>Tell me about yourself: ")
    user_question = input(">>What do you want to ask: ")
    user_profile = user_profile.strip()
    book_content = load_book_summary()
    response = llm.complete(prompt=get_system_prompt(book_content, user_profile, user_question))
    return response

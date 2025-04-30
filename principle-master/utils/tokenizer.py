import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4o-mini")

print(encoding.encode("tiktoken is great!"))
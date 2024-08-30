from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

input_text = "Using a Transformer network is simple"
print(f"Input text: {input_text}")

tokens = tokenizer(input_text)
print(f"Token inputs: {tokens}")

tokens = tokenizer.tokenize(input_text)
print(f"Tokens: {tokens}")

ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"IDs: {ids}")

decoded_string = tokenizer.decode(ids)
print(f"Decoded: {decoded_string}")
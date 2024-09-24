from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
# print(raw_datasets)

'''
DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 1725
    })
})
'''

raw_train_dataset = raw_datasets["train"]
# print(raw_train_dataset[0])

'''
{
    'sentence1': 'Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .', 
    'sentence2': 'Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .', 
    'label': 1, 
    'idx': 0
}
'''

# print(raw_train_dataset.features)

'''
{
     'sentence1': Value(dtype='string', id=None),
     'sentence2': Value(dtype='string', id=None),
     'label': ClassLabel(num_classes=2, names=['not_equivalent', 'equivalent'], names_file=None, id=None),
     'idx': Value(dtype='int32', id=None)
}
'''

from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])

inputs = tokenizer("This is the first sentence.", "This is the second one.")
print(f"Inputs: {inputs}")

'''
{ 
  'input_ids': [101, 2023, 2003, 1996, 2034, 6251, 1012, 102, 2023, 2003, 1996, 2117, 2028, 1012, 102],
  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
'''

print(tokenizer.convert_ids_to_tokens(inputs["input_ids"]))

'''
['[CLS]', 'this', 'is', 'the', 'first', 'sentence', '.', '[SEP]', 
'this', 'is', 'the', 'second', 'one', '.', '[SEP]']
'''

tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(tokenized_datasets)

'''
DatasetDict({
    train: Dataset({
        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
        num_rows: 408
    })
    test: Dataset({
        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
        num_rows: 1725
    })
})
'''
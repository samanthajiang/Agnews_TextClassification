# 参考：https://blog.csdn.net/blmoistawinde/article/details/112713318

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments
from transformers import pipeline
from datasets import load_dataset
from datasets import Dataset

train_dataset = load_dataset("ag_news", split="train[:114000]")
dev_dataset = load_dataset("ag_news", split="train[114000:]")
test_dataset = load_dataset("ag_news", split="test")
print(train_dataset)

# Apply a function to all the elements in the table (individually or in batches)
# and update the table (if function does updated examples).
train_dataset = train_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
dev_dataset = dev_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
test_dataset = test_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)


model_id = 'prajjwal1/bert-tiny'
tokenizer = AutoTokenizer.from_pretrained(model_id)

MAX_LENGTH = 256
train_dataset = train_dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
dev_dataset = dev_dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
test_dataset = test_dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
dev_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

print(train_dataset.features)
print(train_dataset[0])
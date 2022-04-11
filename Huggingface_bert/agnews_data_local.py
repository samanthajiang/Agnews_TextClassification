# data_processing: https://huggingface.co/course/chapter5/2?fw=pt

from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer,DataCollatorWithPadding
from torch.utils.data import DataLoader

# 读取本地数据并预处理：加上header，新闻标题和文章合并，保存为pandas pickle文件形式，load-dataset返回Dict形式的data
# 这样和通过load_dataset("ag_news")来下载网上数据是一样的
Path = "/Users/jiangshan/PycharmProjects/pythonProject3/data/ag_news_csv/"
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def data_loader(path):
    pd_data = pd.read_csv(path, names=["labels", "text", "text1"])
    # pd_data = pd.read_csv(path, header=None)
    pd_data["text"] = pd_data["text"] + " " + pd_data["text1"]
    pd_data.drop(columns="text1").to_pickle(path + "pd_data.pkl")
    # {'labels': [3, 3, 3, 3, 3], 'text': ["Wall St. Bear","Carlyle Looks Toward",..."Oil and Economy"]}
    data = load_dataset("pandas", data_files=path + "pd_data.pkl")
    return data

train_data = data_loader(Path + "train.csv")
agnews_test = data_loader(Path + "test.csv")
'''
DatasetDict({
    train: Dataset({
        features: ['labels', 'text'],
        num_rows: 120000
    })
})
'''

# train_test_split 自动的把 DatasetDict 分割为 train 和 test 两部分 (set the seed argument for reproducibility)
agnews_train = train_data["train"].train_test_split(train_size=0.95, seed=42)
agnews_train["validation"] = agnews_train.pop("test") # Rename the default "test" split to "validation"
'''
DatasetDict({
    train: Dataset({
        features: ['labels', 'text'],
        num_rows: 114000
    })
    test: Dataset({
        features: ['labels', 'text'],
        num_rows: 6000
    })
})
'''

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)

tokenized_agnews_train = agnews_train.map(tokenize_function, remove_columns= "text", batched = True)
tokenized_agnews_test = agnews_test.map(tokenize_function, remove_columns= "text", batched = True)
print(tokenized_agnews_test)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataloader = DataLoader(tokenized_agnews_train["train"], batch_size=64, shuffle=True, collate_fn=data_collator)
valid_dataloader = DataLoader(tokenized_agnews_train["validation"], batch_size=64, shuffle=True, collate_fn=data_collator)
test_dataloader = DataLoader(tokenized_agnews_test["train"], batch_size=64, shuffle=True, collate_fn=data_collator)



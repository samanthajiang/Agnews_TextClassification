
# pytorch的neural network接受的是tensor，并且是不带label数据的。
# 按照 https://pytorch.org/tutorials/beginner/basics/data_tutorial.html 创建Datasets
# Agnews Pytorch官方教程：https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html

import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader, random_split
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer

Path = '/Users/jiangshan/PycharmProjects/pythonProject3/data/ag_news_csv/'
tokenizer = get_tokenizer('basic_english')

data = pd.read_csv(Path + 'train.csv', header=None)


def yield_tokens(data_iter):
    for i in range(len(data.index)):
        data_sentence_list = data_iter.iloc[i, 1:].tolist()
        data_sentence_list_cat = data_sentence_list[0] + data_sentence_list[1]
        yield tokenizer(data_sentence_list_cat)


vocab = build_vocab_from_iterator(yield_tokens(data), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# 创建一个继承Abstract class Dataset的类，包含__init__,__getitem__, __len__
# All datasets that represent a map from keys to data samples should subclass Dataset的类
# 在类的__init__函数中完成csv文件的读取工作
# 在类的__getitem__函数中完成图片的读取工作。这样是为了减小内存开销，只要在需要用到的时候才将图片读入。
# 最终返回的样本数据是一个字典形式的，如下所示：{‘label':xxx,'text':xxx}
# DataFrame.as_matrix(self, columns=None): Convert the frame to its Numpy-array representation

class Ag_news_data(Dataset):
    # init 读取csv数据
    def __init__(self, path):
        self.data = pd.read_csv(path, header=None)

    def __len__(self):
        return len(self.data)

    # 注意getitem对象可以用agnews_sample[i]这样的形式调用的，而不用写成gnews_sample.__getitem__(0)
    # 这是因为继承的Dataset的父类实际上一个iterator
    def __getitem__(self, idx):
        data_label = int(self.data.iloc[idx, 0])
        data_sentence = self.data.iloc[idx, 1:].tolist()
        data_sentence_cat = data_sentence[0] + data_sentence[1]
        # tokenized_sentence = tokenizer(data_sentence_cat)
        # tokenized_sentence_index = vocab(tokenized_sentence)
        sample = {"label": data_label, "text": data_sentence_cat}
        return sample


train_dataset = Ag_news_data(Path + 'train.csv')
agnews_test = Ag_news_data(Path + 'test.csv')

num_train = int(len(train_dataset) * 0.95)
train_data, valid_data = random_split(train_dataset,[num_train,len(train_dataset)-num_train])
agnews_train = train_data
agnews_valid = valid_data

# DataLoader是一个iterator对象
# DataLoader传入的是上面的Ag_news_data对象，可以自动iterate整个数据集
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(label_pipeline(batch[0][_label]))
        processed_text = torch.tensor(text_pipeline(batch[0][_text]), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

train_dataloader = DataLoader(agnews_train, batch_size=64, shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(agnews_valid, batch_size=64, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(agnews_test, batch_size=64, shuffle=True, collate_fn=collate_batch)

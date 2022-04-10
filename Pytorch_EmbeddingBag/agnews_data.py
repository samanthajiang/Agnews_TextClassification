# pytorch的neural network接受的是tensor，并且是不带label数据的。
# 按照 https://pytorch.org/tutorials/beginner/basics/data_tutorial.html 创建Datasets

import pandas as pd
import torch
import pandas as np
from transformers import AutoTokenizer

from torch.utils.data import Dataset,DataLoader,random_split
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from datasets import dataset_dict

Path = '/Users/jiangshan/PycharmProjects/pythonProject3/data/ag_news_csv/'
Root = '/Users/jiangshan/PycharmProjects/pythonProject3/data/ag_news_csv/'
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
sequence = "In a hole in the ground there lived a hobbit."
#
'''
data = pd.read_csv(Path, header=None)
data_label = data.iloc[0,0]
data_sentence = data.iloc[0,1:]
print(data_label)
print(data_sentence)
print(data_sentence.shape)
data_sentence_list = data_sentence.tolist()
print(data_sentence_list)
tokenized_input = tokenizer(data_sentence_list)
print(tokenized_input)
tokenized_input_together = tokenized_input["input_ids"][0]+tokenized_input["input_ids"][1]
print(tokenized_input_together)
'''
# 创建一个继承Abstract class Dataset的类，包含__init__,__getitem__, __len__
# All datasets that represent a map from keys to data samples should subclass Dataset的类
# 在类的__init__函数中完成csv文件的读取工作
# 在类的__getitem__函数中完成图片的读取工作。这样是为了减小内存开销，只要在需要用到的时候才将图片读入。
# 最终返回的样本数据是一个字典形式的，如下所示：{‘image':image,'landmarks':landmarks}
# DataFrame.as_matrix(self, columns=None): Convert the frame to its Numpy-array representation

class Ag_news_data(Dataset):
    # init 读取csv数据
    def __init__(self, path, root):
        self.data = pd.read_csv(path, header=None)
        self.root = root

    def __len__(self):
        return len(self.data)

    # 注意getitem对象可以用agnews_sample[i]这样的形式调用的，而不用写成gnews_sample.__getitem__(0)
    # 这是因为继承的Dataset的父类实际上一个iterator
    def __getitem__(self, idx):
        data_label = self.data.iloc[idx, 0]
        data_sentence = self.data.iloc[idx, 1:].tolist()
        # tokenizer的输入是？
        # tokenizer　返回的其实是词典，包括 'input_ids': tensor，'token_type_ids': tensor，'attention_mask': tensor
        tokenized_sentence = tokenizer(data_sentence)
        tokenized_sentence_cat = tokenized_sentence["input_ids"][0] + tokenized_sentence["input_ids"][1]
        sample = {"label": data_label, "sentence": tokenized_sentence_cat}
        return sample

train_dataset = Ag_news_data(Path+'train.csv',Root)
train_dataset = train_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
exit()
num_train = int(len(train_dataset) * 0.95)
train_data, valid_data = random_split(train_dataset,[num_train,len(train_dataset)-num_train])
agnews_train = train_data

agnews_valid = valid_data
agnews_test = Ag_news_data(Path+'test.csv',Root)

def yield_tokens(data_iter):
    for i in range(len(agnews_train)):
        yield tokenizer(agnews_train[i]["sentence"])
vocab = build_vocab_from_iterator(yield_tokens(agnews_train),specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
# DataLoader
# DataLoader传入的是上面的Ag_news_data对象，可以自动iterate整个数据集
train_dataloader= DataLoader(agnews_train,batch_size=16,shuffle=True)
valid_dataloader= DataLoader(agnews_valid,batch_size=16,shuffle=True)
test_dataloader  = DataLoader(agnews_test,batch_size=16,shuffle=True)

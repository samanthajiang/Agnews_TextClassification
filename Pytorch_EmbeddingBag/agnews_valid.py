import torch
from agnewsdata_vocab import text_pipeline,vocab
from torch import nn
savepath = '/Users/jiangshan/PycharmProjects/pythonProject3/model.pth'
vocab_size = len(vocab)
emsize = 16
num_class = 4
class TextClassificationModel(nn.Module):
    def __init__(self,vocab_size,embed_dim,num_class):
        super().__init__()
        # The linear layer is a module that applies a linear transformation on the input using its stored weights and biases
        # nn.Sequential is an ordered container of modules. The data is passed through all the modules in the same order as defined.
        # You can use sequential containers to put together a quick network like seq_modules.
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()


    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

model = TextClassificationModel(vocab_size,emsize,num_class) # 初始化现有模型的权重参数
loaded_paras = torch.load(savepath)
model.load_state_dict(loaded_paras) # 用本地已有模型来重新初始化网络权重参数
model.eval() # 注意不要忘记


for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

ag_news_label = {1: "World",
                 2: "Sports",
                 3: "Business",
                 4: "Sci/Tec"}

def predict(text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text))
        output = model(text, torch.tensor([0]))
        # tensor.argmax：Returns the indices of the maximum value of all elements in the input tensor.
        return output.argmax(1).item() + 1

ex_text_str = "Late Monday night, some Peloton (PTON) staffers noticed they were unable to access work productivity\
    apps like Slack and Okta, which they used regularly on the job. Peloton's employees had been told about \
    a scheduled maintenance window that might cause service outages, according to one employee, but that didn't stop \
    others from bracing for the worst."
print("This is a %s news" %ag_news_label[predict(ex_text_str, text_pipeline)])

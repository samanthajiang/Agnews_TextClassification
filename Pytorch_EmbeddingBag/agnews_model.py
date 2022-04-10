
from agnewsdata_vocab import *
from torch import nn
import time

savepath = '/Users/jiangshan/PycharmProjects/pythonProject3/model.pth'

# Define the Neural Class. We define our neural network by subclassing nn.Module
# nn.Sequential is an ordered container of modules. The data is passed through all the modules in the same order as defined
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

vocab_size = len(vocab)
emsize = 16
num_class = 4
# To use the model, we pass it the input data. This executes the model’s forward, along with some background operations. Do not call model.forward() directly
model = TextClassificationModel(vocab_size, emsize, num_class)

EPOCHS = 5 # epoch
LR = 5  # learning rate
BATCH_SIZE = 16 # batch size for training

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None

def train(dataloader):
    # set the model into train_mode: weights will be updated
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        # 进行下一次batch梯度计算的时候，前一个batch的梯度计算结果，没有保留的必要了
        # 因为一个batch关于weight的导数是所有sample关于weight的导数的累加和
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        # we call optimizer.step() to adjust the parameters by the gradients collected in the backward pass.
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0
    # to not update weights
    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count


for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(test_dataloader)
    if total_accu is not None and total_accu > accu_val:
      scheduler.step()
    else:
       total_accu = accu_val
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:8.3f} '.format(epoch,
                                           time.time() - epoch_start_time,
                                           accu_val))
    print('-' * 59)
    
# In PyTorch, the learnable parameters (i.e. weights and biases) of an torch.nn.Module model are contained in the model’s parameters (accessed with model.parameters()).
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
# state_dict(): Returns a dictionary containing a whole state of the module.

torch.save(model.state_dict(), savepath)



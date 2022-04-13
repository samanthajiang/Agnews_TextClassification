# https://blog.csdn.net/weixin_43650429/article/details/106325637

from agnewsdata_higgingface_local import train_dataloader, valid_dataloader, test_dataloader
from transformers import AutoModelForSequenceClassification, AdamW, get_scheduler
import torch
from tqdm.auto import tqdm

# ====== 模型配置 ======
checkpoint = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=4)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.train()

# ====== 优化器 & 学习率调度器 ======
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# ====== 训练循环 ======
'''
训练：
    取出输入样本和标签数据, 加载这些数据到 GPU 中
    清除上次迭代的梯度计算(pytorch 中梯度是累加的（在 RNN 中有用），本例中每次迭代前需手动清零)
    前向传播
    反向传播
    使用优化器来更新参数
    监控训练过程
评估：
    取出输入样本和标签数据, 加载这些数据到 GPU 中
    前向计算
    计算 loss 并监控整个评估过程
'''
progress_bar = tqdm(range(num_training_steps))
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

# ====== 模型存储 ======
# 两种存储方式：用huggingface的save_pretrained，或torch的save
model.save_pretrained("/home/lr/jiangshan/NewsClassification/")
torch.save(model.state_dict(), "/home/lr/jiangshan/NewsClassification/model_bert_torchsave.pth", _use_new_zipfile_serialization=False)

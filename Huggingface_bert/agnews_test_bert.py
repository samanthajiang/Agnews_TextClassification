from agnewsdata_higgingface_local import test_dataloader
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

'''
# ======load and use torch.save model=======
savepath = "/home/lr/jiangshan/NewsClassification/model_bert_torchsave.pth"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=4) 
loaded_paras = torch.load(savepath)
model.load_state_dict(loaded_paras) # 用本地已有模型来重新初始化网络权重参数
model.eval() # 注意不要忘记
'''

# ======load and use huggingface.save_pretrained model======
savepath = "/home/lr/jiangshan/NewsClassification/"
model = AutoModelForSequenceClassification.from_pretrained(savepath, local_files_only=True)
model.eval()

'''
# ======打印模型的结构信息======
# 将所有模型参数转换为一个列表
params = list(model.named_parameters())
print('The BERT model has {:} different named parameters.\n'.format(len(params)))
for p in params:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

======打印结果======
bert.embeddings.word_embeddings.weight                  (30522, 768)
......
bert.pooler.dense.weight                                  (768, 768)
bert.pooler.dense.bias                                        (768,)
classifier.weight                                           (2, 768)
classifier.bias                                                 (2,)
'''

# ======打印整体accuracy======
def evaluate(dataloader):
    total_acc, total_count = 0, 0
    # to not update weights
    with torch.no_grad():
        for idx, input in enumerate(dataloader):
            predicted_label = model(**input).logits
            predicted_result = predicted_label.argmax(dim=-1)
            for i in range(input["labels"].size(0)):
                if predicted_result[i] == input["labels"][i]:
                    total_acc = total_acc + 1
            total_count += input["labels"].size(0)
    return total_acc / total_count


# predicted_label: tensor([[-1.0240e+00, -3.8687e+00, -1.7405e-01,  4.7779e+00],
#         [-1.0335e+00, -2.2713e+00,  4.7248e+00, -2.0490e+00],
#         ...
#         [-7.7889e-01, -3.8063e+00,  4.0420e+00, -7.9027e-01]])

# tensor([1, 2, 0, 3, 1, 1, 0, 1, 0, 3, 2, 2, 2, 0, 1, 3, 0, 1, 3, 1, 1, 2, 0, 1,
#         1, 0, 2, 3, 1, 3, 1, 0])

print("test accuracy: ", evaluate(test_dataloader))

'''
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
'''

# ====== 测试一句话======
ag_news_label = {1: "World",
                 2: "Sports",
                 3: "Business",
                 4: "Sci/Tec"}


def predict(text):
    with torch.no_grad():
        # 注意一定要return_tensors='pt'，即把返回变为tensor，否则无法喂给模型
        tokenized = tokenizer(text, return_tensors='pt')
        output = model(**tokenized)
        logits = output.logits
        # tensor.argmax：Returns the indices of the maximum value of all elements in the input tensor.
        result_tensor  = torch.argmax(logits, dim=-1) + 1
        result = result_tensor.item()
        # item() change 1d tensor to int
        return result


ex_text_str = "Late Monday night, some Peloton (PTON) staffers noticed they were unable to access work productivity\
    apps like Slack and Okta, which they used regularly on the job. Peloton's employees had been told about \
    a scheduled maintenance window that might cause service outages, according to one employee, but that didn't stop \
    others from bracing for the worst."
print("This is a %s news" % ag_news_label[predict(ex_text_str)])

import torch
from transformers import BertTokenizer
from model.lstm_attention import Config, LA_Model
import pickle

model=torch.load("model.pth")
model.eval()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载分词器
def load_tokenizer(filename='tokenizer.pkl'):
    with open(filename, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer
tokenizer=load_tokenizer('tokenizer.pkl')

# 输入句子
input_text = "落緊雨仲要趕住去天水圍天城路打乒乓波"

# 将输入文本转化为模型输入格式
inputs = tokenizer(input_text, return_tensors="pt")
input_ids = inputs['input_ids'].to(device)
attention_mask = inputs['attention_mask'].to(device)


# 初始输入为<START>标记
start_token_id = 0
end_token_id = 3
# input_ids = torch.cat([torch.tensor([[start_token_id]], device=device), input_ids, torch.tensor([[end_token_id]], device=device)], dim=1)
# decoder_input_ids = input_ids[:, :1]
decoder_input_ids = torch.tensor([[start_token_id]], device=device)
print(input_ids)
# 最大生成长度
max_length = 30
print("max_len",max_length)
# 存储生成的标记
generated_ids = []

# 模型推理生成序列
with torch.no_grad():

    for _ in range(max_length):

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=decoder_input_ids)
        # print(outputs["logits"].shape)
        next_token_logits = outputs["logits"].squeeze(0)[-1, :]
        next_token_id = next_token_logits.argmax(dim=-1).item()
        # print(next_token_id)
        # 将生成的标记添加到序列中
        generated_ids.append(next_token_id)
        if next_token_id == 3:  
            break
        # 更新解码器输入
        decoder_input_ids = torch.cat([decoder_input_ids, torch.tensor([[next_token_id]]).to(device)], dim=1)

# 将生成的标记转换回文本
print(generated_ids)
generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
print("input:",input_text)
print("result:",generated_text)

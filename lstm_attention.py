import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from data_process import DataProcess
from transformers import PretrainedConfig, PreTrainedModel

# 以下代码仅示例


class Config():
    def __init__(
        self,
        embedding_dim=512,
        hidden_size=512,
        num_layers=1,
        dropout=0.1,
        vocab_size=10000,
        **kwargs,
    ):
        # 存储模型参数
        self.embedding_dim=embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.vocab_size = vocab_size
        
        # 初始化父类 PretrainedConfig，并传递额外的关键字参数
        super().__init__(**kwargs)

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden, encoder_outputs):
        hidden = hidden[-1].squeeze(0).unsqueeze(2)  
        attn_weights = torch.bmm(encoder_outputs, hidden).squeeze(2)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context_vector = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context_vector, attn_weights

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, dropout=dropout, batch_first=True)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        outputs, hidden = self.lstm(embedded)
        return outputs, hidden
    
class Attention_Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout):
        super(Attention_Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = Attention(hidden_size)
        self.lstm = nn.LSTM(embedding_dim + hidden_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, hidden, encoder_outputs):
        embedded = self.embedding(input_ids)
        context_vector, attn_weights = self.attention(hidden, encoder_outputs)
        # print("embedded:",embedded.shape)
        # print(context_vector.unsqueeze(1).shape)
        lstm_input = torch.cat([embedded, context_vector.unsqueeze(1)], dim=-1)
        output, hidden = self.lstm(lstm_input, hidden)# output: [batch_size, 1, hidden_size]
        output = self.out(output.squeeze(1))# [batch_size, vocab_size]
        return output, hidden, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, config):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(config.vocab_size, config.embedding_dim, config.hidden_size, config.num_layers, config.dropout)
        self.decoder = Attention_Decoder(config.vocab_size, config.embedding_dim, config.hidden_size, config.num_layers, config.dropout)
    def forward(self, source, target):
        encoder_outputs, hidden = self.encoder(source)
        # print("encoder_outputs:", encoder_outputs.shape)
        outputs = torch.zeros(target.shape[0], target.shape[1], self.decoder.out.out_features)
        outputs=outputs.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        input_ids = target[:, 0].unsqueeze(1)
        # print("input_ids:",input_ids.shape)
        for t in range(1, target.shape[1]):
            output, hidden, _ = self.decoder(input_ids, hidden, encoder_outputs)
            outputs[:, t] = output
            input_ids = output.max(1)[1].unsqueeze(1)
            

        return outputs

class LA_Model(nn.Module):
    def __init__(self, config):
        super(LA_Model,self).__init__()
        self.model = Seq2Seq(config)
        self.loss_func = nn.CrossEntropyLoss()
    def forward(
        self,  # 接收参数需要与数据集的键对应，下列名称基本不能换
        input_ids,  # 输入张量
        attention_mask,  # 填充部分为0，避免计算
        labels,  # 目标张量
    ):
        outputs = self.model(input_ids, labels)
        # print("output",outputs.shape)
        result=outputs
        outputs = outputs.reshape(-1, outputs.size(-1))
        labels = labels.reshape(-1)

        # outputs = outputs.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        loss = self.loss_func(outputs, labels)
        return {"loss": loss, "logits": result}

# 名称可修改
import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel


class BertConfig(PretrainedConfig):
    def __init__(
        self,
        param=0,
        **kwargs,
    ):
        self.param = param
        super().__init__(**kwargs)


class Bert(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # self.embedding = BertEmbedding(config)
        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=config.hidden_size,
        #     nhead=config.num_attention_heads,
        #     dim_feedforward=config.hidden_size * 4,  # 论文中给出
        #     dropout=config.hidden_dropout_prob,
        #     activation=F.gelu,
        #     batch_first=True,
        # )
        # self.bert_encoder = nn.TransformerEncoder(
        #     encoder_layer,
        #     num_layers=config.num_hidden_layers,
        # )
        # self.mlmfc = nn.Linear(config.hidden_size, config.vocab_size)
        # self.fc = nn.Linear(config.hidden_size, config.label_vocab_size)

    def forward(self, x, position, padding_mask):
        # batchsize,seqlength
        # embed = self.embedding(x, position)
        # encoded = self.bert_encoder(embed, src_key_padding_mask=padding_mask)

        return  # output

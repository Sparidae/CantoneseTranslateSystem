# 名称可修改
import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

# 以下代码仅示例


class BertConfig(PretrainedConfig):
    def __init__(
        self,
        param=0,
        **kwargs,
    ):
        self.param = param
        super().__init__(**kwargs)


class BertForNER(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)  # 模型需要接受 config 参数并且传递给父类
        self.bert = Bert(config)
        self.loss_func = nn.CrossEntropyLoss()
        self.is_pretrain = config.is_pretrain

    def forward(
        self,  # 接收参数需要与数据集的键对应
        x,
        position,
        padding_mask,
        labels,
    ):
        pass
        # 传递给 `Trainer` 的模型必须在模型内部 `.forward()` 计算 loss ，
        # 作为元组第一个元素返回，或者返回包含 `loss` 键的字典
        # y = self.bert(x, position, padding_mask, self.is_pretrain)
        # loss = self.loss_func(y.permute(0, 2, 1), labels)
        # return loss, y
        # return {
        #     "loss": loss,
        #     "y": y,
        # }


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

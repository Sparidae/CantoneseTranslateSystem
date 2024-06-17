# 名称可修改
import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

# 以下代码仅示例


class ConfigTemplate(PretrainedConfig):
    def __init__(
        self,
        param=0,
        # params
        **kwargs,
    ):
        # 模型参数类，将参数存储起来传递给模型，
        self.param = param
        super().__init__(**kwargs)


class ModelTemplate(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # TODO 填充这部分，和用torch定义模型没什么区别
        # lstm层 torch有具体的实现，可以直接找到

    def forward(
        self,
        input_ids,  # 输入张量
        attention_mask,  # 填充部分为0，避免计算
        labels,  # 目标张量
    ):
        # TODO 前馈部分
        # lstm的话返回最后一层的输出和隐层状态张量
        return  # output


class ModelForTaskTemplate(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = ModelTemplate(config)
        # TODO 这个模型主要作为包裹Model的部分，实现损失的计算等
        # 模型需要接受 config 参数并且传递给父类，config即上面定义的
        self.loss_func = None

    def forward(
        self,  # 接收参数需要与数据集的键对应，下列名称基本不能换
        input_ids,  # 输入张量
        attention_mask,  # 填充部分为0，避免计算
        labels,  # 目标张量
    ):
        pass
        # TODO 这部分主要实现lstm一次前馈并计算loss值返回

        # 传递给 `Trainer` 的模型必须在模型内部 `.forward()` 计算 loss ，
        # 作为元组第一个元素返回，或者返回包含 `loss` 键的字典

        # y = self.bert(x, position, padding_mask, self.is_pretrain)
        # loss = self.loss_func(y.permute(0, 2, 1), labels)
        # return loss, y
        # return {
        #     "loss": loss,
        #     "y": y,
        # }

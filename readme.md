
## 粤语翻译系统


## 文件结构


```text
.
├── audio_interface.py  # 音频接口，实现读取语音输出文本等函数
├── dataset             # 数据集文件夹，存放语料
├── main.py             # 程序主要入口，实现前端到输入模型到返回语音的全部功能
├── model               # 模型包，为每个不同的模型定义一个文件
│   └── __init__.py
├── readme.md           
├── train.py            # 负责模型的训练和评估
├── ui                  # 负责前端界面的实现，可更改结构
│   └── ui.py
└── utils.py            # 存储工具函数
```

## 技术路线
1. 语音部分 主要实现一个函数，要求提供语音输入，能提取文本输出（调用api）
2. 模型部分
    1. LSTM+Attention，带注意力的LSTM
    2. LoRA微调中文预训练t5-base,参数量大概250M
    3. （可选）LoRA微调google/madlad400-3b模型，该模型也是基于T5结构，但是规模大，且在madlad400数据集上进行了预训练。
3. UI部分，采用transformers的Gradio库实现，提供输入（音频，文本）和输出（文本）的简单界面




## 粤语翻译系统

### 环境依赖

python==3.10
pip/conda
- cffi
- gevent
- greenlet
- pycparser
- six
- websocket
- websocket-client

ffmpg：需要从官网(https://ffmpeg.org/)下载后添加到环境变量

### 快速上手

语音转文本需要用到科大讯飞提供的api，在官网(https://www.xfyun.cn/)中找到 语音识别->语音听写，创建一个应用，在应用的服务接口认证信息中有APPID，APISecret，APIKey。

在主目录下创建api.json,内容为：

{
    "app_id":"你的appid",

    "api_key":"你的apikey",

    "api_secret":"你的apisecret"

}

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
1. 语音部分 实现'speech2text'函数，要求提供语音输入，能提取文本输出（调用api：科大讯飞的语音听写https://www.xfyun.cn/services/voicedictation）
2. 模型部分
    1. LSTM+Attention，带注意力的LSTM

    2. LoRA微调中文预训练t5-base,参数量大概250M

    3. （可选）LoRA微调google/madlad400-3b模型，该模型也是基于T5结构，但是规模大，且在madlad400数据集上进行了预训练。

3. UI部分，采用transformers的Gradio库实现，提供输入（音频，文本）和输出（文本）的简单界面


## 进度
Sparidae：
- [ ] LoRA微调中文预训练t5-base

ZZYF:
- [ ] LSTM+Attention

Wang：
- [x] 语音转文本功能: 调用audio_interface.py中的speech2text函数，传参为音频地址，返回文本字符串。
- [x] 前端界面
- [ ] 前后端接口

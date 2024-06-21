
# 粤语翻译系统

## 环境依赖

环境配置

```bash
conda create -n translate python=3.10
conda activate translate
pip install -r requirements.txt
```

ffmpeg

```bash
sudo apt update
sudo apt install ffmpeg
```

主目录下创建api.json，按照以下模板配置api
```json
{
    "voice_app_id":"...",
    "voice_api_key":"...",
    "voice_api_secret":"...",

    "translate_app_id": "...",
    "translate_api_key": "...",
    "translate_api_secret": "..."
}
```


语音转文本需要用到科大讯飞提供的api，在官网[科大讯飞](https://www.xfyun.cn/)查看教程


## 文件结构


```text
.
├── dataset                     # 数据集，经过dataprocess处理后可以得到full数据集
│   ├── full
│   ├── new                     # new为接入api自行翻译数据集，质量较差，不提供
│   ├── tokenizer.json
│   └── uploads
├── interface                   # 接口包，主要包含语音转文字接口和翻译接口
│   ├── __init__.py
│   ├── audio.py                # 粤语语音转文字api
│   └── translate.py            # 翻译api
├── model                       # 自定义模型模板包
│   ├── __init__.py
│   └── template.py
├── output                      # 输出文件夹，包含 模型_数据集 的checkpoints
│   ├── human_eval.json         # 人类评估暂存json文件
│   ├── t5_lora_full
│   ├── t5_lora_new
│   ├── t5_madlad400_3b_full
│   └── t5_madlad400_3b_new
├── api.json                    # 存放api信息，自行创建
├── data_process.py             # 处理数据集
├── finetune_t5.py              # 微调mengzi-t5 
├── finetune_t5_3b.py           # 微调madlad400-3b-mt
├── human_evaluate.py           # 简单的人类测试评估脚本
├── infer_t5.py                 # mengzi-t5 模型推理接口
├── infer_t5_3b.py              # madlad400-3b-mt 模型推理接口
├── main.py                     # ⭐系统从这里运行，主要包含UI和对各种接口的调用
├── metrics.py                  # 评价指标函数
├── readme.md                   
├── requirements.txt            # 环境依赖
└── utils.py                    # 工具函数
```


## 运行说明

完全执行

1. 先运行`data_process.py`来生成full数据集，数据集会存储在`dataset/full/`文件夹下
2. 然后使用`finetune_[model].py`来微调模型，得到checkpoints，存储在`output/[checkpoint]/[time]/checkpoint_[steps]/`
3. 将对应`infer_[model].py`中的ckpt路径修改为2.得到的路径来进行推理测试
4. 得到所有的ckpt，正确填入`main.py`来执行UI界面程序。

如果是完整的仓库，只需要执行第4步即可

注意：第一次执行需要下载预训练模型，需要占用约10G存储，且下载时间较长

## 额外说明

在`human_evaluate.py`中,实现了对多模型的翻译质量的简单个人评估。

可以提供一些句子，并对所有的模型进行翻译质量的评估，最终可以得到所有模型的翻译分数，高于1 则认为是高于翻译平均水平，低于一则是水平较差，满分为2 

根据项目几位人员测试，得到的平均分数如下

|模型_数据集|人类评分|
|---|---|
|t5_full |0.64|
|t5_new |0.5|
|t5_3b_full| 1.29|
|t5_3b_new|1.10|
|api|1.16|


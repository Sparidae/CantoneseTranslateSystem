import os
import shutil
from datetime import datetime

import gradio as gr

from infer_t5 import InferT5
from infer_t5_3b import InferT53b
from interface import UPLOAD_DIR, get_speech2text_api, get_translate_api
from utils import get_logger, get_time_str

logger = get_logger("Main")

# 加载
logger.info("loading t5 checkpoint")
t5 = InferT5("output/t5_lora/Jun17_14-11-08/checkpoint-9216", mode="lora")

logger.info("loading t5_3b checkpoint")
# t5_3b = None
t5_3b = InferT53b("output/t5_madlad400_3b_new/Jun18_23-16-17/checkpoint-11264")

logger.info("loading lstm checkpoint")
lstm = None

logger.info("initializing api")
translate_api = get_translate_api()
s2t_api = get_speech2text_api()


def translate_text(yue_text, model_name, strategy):
    results = {}
    result_text = ""
    if model_name in ["t5", "ALL"]:
        text = t5.translate_yue_to_zh(
            yue_text, do_sample=False if strategy == "搜索" else True
        )
        results["t5"] = f"{text[0]}"

    if model_name in ["t5_3b", "ALL"]:
        text = t5_3b.translate_yue_to_zh(
            yue_text, do_sample=False if strategy == "搜索" else True
        )
        results["t5_3b"] = f"{text[0]}"

    if model_name in ["lstm", "ALL"]:
        # text = lstm.translate_yue_to_zh(yue_text)
        # results['lstm']= text
        pass

    if model_name in ["讯飞API", "ALL"]:
        text = translate_api.translate_yue_to_zh(yue_text)
        results["讯飞API"] = text

    for k, v in results.items():
        result_text += f"{k:<6}: {v}\n"
    return result_text


# 主函数，处理音频文件和文本输入
def process_input(audio_file, mic_file, yue_text_input, model, strategy):
    yue_text = ""
    zh_text = ""

    if model is None or strategy is None:
        return ("", "", history, "<p style='color:red'>请选择模型和生成策略</p>")
    # 检查输入的数量
    inputs_count = sum(
        [audio_file is not None, mic_file is not None, bool(yue_text_input)]
    )
    if inputs_count != 1:
        return ("", "", history, "<p style='color:red'>确保三种输入方式仅选其一</p>")

    # 处理输入文本
    if yue_text_input:
        yue_text = yue_text_input
    else:
        audio_path = None
        for audio in [audio_file, mic_file]:
            if audio is not None:
                # 保存到指定目录
                audio_path = os.path.join(
                    UPLOAD_DIR,
                    f"{get_time_str()}_{os.path.basename(audio)}",
                )
                shutil.copy(audio, audio_path)
                yue_text = s2t_api.speech2text(audio_path)
                break

    # 翻译粤语文本为简体中文
    if yue_text:
        zh_text = translate_text(yue_text, model, strategy)

    # 保存历史记录
    history.append([yue_text, zh_text])

    return yue_text, zh_text, history, "<p style='color:green'>操作成功</p>"


def update_strategy(model_name):
    strategies = {
        "ALL": ["搜索", "采样"],  # 输出全部模型的结果
        "t5_3b": ["搜索", "采样"],
        "t5": ["搜索", "采样"],
        "lstm": ["搜索"],
        "讯飞API": ["无"],
    }
    choices = strategies.get(model_name)
    return gr.update(choices=choices, value=choices[0])


# 初始化历史记录
history = []

# Gradio 界面
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="stone")
) as demo:  # slate gray zinc neutral stone red orange amber yellow lime green emerald teal cyan sky blue indigo violet purple fuchsia pink rose
    gr.HTML("""<h1 align="center">粤语-简体中文翻译系统</h1>
            <p align="center">支持语音翻译和文本翻译""")

    with gr.Row():
        model_name = gr.Dropdown(
            choices=["ALL", "t5_3b", "t5", "lstm", "讯飞API"], label="选择: 翻译模型"
        )
        strategy = gr.Dropdown(choices=["无"], label="选择: 生成策略")
        model_name.change(fn=update_strategy, inputs=model_name, outputs=strategy)

    # gr.Markdown("## 上传语音文件和输入粤语文本")
    with gr.Row():
        audio_input = gr.Audio(
            sources="upload", type="filepath", label="输入: 上传语音文件"
        )
        mic_input = gr.Audio(
            sources="microphone", type="filepath", label="输入: 用麦克风录音"
        )

    # 粤语文本输入接口
    yue_text_input = gr.Textbox(label="输入: 粤语文本", lines=2)

    # 提示信息弹出框
    alert_output = gr.HTML()

    # 处理输入的按钮
    process_button = gr.Button("点击翻译", size="lg", variant="primary")

    with gr.Column():
        # 显示粤语文本和翻译的简体中文文本
        yue_text_output = gr.Textbox(label="粤语文本", interactive=False, lines=2)
        zh_text_output = gr.Textbox(label="翻译结果", interactive=False, lines=5)

    # 显示历史记录
    history_output = gr.DataFrame(label="历史记录", headers=["粤语", "简体中文"])

    # 处理输入
    process_button.click(
        fn=process_input,
        inputs=[audio_input, mic_input, yue_text_input, model_name, strategy],
        outputs=[yue_text_output, zh_text_output, history_output, alert_output],
    )

# 运行 Gradio 应用
demo.launch()

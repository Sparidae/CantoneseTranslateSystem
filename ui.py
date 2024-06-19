import os
import shutil
from datetime import datetime

import gradio as gr

from infer_t5 import InferT5
from interface.audio import speech2text
from interface.translate import translate_yue_to_zh
from utils import get_logger

logger = get_logger("UI")

logger.info("loading model")
t5_model = InferT5("output/t5_lora/Jun17_14-11-08/checkpoint-9216", mode="lora")


# 保存上传的音频文件到指定目录
def save_audio_file(file_path):
    upload_dir = "uploads"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    new_file_path = os.path.join(
        upload_dir,
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.path.basename(file_path)}",
    )
    shutil.copy(file_path, new_file_path)
    return new_file_path


def translate_text(yue_text, model_name, strategy):
    if model_name == "t5":
        if strategy == "搜索":
            text = t5_model.translate_yue_to_zh(yue_text, do_sample=False)
            text = f"{text[0]}"
        elif strategy == "采样":
            text = t5_model.translate_yue_to_zh(yue_text, do_sample=True)
            text = f"{text[0]}"
    # elif model_name == 't5_3b':

    elif model_name == "lstm":
        pass
    elif model_name == "讯飞API":
        text = translate_yue_to_zh(yue_text)

    return text


# 主函数，处理音频文件和文本输入
def process_input(audio_file, mic_file, yue_text_input, model, strategy):
    yue_text = ""
    cn_text = ""

    if model is None or strategy is None:
        return (
            "",
            "",
            history,
            "<p style='color:red'>请选择模型和生成策略</p>",
        )
    # 检查输入的数量
    inputs_count = sum(
        [audio_file is not None, mic_file is not None, bool(yue_text_input)]
    )
    if inputs_count == 0:
        return (
            "",
            "",
            history,
            "<p style='color:red'>请上传音频文件、使用麦克风录音或输入文本</p>",
        )
    elif inputs_count > 1:
        return (
            "",
            "",
            history,
            "<p style='color:red'>请仅输入音频文件、麦克风录音或文本中的一个</p>",
        )

    # 处理上传的音频文件
    if audio_file is not None:
        audio_path = save_audio_file(audio_file)
        yue_text = speech2text(audio_path)

    # 处理麦克风录音
    if mic_file is not None:
        mic_path = save_audio_file(mic_file)
        yue_text = speech2text(mic_path)

    # 处理直接输入的粤语文本
    if yue_text_input:
        yue_text = yue_text_input

    # 翻译粤语文本为简体中文
    if yue_text:
        cn_text = translate_text(yue_text, model, strategy)

    # 保存历史记录
    history.append([yue_text, cn_text])

    return yue_text, cn_text, history, "<p style='color:green'>操作成功</p>"


def update_strategy(model_name):
    strategies = {
        "t5_3b": ["搜索", "采样"],
        "t5": ["搜索", "采样"],
        "lstm": ["搜索"],
        "讯飞API": ["无"],
    }
    choices = strategies.get(model_name)
    return gr.update(choices=choices, value=choices[0])
    # if model_name == "t5":
    #     return gr.update(choices=["搜索", "采样"], value="搜索")
    # elif model_name == "lstm":
    #     return gr.update(choices=["搜索"], value="搜索")
    # elif model_name == "讯飞API":
    #     return gr.update(choices=["无"], value="无")


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
            choices=["t5_3b", "t5", "lstm", "讯飞API"], label="选择翻译模型"
        )
        strategy = gr.Dropdown(choices=["无"], label="选择生成策略")

    # 更新策略选项
    model_name.change(fn=update_strategy, inputs=model_name, outputs=strategy)

    # 上传语音文件接口
    with gr.Group():
        # gr.Markdown("## 上传语音文件和输入粤语文本")
        with gr.Row():
            audio_input = gr.Audio(
                sources="upload", type="filepath", label="上传语音文件"
            )
            mic_input = gr.Audio(
                sources="microphone", type="filepath", label="用麦克风录音"
            )

        # 粤语文本输入接口
        yue_text_input = gr.Textbox(label="输入粤语文本直接翻译", lines=3)

    # 提示信息弹出框
    alert_output = gr.HTML()

    # 处理输入的按钮
    process_button = gr.Button("点击翻译", size="lg", variant="primary")
    with gr.Row():
        # 显示粤语文本和翻译的简体中文文本
        yue_text_output = gr.Textbox(label="粤语文本", interactive=False, lines=5)
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

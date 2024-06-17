import gradio as gr
import os
from datetime import datetime
import shutil
from interface.audio import speech2text
from interface.translate import translate_yue_to_cn

# 保存上传的音频文件到指定目录
def save_audio_file(file_path):
    upload_dir = "uploads"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    new_file_path = os.path.join(upload_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.path.basename(file_path)}")
    shutil.copy(file_path, new_file_path)
    return new_file_path

# 主函数，处理音频文件和文本输入
def process_input(audio_file, mic_file, yue_text_input):

    yue_text = ""
    cn_text = ""

    # 检查输入的数量
    inputs_count = sum([audio_file is not None, mic_file is not None, bool(yue_text_input)])
    if inputs_count == 0:
        return "", "", history, "<p style='color:red'>请上传音频文件、使用麦克风录音或输入文本</p>"
    elif inputs_count > 1:
        return "", "", history, "<p style='color:red'>请仅输入音频文件、麦克风录音或文本中的一个</p>"

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
        cn_text = translate_yue_to_cn(yue_text)

    # 保存历史记录
    history.append([yue_text, cn_text])

    return yue_text, cn_text, history, "<p style='color:green'>操作成功</p>"

# 初始化历史记录
history = []

# Gradio 界面
with gr.Blocks(theme=gr.themes.Soft(primary_hue="stone")) as demo:  # slate gray zinc neutral stone red orange amber yellow lime green emerald teal cyan sky blue indigo violet purple fuchsia pink rose
    gr.HTML("""<h1 align="center">粤语-简体中文翻译系统</h1>
            <p align="center">支持语音翻译和文本翻译""")
    
    # 上传语音文件接口
    with gr.Row():
        audio_input = gr.Audio(sources="upload", type="filepath", label="上传语音文件")
        mic_input = gr.Audio(sources="microphone", type="filepath", label="用麦克风录音")
    
    # 粤语文本输入接口
    yue_text_input = gr.Textbox(label="输入粤语文本直接翻译", lines=3)

    # 提示信息弹出框
    alert_output = gr.HTML()


    # 处理输入的按钮
    process_button = gr.Button("点击翻译",size='lg', variant='primary')
    with gr.Row():
        # 显示粤语文本和翻译的简体中文文本
        yue_text_output = gr.Textbox(label="粤语文本", interactive=False, lines=5)
        cn_text_output = gr.Textbox(label="翻译结果", interactive=False, lines=5)
    
    # 显示历史记录
    history_output = gr.DataFrame(label="历史记录", headers=["粤语", "简体中文"])

    # 处理输入
    process_button.click(
        fn=process_input,
        inputs=[audio_input, mic_input, yue_text_input],
        outputs=[yue_text_output, cn_text_output, history_output, alert_output]
    )

# 运行 Gradio 应用
demo.launch()


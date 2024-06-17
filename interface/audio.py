import _thread as thread
import base64
import hashlib
import hmac
import json
import os
import random
import ssl
import time
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time

import numpy as np
import soundfile as sf
import websocket
import whisper
from playsound import playsound
from pydub import AudioSegment

STATUS_FIRST_FRAME = 0  # 第一帧的标识
STATUS_CONTINUE_FRAME = 1  # 中间帧标识
STATUS_LAST_FRAME = 2  # 最后一帧的标识


class Ws_Param(object):
    # 初始化
    def __init__(self, APPID, APIKey, APISecret, AudioFile):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.AudioFile = AudioFile

        # 公共参数(common)
        self.CommonArgs = {"app_id": self.APPID}
        # 业务参数(business)，更多个性化参数可在官网查看
        self.BusinessArgs = {
            "domain": "iat",
            "language": "zh_cn",
            "accent": "cantonese",
            "vinfo": 1,
            "vad_eos": 10000,
        }

    # 生成url
    def create_url(self):
        url = "wss://ws-api.xfyun.cn/v2/iat"
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + "ws-api.xfyun.cn" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v2/iat " + "HTTP/1.1"
        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(
            self.APISecret.encode("utf-8"),
            signature_origin.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding="utf-8")

        authorization_origin = (
            'api_key="%s", algorithm="%s", headers="%s", signature="%s"'
            % (self.APIKey, "hmac-sha256", "host date request-line", signature_sha)
        )
        authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode(
            encoding="utf-8"
        )
        # 将请求的鉴权参数组合为字典
        v = {"authorization": authorization, "date": date, "host": "ws-api.xfyun.cn"}
        # 拼接鉴权参数，生成url
        url = url + "?" + urlencode(v)
        return url


class SpeechRecognizer:
    def __init__(self, app_id, api_key, api_secret):
        self.app_id = app_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.result_text = ""

    def on_message(self, ws, message):
        try:
            code = json.loads(message)["code"]
            sid = json.loads(message)["sid"]
            if code != 0:
                errMsg = json.loads(message)["message"]
                print("sid:%s call error:%s code is:%s" % (sid, errMsg, code))
            else:
                data = json.loads(message)["data"]["result"]["ws"]
                result = ""
                for i in data:
                    for w in i["cw"]:
                        result += w["w"]
                print(
                    "sid:%s call success!, data is:%s"
                    % (sid, json.dumps(data, ensure_ascii=False))
                )
                self.result_text += result
        except Exception as e:
            print("receive msg,but parse exception:", e)

    def on_error(self, ws, error):
        print("### error:", error)

    def on_close(self, ws, a, b):
        print("### closed ###")

    def on_open(self, ws):
        def run(*args):
            frameSize = 8000  # 每一帧的音频大小
            intervel = 0.04  # 发送音频间隔(单位:s)
            status = STATUS_FIRST_FRAME  # 音频的状态信息，标识音频是第一帧，还是中间帧、最后一帧

            with open(wsParam.AudioFile, "rb") as fp:
                while True:
                    buf = fp.read(frameSize)
                    # 文件结束
                    if not buf:
                        status = STATUS_LAST_FRAME
                    # 第一帧处理
                    if status == STATUS_FIRST_FRAME:
                        d = {
                            "common": wsParam.CommonArgs,
                            "business": wsParam.BusinessArgs,
                            "data": {
                                "status": 0,
                                "format": "audio/L16;rate=16000",
                                "audio": str(base64.b64encode(buf), "utf-8"),
                                "encoding": "raw",
                            },
                        }
                        ws.send(json.dumps(d))
                        status = STATUS_CONTINUE_FRAME
                    # 中间帧处理
                    elif status == STATUS_CONTINUE_FRAME:
                        d = {
                            "data": {
                                "status": 1,
                                "format": "audio/L16;rate=16000",
                                "audio": str(base64.b64encode(buf), "utf-8"),
                                "encoding": "raw",
                            }
                        }
                        ws.send(json.dumps(d))
                    # 最后一帧处理
                    elif status == STATUS_LAST_FRAME:
                        d = {
                            "data": {
                                "status": 2,
                                "format": "audio/L16;rate=16000",
                                "audio": str(base64.b64encode(buf), "utf-8"),
                                "encoding": "raw",
                            }
                        }
                        ws.send(json.dumps(d))
                        time.sleep(1)
                        break
                    # 模拟音频采样间隔
                    time.sleep(intervel)
            ws.close()

        thread.start_new_thread(run, ())

    def recognize(self, audio_path):
        global wsParam
        wsParam = Ws_Param(
            APPID=self.app_id,
            APISecret=self.api_secret,
            APIKey=self.api_key,
            AudioFile=audio_path,
        )
        websocket.enableTrace(False)
        wsUrl = wsParam.create_url()
        ws = websocket.WebSocketApp(
            wsUrl,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )
        ws.on_open = self.on_open
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        return self.result_text


def audio_sample(audio_path):
    # 读取音频文件
    audio = AudioSegment.from_file(audio_path)
    output_file = "dataset/yue/clips/test.wav"

    # 设置采样率和位深度
    target_sample_rate = 16000
    target_sample_width = 2  # 16-bit depth

    # 调整采样率
    audio = audio.set_frame_rate(target_sample_rate)

    # 调整位深度
    audio = audio.set_sample_width(target_sample_width)

    # 导出为新的音频文件
    audio.export(output_file, format="wav")
    return output_file


def transcribe_audio(video_path):
    model = whisper.load_model("base")
    result = model.transcribe(video_path, language="zh")
    return result["text"]


# 使用 soundfile 将 MP3 文件转换为 WAV 文件
def convert_mp3_to_wav(mp3_path):
    wav_path = mp3_path.replace(".mp3", ".wav")

    # 使用 soundfile 读取 MP3 文件并写入 WAV 文件
    data, samplerate = sf.read(mp3_path)
    sf.write(wav_path, data, samplerate)

    return wav_path


def get_random_audio_file(folder_path, file_extension=".mp3"):
    """
    从指定文件夹中随机选择一个音频文件。

    参数:
    folder_path (str): 文件夹路径。
    file_extension (str): 音频文件的扩展名，默认是 ".mp3"。

    返回:
    str: 随机选择的音频文件的路径。
    """
    # 获取指定文件夹中所有音频文件的列表
    audio_files = [
        file for file in os.listdir(folder_path) if file.endswith(file_extension)
    ]

    # 随机选择一个音频文件
    random_audio_file = random.choice(audio_files)

    # 返回完整的文件路径
    return os.path.join(folder_path, random_audio_file)


def speech2text(audio_path="dataset/yue/clips\common_voice_yue_38338687.mp3"):
    with open("api.json") as f:
        api = json.load(f)
    app_id = api["app_id"]
    api_key = api["api_key"]
    api_secret = api["api_secret"]
    audio_path = convert_mp3_to_wav(audio_path)
    audio_path = audio_sample(audio_path)
    recognizer = SpeechRecognizer(app_id, api_key, api_secret)
    text = recognizer.recognize(audio_path)
    return text


if __name__ == "__main__":
    # 设定音频文件夹路径
    folder_path = "dataset/yue/clips"

    # 获取随机选择的音频文件路径
    random_audio_path = get_random_audio_file(folder_path, file_extension=".mp3")
    print(random_audio_path)
    playsound(random_audio_path)
    audio_path = convert_mp3_to_wav(random_audio_path)
    audio_path = audio_sample(audio_path)

    # 测试讯飞API
    text = speech2text(audio_path)
    print(text)

    # 测试transcribe_audio函数
    # text_output = transcribe_audio(audio_path)
    # print(text_output)

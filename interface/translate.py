import base64
import datetime
import hashlib
import hmac
import json

import requests


class GetResult(object):
    def __init__(self, host, app_id, api_key, api_secret):
        self.APPID = app_id
        self.Secret = api_secret
        self.APIKey = api_key

        self.Host = host
        self.RequestUri = "/v2/its"
        self.url = "https://" + host + self.RequestUri
        self.HttpMethod = "POST"
        self.Algorithm = "hmac-sha256"
        self.HttpProto = "HTTP/1.1"

        curTime_utc = datetime.datetime.utcnow()
        self.Date = self.httpdate(curTime_utc)

        self.BusinessArgs = {
            "from": "yue",
            "to": "cn",
        }

    def hashlib_256(self, res):
        m = hashlib.sha256(bytes(res.encode(encoding="utf-8"))).digest()
        result = "SHA-256=" + base64.b64encode(m).decode(encoding="utf-8")
        return result

    def httpdate(self, dt):
        weekday = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][dt.weekday()]
        month = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ][dt.month - 1]
        return "%s, %02d %s %04d %02d:%02d:%02d GMT" % (
            weekday,
            dt.day,
            month,
            dt.year,
            dt.hour,
            dt.minute,
            dt.second,
        )

    def generateSignature(self, digest):
        signatureStr = "host: " + self.Host + "\n"
        signatureStr += "date: " + self.Date + "\n"
        signatureStr += (
            self.HttpMethod + " " + self.RequestUri + " " + self.HttpProto + "\n"
        )
        signatureStr += "digest: " + digest
        signature = hmac.new(
            bytes(self.Secret.encode(encoding="utf-8")),
            bytes(signatureStr.encode(encoding="utf-8")),
            digestmod=hashlib.sha256,
        ).digest()
        result = base64.b64encode(signature)
        return result.decode(encoding="utf-8")

    def init_header(self, data):
        digest = self.hashlib_256(data)
        sign = self.generateSignature(digest)
        authHeader = (
            'api_key="%s", algorithm="%s", headers="host date request-line digest", signature="%s"'
            % (self.APIKey, self.Algorithm, sign)
        )
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Method": "POST",
            "Host": self.Host,
            "Date": self.Date,
            "Digest": digest,
            "Authorization": authHeader,
        }
        return headers

    def get_body(self, text):
        content = str(base64.b64encode(text.encode("utf-8")), "utf-8")
        postdata = {
            "common": {"app_id": self.APPID},
            "business": self.BusinessArgs,
            "data": {"text": content},
        }
        body = json.dumps(postdata)
        return body

    def call_url(self, text):
        curTime_utc = datetime.datetime.utcnow()
        self.Date = self.httpdate(curTime_utc)
        if self.APPID == "" or self.APIKey == "" or self.Secret == "":
            print("Appid 或APIKey 或APISecret 为空！请打开demo代码，填写相关信息。")
        else:
            body = self.get_body(text)
            headers = self.init_header(body)
            response = requests.post(self.url, data=body, headers=headers, timeout=8)
            status_code = response.status_code
            if status_code != 200:
                print(
                    "Http请求失败，状态码："
                    + str(status_code)
                    + "，错误信息："
                    + response.text
                )
                print(
                    "请根据错误信息检查代码，接口文档：https://www.xfyun.cn/doc/nlp/xftrans/API.html"
                )
            else:
                respData = json.loads(response.text)
                if str(respData["code"]) != "0":
                    print(
                        "请前往https://www.xfyun.cn/document/error-code?code="
                        + str(respData["code"])
                        + "查询解决办法"
                    )
                else:
                    return respData["data"]["result"]["trans_result"]["dst"]

    # --------------------------------------------------------------------------------

    def translate_yue_to_zh(self, text):
        return self.call_url(text)


def get_translate_api():
    host = "itrans.xfyun.cn"
    with open("api.json") as f:
        api = json.load(f)
    app_id = api["app_id2"]
    api_key = api["api_key2"]
    api_secret = api["api_secret2"]

    translator = GetResult(host, app_id, api_key, api_secret)
    return translator


if __name__ == "__main__":
    pass
    # 设定音频文件夹路径
    # folder_path = "dataset/yue/clips"

    # # 翻译
    # text = "啲氣氛真係好好，好掂。誒，煙花，又放煙花喇，但係影出來就唔掂囖，但係喺當場睇都幾開心𡃉。"
    # translated_text = translate_yue_to_zh(text)
    # print(translated_text)

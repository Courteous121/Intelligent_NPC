import asyncio
import streamlit as st
import os
import websocket
import datetime
import hashlib
import base64
import hmac
import json
from urllib.parse import urlencode
import urllib.request
import time
import ssl
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime
import _thread as thread
import pyaudio
import wave
import requests
import webrtcvad
import numpy as np
import collections
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import edge_tts
from pydub import AudioSegment
import io
import pyaudio
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
import threading
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage



#TTS declaration
async def run_tts(text: str, output: str, voice: str ='zh-CN-YunxiNeural') -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output)

def play_audio(filename: str):
    chunk = 1024
    wf = AudioSegment.from_file(filename, format='wav')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.sample_width),
                    channels=wf.channels,
                    rate=wf.frame_rate,
                    output=True)
    data = wf.raw_data
    while len(data) > 0:
        stream.write(data[:chunk])
        data = data[chunk:]
    stream.stop_stream()
    stream.close()
    p.terminate()

#LLM declaration
# 创建 ChatOpenAI 实例
llm=ChatOpenAI (temperature=0.0,
                openai_api_key="sk-gHcXa4vzIHe69kN1vZsXLYaAidvZPG6jDTSVOmyi8Xflm1tr",
                base_url='https://api.chatanywhere.tech/v1',
                model='gpt-3.5-turbo')
#memory = ConversationBufferMemory()
#USER_AGENT = os.getenv('USER_AGENT', 'MyApp/1.0 (Ubuntu 20.04)')
#requests_kwargs = {'headers': {'User-Agent': USER_AGENT}}
# 加载百度百科文档
loader = WebBaseLoader("https://harrypotter.fandom.com/zh/wiki/%E8%A5%BF%E5%BC%97%E5%8B%92%E6%96%AF%C2%B7%E6%96%AF%E5%86%85%E6%99%AE")
docs = loader.load()
# 创建 OpenAIEmbeddings 实例
embeddings = OpenAIEmbeddings(openai_api_key="sk-gHcXa4vzIHe69kN1vZsXLYaAidvZPG6jDTSVOmyi8Xflm1tr")

# 创建文本拆分器实例并拆分文档
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

# 使用 FAISS 构建文档向量存储
vector = FAISS.from_documents(documents, embeddings)

# 创建 ChatPromptTemplate 实例
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
<context>
{context}
</context>
Question: {input}""")

# 创建文档处理工作流链
document_chain = create_stuff_documents_chain(llm, prompt)

# 创建检索链
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# 提出问题并获取答案
#response = retrieval_chain.invoke({"input": "为我介绍西弗勒斯·斯内普。"})
#print(response["answer"])
# First we need a prompt that we can pass into an LLM to generate this search query

prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
])
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)


chat_history = [HumanMessage(content="西弗勒斯·斯内普爱莉莉吗？"), AIMessage(content="爱！")]
# retriever_chain.invoke({
#     "chat_history": chat_history,
#     "input": "告诉我他怎么爱她的。"
# })
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
chat_history = [HumanMessage(content="西弗勒斯·斯内普爱莉莉吗？"), AIMessage(content="爱！")]
 
#Record declaration

STATUS_FIRST_FRAME = 0  # 第一帧的标识
STATUS_CONTINUE_FRAME = 1  # 中间帧标识
STATUS_LAST_FRAME = 2  # 最后一帧的标识

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 320  # 20ms frame at 16000Hz
#SILENCE_THRESHOLD = 2  # silence duration threshold for VAD in seconds
RECORD_SECONDS = 20 # 录音时长

final_results={}
audio_filename=r""
f_ride=open('output.txt','w')
f_ride.close()
fo=open("output.txt",'r+')


def record_audio(filename):
    audio = pyaudio.PyAudio()
    vad = webrtcvad.Vad()
    vad.set_mode(2)  # VAD模式，0-3，数字越高越敏感

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    frames = []
    print("Recording...")

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        if vad.is_speech(data, RATE):
            frames.append(data)

    print("Finished recording")
    
    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(filename, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

def start_recording(filename):
    record_thread = threading.Thread(target=record_audio, args=(filename,))
    record_thread.start()
    record_thread.join()
    st.success(f"Audio recorded and saved as {filename}")

#ASR declaration

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
        self.BusinessArgs = {"domain": "iat", "language": "zh_cn", "accent": "mandarin", "vinfo":1,"vad_eos":10000}

    # 生成url
    def create_url(self):
        url = 'wss://ws-api.xfyun.cn/v2/iat'
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + "ws-api.xfyun.cn" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v2/iat " + "HTTP/1.1"
        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
            self.APIKey, "hmac-sha256", "host date request-line", signature_sha)
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": "ws-api.xfyun.cn"
        }
        # 拼接鉴权参数，生成url
        url = url + '?' + urlencode(v)
        # print("date: ",date)
        # print("v: ",v)
        # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
        # print('websocket url :', url)
        return url


# 收到websocket消息的处理
def on_message(ws, message):
    try:
        code = json.loads(message)["code"]
        sid = json.loads(message)["sid"]
        if code != 0:
            errMsg = json.loads(message)["message"]
            print("sid:%s call error:%s code is:%s" % (sid, errMsg, code))

        else:
            data = json.loads(message)["data"]["result"]["ws"]
            # print(json.loads(message))
            result = ""
            for i in data:
                for w in i["cw"]:
                    result += w["w"]
            #print("sid:%s call success!,data is:%s" % (sid, json.dumps(data, ensure_ascii=False)))
            #print(f"识别结果：{result}")
            #print(result)
            fo.write(result)
    except Exception as e:
        print("receive msg,but parse exception:", e)



# 收到websocket错误的处理
def on_error(ws, error):
    print("### error:", error)


# 收到websocket关闭的处理
def on_close(ws,a,b):
    print("### closed ###")


# 收到websocket连接建立的处理
def on_open(ws):
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
                # 发送第一帧音频，带business 参数
                # appid 必须带上，只需第一帧发送
                if status == STATUS_FIRST_FRAME:

                    d = {"common": wsParam.CommonArgs,
                         "business": wsParam.BusinessArgs,
                         "data": {"status": 0, "format": "audio/L16;rate=16000",
                                  "audio": str(base64.b64encode(buf), 'utf-8'),
                                  "encoding": "raw"}}
                    d = json.dumps(d)
                    ws.send(d)
                    status = STATUS_CONTINUE_FRAME
                # 中间帧处理
                elif status == STATUS_CONTINUE_FRAME:
                    d = {"data": {"status": 1, "format": "audio/L16;rate=16000",
                                  "audio": str(base64.b64encode(buf), 'utf-8'),
                                  "encoding": "raw"}}
                    ws.send(json.dumps(d))
                # 最后一帧处理
                elif status == STATUS_LAST_FRAME:
                    d = {"data": {"status": 2, "format": "audio/L16;rate=16000",
                                  "audio": str(base64.b64encode(buf), 'utf-8'),
                                  "encoding": "raw"}}
                    ws.send(json.dumps(d))
                    time.sleep(1)
                    break
                # 模拟音频采样间隔
                time.sleep(intervel)
        ws.close()

    thread.start_new_thread(run, ())
    
# Streamlit应用
st.title("Slytherin")


if 'response' not in st.session_state:
    st.session_state['response'] = ""

if st.button("Recording"):
    audio_filename = "audio.wav"
    start_recording(audio_filename)
    st.write("Audio recording in progress...")
    fo.write(f"1-trial\n") 
    time1 = datetime.now()
    #wsParam = Ws_Param(APPID='11d87c57', APISecret='NWY2YTZiZTRmZmE4MTVkOTJiYWMyNTJl', APIKey='d6f1f1d1c9aa236d71fb06046f3b7aa4', AudioFile=audio_filename)
    wsParam = Ws_Param(APPID='11d87c57', APISecret='NWY2YTZiZTRmZmE4MTVkOTJiYWMyNTJl',
                       APIKey='d6f1f1d1c9aa236d71fb06046f3b7aa4',
                       AudioFile=audio_filename)
    websocket.enableTrace(False)
    wsUrl = wsParam.create_url()
    ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.on_open = on_open
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    time2 = datetime.now()
    print(time2-time1)
       
    fo.write("\n")
    fo.seek(0)
    new_content=fo.read()
    splits=new_content.split("\n")
    #print(splits)
    last=splits[-2]
    st.write(f"Recognition Result: {last}")
    response = retrieval_chain.invoke({
            "chat_history": chat_history,
            "input": last
    })

    st.session_state.response = response["answer"]
    st.write(f"Response: {st.session_state.response}")
    asyncio.run(run_tts(st.session_state.response, 'response.mp3'))
    sound = AudioSegment.from_file('response.mp3')
    sound.export('response.wav', format='wav')
    play_audio('response.wav')

    #st.write("录音结果：", st.session_state.transcript)
    #st.write("Response：", st.session_state.response)

if st.session_state.response:
    if st.button("Play Audio"):
        play_audio('response.wav')

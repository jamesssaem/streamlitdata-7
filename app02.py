#
# WebRTC 사용 가위바위보 앱 (선택).
# 

# 먼저 커맨드라인에서 다음과 같이 라이브러리 설치 필요.
# pip install streamlit
# pip install streamlit-lottie
# pip install matplotlib
# pip install numpy
# pip install pandas
# pip install pillow
# pip install mediapipe
# pip install opencv-python-headless
# pip install streamlit-webrtc

import cv2
import av
import json
import time
import random
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import handTracking as ht                   # 준비해 둔 모듈을 읽어온다.
from PIL import Image
from streamlit_lottie import st_lottie
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# 초기화 설정.
img = np.asarray(Image.open('./images/None.png'))
img0 = np.asarray(Image.open('./images/Kawi.png'))
img1 = np.asarray(Image.open('./images/Bawi.png'))
img2 = np.asarray(Image.open('./images/Bo.png'))
labels = {-1:'---', 0:'Kawi', 1:'Bawi', 2:'Bo'}
hand_images = {-1:img, 0:img0, 1:img1, 2:img2}
random.seed(time.time())                                    # 현재 시간으로 랜덤 시드 설정.
states = {'human':-1, 'computer':-1}                        # 콜백 함수와 데이터를 주고받을 딕셔너리.
lock = threading.Lock()                                     # 쓰레드 록.

# JSON을 읽어 들이는 함수.
def loadJSON(path):
    f = open(path, 'r')
    res = json.load(f)
    f.close()
    return res

# Landmarks를 받아서 마디 벡터 사이의 각도를 반환해 주는 함수.
def getAngles(lms):
    base = lms[0][1:]             # 0번 landmark의 가로, 세로 좌표.
    lms = np.array( [ (x,y) for id, x, y in lms  ] )
    vectors = lms[1:] - np.array([base]*20)                                   # 마디와 마디를 연결해서 벡터를 만든다.
    norms = np.linalg.norm(vectors, axis=1)[:, np.newaxis]           # 축의 수가 2개 되도록 구성된 norm. 
    vectors = vectors/norms                                          # 길이가 1인 벡터로 정규화.
    cos = np.einsum( 'ij,ij->i', vectors[:-1], vectors[1:])
    angles = np.arccos(cos)*180/np.pi                                # Radian => Degree 변환.
    return angles   

# 비디오 프레임에 문자열을 삽입해 주는 함수.
def insertString(_img, _text):
    cv2.putText(img=_img, text=_text,    
                org=(10,70),                             # 좌표.  
                fontFace=cv2.FONT_HERSHEY_PLAIN,         # 글꼴.
                fontScale=4,                             # 글꼴의 크기.
                color=(0,0,255),                         # 색상.
                thickness=3)                             # 두께.

# 비디오 프레임을 처해 주는 콜백 함수.
def video_frame_processor(frame):
    img = frame.to_ndarray(format="rgb24")  
    img = my_detector.findHands(img)
    lms = my_detector.getLandmarks(img)                     # 검출된 landmark 좌표점.    
    if lms:
        angles = getAngles(lms)
        angles = angles[np.newaxis, :]
        pred = knn.findNearest(angles.astype('float32'), 3) # 검출.
        human = int(pred[0])                                # 사람의 손모양.
        with lock:
            if states['human'] != human:                    # 사람의 손모양이 바뀌었을 때.
                states['human'] = human                     # 새로운 손모양 기록.
                states['computer'] = random.randrange(3)    # 새롭게 컴퓨터의 손모양을 정한다.
    else:
        human = -1
        with lock:
            states['human'] = -1                            # 사람이 손을 내지 않은 상태.
            states['computer'] = -1
    insertString(img, labels[human])                        # 이후 텍스트 삽입.
    return av.VideoFrame.from_ndarray(img, format="rgb24")  # 프레임을 원래 형식으로 되돌려서 반환.

# 머신러닝 모형 학습.
@st.cache_resource
def initML():
    df = pd.read_csv('data_train.csv')
    X = df.drop(columns=['19']).values.astype('float32')    # 자료형 주의!
    Y = df[['19']].values.astype('float32')                 # 자료형 주의!

    knn = cv2.ml.KNearest_create()
    knn.train(X, cv2.ml.ROW_SAMPLE, Y)
    return knn

knn = initML()
my_detector = ht.HandDetector()               # 검출 객체 생성.

# 로고 Lotti와 타이틀 출력.
col1, col2 = st.columns([1,2])
with col1:
    lottie = loadJSON('lottie-rock-paper-scissors.json')
    st_lottie(lottie, speed=1, loop=True, width=150, height=150)
with col2:
    ''
    ''
    st.title('가위 바위 보~')

'---'
''

col3,_, col4 = st.columns([2,1,2])
with col3:
    st.header(':blue[나]')
    # 로컬.
    # webstr = webrtc_streamer(key="example",
    #             video_frame_callback=video_frame_processor,
    #             media_stream_constraints={'video':True, 'audio':False})
    # 원격.
    webstr = webrtc_streamer(key="example",
                mode=WebRtcMode.SENDRECV,
                video_frame_callback=video_frame_processor,
                media_stream_constraints={'video':True, 'audio':False},
                rtc_configuration={ 
                    "iceServers": [{"urls": ["stun:stun2.l.google.com:19302"]}] },
                async_processing=True )    
with col4:
    st.header(':red[컴퓨터]')
    fig_place = st.empty()
    fig, ax = plt.subplots(1, 1)

#while webstr.state.playing:            # 무한 루프.
while True:                             # 무한 루프.
    with lock:
        i = states['computer']
    ax.imshow(hand_images[i])
    ax.axis('off')
    fig_place.pyplot(fig)


#
# 가위바위보 앱.
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

import cv2
import json
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import handTracking as ht                   # 준비해 둔 모듈을 읽어온다.
from PIL import Image
from streamlit_lottie import st_lottie

# 초기화 설정.
img = np.asarray(Image.open('./images/None.png'))
img0 = np.asarray(Image.open('./images/Kawi.png'))
img1 = np.asarray(Image.open('./images/Bawi.png'))
img2 = np.asarray(Image.open('./images/Bo.png'))
labels = {-1:'--', 0:'Kawi', 1:'Bawi', 2:'Bo'}
hand_images = {-1:img, 0:img0, 1:img1, 2:img2}
random.seed(time.time())                                    # 현재 시간으로 랜덤 시드 설정.

if 'show' not in st.session_state:
    st.session_state['show'] = True

if 'computer' not in st.session_state:
    st.session_state['computer'] = -1

if 'human' not in st.session_state:
    st.session_state['human'] = -1

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

# 결과를 해석해 주는 함수.
def getResult(human, computer):
    if human == 0 and computer == 0:        # 인간 가위, 컴퓨터 가위.
        res = '무승부!'
    elif human == 0 and computer == 1:      # 인간 가위, 컴퓨터 바위.
        res = '컴퓨터 승!'
    elif human == 0 and computer == 2:      # 인간 가위, 컴퓨터 보.
        res = '인간 승!'    
    elif human == 1 and computer == 0:      # 인간 바위, 컴퓨터 가위.
        res = '인간 승!'
    elif human == 1 and computer == 1:      # 인간 바위, 컴퓨터 바위.
        res = '무승부!'
    elif human == 1 and computer == 2:      # 인간 바위, 컴퓨터 보.
        res = '컴퓨터 승!'    
    elif human == 2 and computer == 0:      # 인간 보, 컴퓨터 가위.
        res = '컴퓨터 승!'
    elif human == 2 and computer == 1:      # 인간 보, 컴퓨터 바위.
        res = '인간 승!'
    elif human == 2 and computer == 2:      # 인간 보, 컴퓨터 보.
        res = '무승부!'   
    else:
        res = '판별 불가~'
    
    return res

# 이미지 프레임을 처해 주는 함수.
def image_processor(img):
    img = my_detector.findHands(img)
    lms = my_detector.getLandmarks(img)                     # 검출된 landmark 좌표점.    
    if lms:
        angles = getAngles(lms)
        angles = angles[np.newaxis, :]
        pred = knn.findNearest(angles.astype('float32'), 3) # 검출.
        human = int(pred[0])                                # 사람의 손모양.
    else:                       
        human = -1
    return human

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

img_file_buffer = st.camera_input(' ')
if not img_file_buffer:
    st.session_state['show'] = True

''
col3,col4 = st.columns(2)

result = st.empty()

with col3:
    if img_file_buffer:
        st.header(':blue[나]')
        img = Image.open(img_file_buffer)
        img_array = np.array(img)
        human = image_processor(img_array)
        st.session_state['human'] = human
        fig = plt.figure()
        plt.imshow(hand_images[human])
        plt.axis('off')
        st.pyplot(fig)
    
with col4:
    if img_file_buffer and st.session_state['show']:
        st.header(':red[컴퓨터]')
        st.session_state['show'] = False
        computer = random.randrange(3)
        st.session_state['computer'] = computer
        fig = plt.figure()
        plt.imshow(hand_images[computer])
        plt.axis('off')
        st.pyplot(fig)

        # 판정.
        result.title(f'{getResult(st.session_state["human"], st.session_state["computer"])}')


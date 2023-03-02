# Hand Tracking 모듈.

# 다음 사이트 참고해 본다.
# https://google.github.io/mediapipe/solutions/hands.html

# 먼저 커맨드라인에서 다음과 같이 라이브러리 설치 필요.
# pip install mediapipe
# pip install opencv-python

import cv2
import mediapipe as mp

class HandDetector():
    def __init__(self, mode=False, maxHands=1, detectionConf=0.5, trackConf=0.5 ):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackConf = trackConf

        self.mpDraw = mp.solutions.drawing_utils        # 검출된 점들을 이어주는 모듈.
        self.mpHands =  mp.solutions.hands              # 검출 모듈을 가져온다.
        self.hands = self.mpHands.Hands(                # Hands 객체 생성.
            static_image_mode=self.mode,                # 트래킹 병행의 의미. Default=False.
            max_num_hands=self.maxHands,                # 손의 갯수.
            min_detection_confidence=self.detectionConf,# 검출 최소 한계. Default=0.5.
            min_tracking_confidence=self.trackConf)     # 트래킹 최소 한계. Default=0.5.     

    def findHands(self, imgRGB, flag=True):
        self.res = self.hands.process(imgRGB)                   # RGB 이미지를 받아서 처리.

        if(self.res.multi_hand_landmarks):                      # 손이 검출되었다면.
            for a_hand in self.res.multi_hand_landmarks:        # 개개 손을 가져온다.
                if flag:
                    self.mpDraw.draw_landmarks(imgRGB, a_hand, self.mpHands.HAND_CONNECTIONS)   # 연결 선까지 그려준다.
        return imgRGB
    
    def getLandmarks(self, img, handNo=0):
        lms = []                                          # 반환될 landmark 의 리스트.
        if(self.res.multi_hand_landmarks):                   # 손이 검출되었다면.
            a_hand = self.res.multi_hand_landmarks[handNo]   # 특정 손을 가져온다.
            for id, lm in enumerate(a_hand.landmark):        # 0~20까지 하나씩 처리. 
                h, w, c = img.shape                          # shape속성에서 이미지의 크기 (픽셀) 추출.
                cx, cy = int(lm.x * w), int( lm.y * h)       # 실수 비율을 실제 픽셀 포지션으로 변환.    
                lms.append([id,cx,cy])             
        return lms   


def main():
    my_cap = cv2.VideoCapture(0)                  # 0 = 첫번째 카메라.
    my_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)     # 카메라 입력의 너비.
    my_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)    # 카메라 입력의 높이.
    my_detector = HandDetector()                  # 검출 객체 생성.
    while True:
        _, img = my_cap.read()               # 카메라 입력을 받는다.
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     # BGR => RGB 컬러 채널 변환.
        imgRGB = my_detector.findHands(imgRGB)            # 검출하고 시각화.
        lms = my_detector.getLandmarks(imgRGB)            # 검출된 landmark 좌표점.          

        # 검출이 성공한 경우만 출력.
        if lms:
            print(lms)

        imgBGR = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)  # RGB => BGR 역변환.
        cv2.imshow("Image", imgBGR)
        if cv2.waitKey(1) & 0xFF == ord('q'):             # 'q' 키가 눌려지면 나간다.
            break

if __name__ == "__main__":
    main()
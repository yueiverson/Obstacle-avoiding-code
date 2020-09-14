import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
from voice import playvoice
import signal
sys.path.insert(0, "build/lib.linux-armv7l-2.7/")
import VL53L1X
tof = VL53L1X.VL53L1X(i2c_bus=1, i2c_address=0x29)
tof.open()
tof.set_user_roi(VL53L1X.VL53L1xUserRoi(6, 9, 9, 6))
tof.start_ranging(3)
class VideoStream:
    """控制樹梅派相機即時串流"""
    def __init__(self,resolution=(640,480),framerate=30):
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        #於即時串流上讀取第一幀
        (self.grabbed, self.frame) = self.stream.read()

    # 用於相機停止的變數
        self.stopped = False

    def start(self):
    # 藉由執行緒開始執行FPS更新
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # 持續監看執行緒執行結果直到執行緒結束
        while True:
            # 相機關閉則執行緒執行停止
            if self.stopped:

                self.stream.release()
                return


            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

parser = argparse.ArgumentParser()
#parser.add_argument('--modeldir=',help='Folder the .tflite file is located in',required=True)
#預訓練神經網路模型位置
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='/home/pi/TFmodel/model/detect.tflite')
#預訓練神經網路標籤位置
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='/home/pi/TFmodel/model/labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='640x480')

args = parser.parse_args()
#MODEL_NAME='/home/pi/TFmodel/model'
#MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)

# 導入tensorflow 函式庫
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter       

# 讀取模型檔案位置
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
#PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)
PATH_TO_CKPT = os.path.join(CWD_PATH,GRAPH_NAME)

# Path to label map file
#PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)
PATH_TO_LABELS = os.path.join(CWD_PATH,LABELMAP_NAME)

# 載入模型標籤檔
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]


# 偵測第一個標籤會為???故將第一個標籤刪除
if labels[0] == '???':
    del(labels[0])

interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# 讀取模型
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# 初始化FPS計算
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# 初始化相機串流
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)
left_count,right_count = 0,0
while True:
    distance_cm = tof.get_distance()/10
    print("Distance: {}cm".format(distance_cm))
    time.sleep(0.1)
    if distance_cm >=150 and distance_cm <=200:
        playvoice('1.5m.m4a')
    # 第一幀時間
    t1 = cv2.getTickCount()

    # 讀取偵測串流
    frame1 = videostream.read()

    # 獲得視窗並調整視窗預期大小 [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

 
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # 物件辨識標記邊界框座標
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # 偵測類別的索引值
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # 偵測物件信心指數

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            # 取得邊界框座標
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (255, 255, 255), 4)
            gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
            blur_gray=cv2.GaussianBlur(gray,(3,3),0)
            mask = cv2.Canny(blur_gray,300,400)
            thresh =cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            x,y,w,h = cv2.boundingRect(thresh)
            ROI = frame[y:y+h,x:x+w]
            x,y,w,h =0,0,ROI.shape[1]//2,ROI.shape[0]  #將偵測視窗分為左右兩邊
            left=mask[y:y+h,x:x+w] #左邊遮罩
            right=mask[y:y+h,x+w:x+w+w] #右邊遮罩
            left_piexl = cv2.countNonZero(left) #計算左邊白點像素量
            right_piexl= cv2.countNonZero(right)#計算右邊白點像素量
            
            # 標示物件類別
            object_name = labels[int(classes[i])] # 讀取偵測結果中信心指數最高的類別，以陣列浮動表示
            label = '{}: {}%'.format(object_name, int(scores[i]*100)) # 標籤信心指數'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) #標籤顯示大小
            label_ymin = max(ymin, labelSize[1] + 10) 
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) #標籤外框
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # 標籤文字
            if object_name =='greenlight':
                playvoice('red.m4a')
                if object_name =='redlight':
                    playvoice('green.m4a')
            if object_name == 'Upstair':
                left_count,right_count = 0,0
                playvoice('upstair.m4a')
            elif object_name == 'Downstair':
                left_count,right_count = 0,0
                playvoice('downstair.m4a')
            elif left_piexl > right_piexl and object_name == 'chair':
                left_count += 1
                if left_count >=1 and object_name=='chair':
                    left_count,right_count = 0,0
                    playvoice('lc.m4a')
            elif left_piexl > right_piexl and object_name == 'obstacle':
                left_count += 1
                if left_count >=2 and object_name=='obstacle':
                    left_count,right_count = 0,0
                    playvoice('lo.m4a')
            elif left_piexl > right_piexl and object_name == 'Transformer box':
                left_count += 1
                if left_count >=2 and object_name=='Transformer box':
                    left_count,right_count = 0,0
                    playvoice('lt.m4a')
            elif left_piexl < right_piexl and object_name == 'chair':
                right_count += 1
                if right_count >=1 and object_name=='chair':
                    left_count,right_count = 0,0
                    playvoice('rc.m4a')
            elif left_piexl < right_piexl and object_name == 'obstacle':
                right_count += 1
                if right_count >=2 and object_name=='obstacle':
                    left_count,right_count = 0,0
                    playvoice('ro.m4a')
            elif left_piexl < right_piexl and object_name == 'Transformer box':
                right_count += 1
                if right_count >=2 and object_name=='Transformer box':
                    left_count,right_count = 0,0
                    playvoice('rt.m4a')
            else:
                pass
            
            #print("left:{} right:{}".format(left_count,right_count))
            #print("left:{} right:{}".format(left_piexl,right_piexl))
            #cv2.imshow('ob', mask)
    # FPS於視窗右上方顯示
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    # 偵測結果與FPS顯示視窗
    cv2.imshow('detector', frame)

    # 計算FPS
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # 按下q離開偵測
    if cv2.waitKey(1) == ord('q'):
        break

# 結束偵測視窗
cv2.destroyAllWindows()
videostream.stop()
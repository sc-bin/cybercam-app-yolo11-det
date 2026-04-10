'''
实验名称：YOLO11检测
实验平台：核桃派2B
说明：摄像头采集检测
'''

from walnutpi_kpu import YOLO11
import cv2,time
import k230_display
k230_display.init()

#【可选代码】允许Thonny远程运行
import os
os.environ["DISPLAY"] = ":0.0"

#加载模型
path_model = "yolo11n.kmodel"
yolo = YOLO11.YOLO11_DET(path_model,224)

# 打开摄像头
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

display_width = k230_display.get_width()
display_height = k230_display.get_height()
# 设置为1080p
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # 设置宽度
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # 设置长度

class_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
boxes = []

#计算帧率
count=0
pt=0
fps = 0

ret, img = cap.read()
while True:
    
    #计算帧率
    count+=1    
    if time.time()-pt >=1 : #超过1秒
        fps=1/((time.time()-pt)/count)#计算帧率
        count=0
        pt=time.time()
    
    # 摄像头读取一帧图像    
    ret, img = cap.read()
    # bgr转rgb
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # #非阻塞式推理图片    
    if not yolo.is_running:
        # 执行目标检测，设置置信度阈值为 0.5，IoU 阈值为 0.45
        yolo.run_async(img, 0.5, 0.45)
    # yolo.run(img, 0.5, 0.45)
        
    boxes = yolo.get_result()
    
        
    # 输出检测结果
    if boxes is not None:
        for box in boxes:
            print(
                "{:f} ({:4d},{:4d}) w{:4d} h{:4d} {:s}".format(
                    box.reliability,
                    box.x,
                    box.y,
                    box.w,
                    box.h,
                    class_names[box.label],
                )
            )
    
    # 到图上画框
    for box in boxes:
        label = str(class_names[box.label]) + " " + str('%.2f'%box.reliability)
        left_x = int(box.x - box.w / 2)
        left_y = int(box.y - box.h / 2)
        right_x = int(box.x + box.w / 2)
        right_y = int(box.y + box.h / 2)
        (label_width, label_height), bottom = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1,
        )
        (label_width, label_height), bottom = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1,
        )
        cv2.rectangle(
            img,
            (left_x, left_y),
            (right_x, right_y),
            (255, 255, 0),
            2,
        )
        cv2.rectangle(
            img,
            (left_x, left_y - label_height * 2),
            (left_x + label_width, left_y),
            (255, 255, 255),
            -1,
        )
        cv2.putText(
            img,
            label,
            (left_x, left_y - label_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
        
    cv2.putText(img, 'FPS: '+str(fps), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2) #图像绘制帧率
    k230_display.show(img)
    
    # cv2.imshow("result", img)#窗口显示图片
    
    # key = cv2.waitKey(1) # 窗口的图像刷新时间为1毫秒，防止阻塞    
    # if key == 32: # 如果按下空格键，打断退出
    #     break
    
cap .release() # 关闭摄像头
cv2.destroyAllWindows() # 销毁显示摄像头视频的窗口

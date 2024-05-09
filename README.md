# BCCD-Detect
혈소판,적혈구,백혈구를 판별
## Location of dataSet 
https://public.roboflow.com/object-detection/bccd
## 작업환경 
장치 이름	DESKTOP-O998J3H
프로세서	Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz   3.19 GHz
설치된 RAM	16.0GB(15.9GB 사용 가능)
장치 ID	79976040-AE86-48D2-BE1C-72385C61EAC7
제품 ID	00330-80000-00000-AA923
시스템 종류	64비트 운영 체제, x64 기반 프로세서
펜 및 터치	펜 지원
에디션	Windows 10 Pro
버전	22H2
설치 날짜	‎2023-‎04-‎12
OS 빌드	19045.4291
경험	Windows Feature Experience Pack 1000.19056.1000.0
## 모델
YoloV5

## 1. 모델 가져오고 환경 구축 
  1. 폴더를 만듭니다. :D:\hdh2024\BCCD
  2. VSCODE를 실행
  3. 터미널을 열고
  4. conda create -n yolov5 python=3.9
  5. VSCODE와 연결 하고
  6. 새 터미널을 엽니다. : (yolov5) D:\hdh2024\BCCD
  7. git clone으로 yolov5를 내려 받습니다.
  8. 'yolov5' Folder로 이동합니다. 
  9. pip install -r requirements.txt

## 2. 데이터세트 준비 과정   
  10. myGlob.py를 만듭니다. 내용은 colab에 있던 2개의 셀을 복사해오고 폴더 위치만 맞춥니다. 
```
# 여기서 주의 할것은 데이터셋의 위치를 잘 맞추어 주세요 
# 실행하는 폴더에 따라서 상대 경로가 달라 지므로 
# 절대 경로를 쓰는것도 괜찮아요 
from glob import glob
train_img_list = glob('D:/hdh2024/BCCD/yolov5/dataSet/train/images/*.jpg')
test_img_list = glob('D:/hdh2024/BCCD/yolov5/dataSet/test/images/*.jpg')
valid_img_list = glob('D:/hdh2024/BCCD/yolov5/dataSet/valid/images/*.jpg')
print(len(train_img_list), len(test_img_list), len(valid_img_list))
import yaml

if len(train_img_list) > 0 :    
    with open('D:/hdh2024/BCCD/yolov5/dataSet/train.txt','w') as f:
        f.write('\n'.join(train_img_list) + '\n')
    with open('D:/hdh2024/BCCD/yolov5/dataSet/test.txt','w') as f:
        f.write('\n'.join(test_img_list) + '\n')
    with open('D:/hdh2024/BCCD/yolov5/dataSet/val.txt','w') as f:
        f.write('\n'.join(valid_img_list) + '\n')
```
  11. data.yaml을 수정합니다. : 폴더경로만 수정
```
train: D:/hdh2024/BCCD/yolov5/dataSet/train/images
val: D:/hdh2024/BCCD/yolov5/dataSet/valid/images

nc: 3
names: ['Platelets', 'RBC', 'WBC']
```
  12. custom_yolov5s.yaml을 ./models/yolov5s.yaml을 복사해서 사용하고 nc : 80을 nc : 3 으로 수정 (데이터세트의nc를 기준)
```
# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license

# Parameters
nc: 3 # number of classes
depth_multiple: 0.33 # model depth multiple
width_multiple: 0.50 # layer channel multiple
anchors:
  - [10, 13, 16, 30, 33, 23] # P3/8
  - [30, 61, 62, 45, 59, 119] # P4/16
  - [116, 90, 156, 198, 373, 326] # P5/32

# YOLOv5 v6.0 backbone
backbone:
  # [from, number, module, args]
  [
    [-1, 1, Conv, [64, 6, 2, 2]], # 0-P1/2
    [-1, 1, Conv, [128, 3, 2]], # 1-P2/4
    [-1, 3, C3, [128]],
    [-1, 1, Conv, [256, 3, 2]], # 3-P3/8
    [-1, 6, C3, [256]],
    [-1, 1, Conv, [512, 3, 2]], # 5-P4/16
    [-1, 9, C3, [512]],
    [-1, 1, Conv, [1024, 3, 2]], # 7-P5/32
    [-1, 3, C3, [1024]],
    [-1, 1, SPPF, [1024, 5]], # 9
  ]

# YOLOv5 v6.0 head
head: [
    [-1, 1, Conv, [512, 1, 1]],
    [-1, 1, nn.Upsample, [None, 2, "nearest"]],
    [[-1, 6], 1, Concat, [1]], # cat backbone P4
    [-1, 3, C3, [512, False]], # 13

    [-1, 1, Conv, [256, 1, 1]],
    [-1, 1, nn.Upsample, [None, 2, "nearest"]],
    [[-1, 4], 1, Concat, [1]], # cat backbone P3
    [-1, 3, C3, [256, False]], # 17 (P3/8-small)

    [-1, 1, Conv, [256, 3, 2]],
    [[-1, 14], 1, Concat, [1]], # cat head P4
    [-1, 3, C3, [512, False]], # 20 (P4/16-medium)

    [-1, 1, Conv, [512, 3, 2]],
    [[-1, 10], 1, Concat, [1]], # cat head P5
    [-1, 3, C3, [1024, False]], # 23 (P5/32-large)

    [[17, 20, 23], 1, Detect, [nc, anchors]], # Detect(P3, P4, P5)
  ]
```
## 3. 학습 


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
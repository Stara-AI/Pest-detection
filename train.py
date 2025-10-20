import os
from ultralytics import YOLO

if __name__ == '__main__':
    # 原yolov8s
    yaml_yolov8s = 'ultralytics/cfg/models/v8/yolov8s.yaml'
    # SE 注意力机制
    yaml_yolov8_SE = 'ultralytics/cfg/models/v8/det_self/yolov8s-attention-SE.yaml'

    # 替换一下变量名即可
    model_yaml = yaml_yolov8s
    # 模型加载
    model = YOLO(model_yaml)
    # 数据集路径的yaml文件
    data_path = r'config\traindata.yaml'
    
    # 文档中对参数有详细的说明
    model.train(data=data_path,             # 数据集
                imgsz=640,                  # 训练图片大小
                epochs=1,                 # 训练的轮次
                batch=2,                    # 训练batch
                workers=0,                  # 加载数据线程数
                device='0',                 # 使用显卡
                optimizer='SGD',            # 优化器
                project='runs'               # 模型保存路径
                )

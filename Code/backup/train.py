import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
 
if __name__ == '__main__':
    model = YOLO(model='Model_file/yaml/yolo12n.yaml')
    #model.load('yolov13n.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model.train(data='D:\YOLO\Dataset\dataset_RGBD_20251014_212809\data_config.yaml',
                imgsz=640,
                epochs=2,
                batch=64,
                amp=True,
                workers=0,
                device='cuda',
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='runs/train',
                name='test',
                single_cls=False,
                cache=False,
                hsv_h=0,
                hsv_s=0,
                hsv_v=0,
                bgr=0,
                auto_augment=None,
                degrees=0,
                translate=0,
                scale=0,
                shear=0,
                perspective=0,
                flipud=0,
                fliplr=0,
                mosaic=0,
                mixup=0,
                copy_paste=0,
                erasing=0,
                crop_fraction=0
                )
import argparse
import os

import cv2
import torch
from mtcnn import MTCNN
from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# COCO 80 类里没有 face，YOLO26 只负责定位人体，人脸由 MTCNN 在人体区域内检出
PERSON_CLASS_ID = 0
# MTCNN 在过小的切片上会抛异常
MIN_PERSON_SIZE = 32


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO26 定位人体 + MTCNN 检测人脸')
    parser.add_argument('-i', '--input', default=os.path.join(PROJECT_ROOT, 'data', 'a2.png'),
                        help='输入图片路径')
    parser.add_argument('-o', '--output', default=os.path.join(PROJECT_ROOT, 'processed_images'),
                        help='输出目录')
    parser.add_argument('--weights', default=os.path.join(PROJECT_ROOT, 'weights', 'yolo26s.pt'),
                        help='YOLO26 权重路径')
    parser.add_argument('--conf', type=float, default=0.25, help='人体框置信度阈值')
    parser.add_argument('--face-conf', type=float, default=0.90, help='人脸框置信度阈值')
    parser.add_argument('--device', default=None, help='cuda:0 或 cpu，缺省自动选择')
    return parser.parse_args()


def pick_device(device_arg):
    if device_arg:
        return device_arg
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


def preprocess_image(image_path):
    if not os.path.isfile(image_path):
        print(f'输入图片不存在: {image_path}')
        return None
    image = cv2.imread(image_path)
    if image is None:
        print(f'无法读取图像，请检查图像路径: {image_path}')
    return image


def detect_persons(model, image, conf, device):
    """返回原图分辨率下的 (x1, y1, x2, y2) 像素坐标，不做缩放。"""
    boxes = []
    for result in model.predict(source=image, classes=[PERSON_CLASS_ID], conf=conf,
                                device=device, verbose=False):
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return boxes


def detect_faces(detector, image, person_boxes, face_conf):
    """在每个人体框内跑 MTCNN，并把人脸框平移回全图坐标。"""
    detected_faces = []
    for (px1, py1, px2, py2) in person_boxes:
        pw, ph = px2 - px1, py2 - py1
        if pw < MIN_PERSON_SIZE or ph < MIN_PERSON_SIZE:
            continue
        crop = image[py1:py2, px1:px2]
        for face in detector.detect_faces(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)):
            if face['confidence'] < face_conf:
                continue
            fx, fy, fw, fh = face['box']
            fx = max(0, fx)
            fy = max(0, fy)
            fw = min(fw, pw - fx)
            fh = min(fh, ph - fy)
            if fw <= 0 or fh <= 0:
                continue
            detected_faces.append((px1 + fx, py1 + fy, fw, fh))
    return detected_faces


def draw_faces(image, detected_faces):
    for (x, y, w, h) in detected_faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)


def draw_persons(image, person_boxes):
    for (x1, y1, x2, y2) in person_boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)


def main():
    args = parse_args()
    device = pick_device(args.device)

    model = YOLO(args.weights)
    detector = MTCNN()

    original_image = preprocess_image(args.input)
    if original_image is None:
        return

    person_boxes = detect_persons(model, original_image, args.conf, device)
    detected_faces = detect_faces(detector, original_image, person_boxes, args.face_conf)

    print(f'设备: {device}')
    print(f'检测到的人体数量: {len(person_boxes)}')
    print(f'检测到的面部数量: {len(detected_faces)}')

    draw_persons(original_image, person_boxes)
    draw_faces(original_image, detected_faces)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    output_image_path = os.path.join(args.output, 'processed_image.jpg')
    try:
        cv2.imwrite(output_image_path, original_image)
    except Exception as e:
        print(f'保存图像失败: {e}')
        return

    cv2.imshow('Face Detection', original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

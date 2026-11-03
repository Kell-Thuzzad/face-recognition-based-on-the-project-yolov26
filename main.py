import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from face_pipeline import (IMAGE_SUFFIXES, UNKNOWN, FaceDetector, Gallery,
                           Recognizer, assign_persons)

PROJECT_ROOT = Path(__file__).resolve().parent
PERSON_CLASS_ID = 0
FONT_CANDIDATES = (r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\simhei.ttf',
                   '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
STREAM_PREFIXES = ('rtsp://', 'rtsps://', 'rtmp://', 'http://', 'https://')


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO26 人体定位 + YuNet 人脸检测 + ArcFace 1:N 识别')
    parser.add_argument('-i', '--input', required=True,
                        help='图片 / 图片目录 / 视频文件 / 相机索引 / RTSP-HTTP 流地址')
    parser.add_argument('-o', '--output', type=Path, default=PROJECT_ROOT / 'processed_images',
                        help='输出目录')
    parser.add_argument('--weights', type=Path, default=PROJECT_ROOT / 'weights' / 'yolo26s.pt',
                        help='YOLO26 权重')
    parser.add_argument('--face-model', type=Path,
                        default=PROJECT_ROOT / 'weights' / 'face_detection_yunet_2023mar.onnx')
    parser.add_argument('--rec-model', type=Path,
                        default=PROJECT_ROOT / 'weights' / 'w600k_mbf.onnx',
                        help='ArcFace 人脸特征模型 onnx')
    parser.add_argument('--gallery-dir', type=Path, default=None,
                        help='底库目录：子目录名或图片文件名即人员编号')
    parser.add_argument('--gallery-cache', type=Path, default=None,
                        help='底库特征缓存 npz，默认 <output>/gallery.npz')
    parser.add_argument('--build-gallery', action='store_true', help='重新注册底库')
    parser.add_argument('--conf', type=float, default=0.25, help='人体框置信度阈值')
    parser.add_argument('--face-conf', type=float, default=0.6, help='人脸框置信度阈值')
    parser.add_argument('--match-threshold', type=float, default=0.40,
                        help='1:N 比对的余弦相似度阈值，需用本厂样本标定')
    parser.add_argument('--no-person', action='store_true', help='跳过 YOLO26 人体定位（门禁近景）')
    parser.add_argument('--device', default=None, help='cuda:0 或 cpu，缺省自动选择')
    parser.add_argument('--frame-stride', type=int, default=1, help='视频流每 N 帧处理一次')
    parser.add_argument('--max-frames', type=int, default=0, help='视频流最多处理帧数，0 为不限')
    parser.add_argument('--json-out', type=Path, default=None, help='结果 jsonl 路径，默认 <output>/results.jsonl')
    parser.add_argument('--save-frames', action='store_true', help='视频流也保存标注帧')
    parser.add_argument('--no-display', action='store_true', help='不弹窗显示')
    return parser.parse_args()


def pick_device(device_arg):
    if device_arg:
        return device_arg
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


class Viewer:
    """边缘盒子常用 opencv-python-headless，没有 highgui 时自动降级为不显示。"""

    def __init__(self, enabled=True):
        self.enabled = enabled

    def show(self, image, wait_ms=1):
        if not self.enabled:
            return None
        try:
            cv2.imshow('Face Recognition', image)
            return cv2.waitKey(wait_ms) & 0xFF
        except cv2.error as error:
            print(f'当前 OpenCV 无 GUI，已停止显示（可用 --save-frames 保留标注图）: {error}')
            self.enabled = False
            return None

    def close(self):
        if self.enabled:
            cv2.destroyAllWindows()


class TextRenderer:
    """cv2.putText 画不出中文姓名，标签统一走一次 PIL。"""

    def __init__(self, size=18):
        self.font = None
        for path in FONT_CANDIDATES:
            if Path(path).is_file():
                self.font = ImageFont.truetype(path, size)
                break

    def draw(self, image, labels):
        if self.font is None:
            for text, (x, y), color in labels:
                cv2.putText(image, text if text.isascii() else text.encode('ascii', 'ignore').decode(),
                            (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            return image
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        for text, (x, y), color in labels:
            draw.text((x, y), text, font=self.font, fill=tuple(color[::-1]))
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def detect_persons(model, image, conf, device):
    persons = []
    for result in model.predict(source=image, classes=[PERSON_CLASS_ID], conf=conf,
                                device=device, verbose=False):
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            persons.append(((int(x1), int(y1), int(x2), int(y2)), float(box.conf[0])))
    return persons


def load_or_build_gallery(args, detector, recognizer):
    if args.gallery_dir is None:
        return None
    cache = args.gallery_cache or (args.output / 'gallery.npz')
    if cache.is_file() and not args.build_gallery:
        gallery = Gallery.load(cache)
        if gallery.model and gallery.model != args.rec_model.name:
            raise RuntimeError(
                f'底库缓存由 {gallery.model} 生成，与当前 --rec-model {args.rec_model.name} 不一致，'
                f'向量空间不互通，请加 --build-gallery 重建')
        print(f'底库缓存: {cache}（{len(gallery)} 人，特征模型 {gallery.model or "未记录"}）')
        return gallery
    if not args.gallery_dir.is_dir():
        raise FileNotFoundError(f'底库目录不存在: {args.gallery_dir}')
    print(f'注册底库: {args.gallery_dir}')
    gallery = Gallery.build(args.gallery_dir, detector, recognizer)
    gallery.save(cache, model=args.rec_model.name)
    print(f'底库写入缓存: {cache}（{len(gallery)} 人，特征模型 {args.rec_model.name}）')
    return gallery


def make_processor(args, detector, recognizer, gallery, yolo, device):
    def process(image):
        height, width = image.shape[:2]
        timings = {}

        started = time.perf_counter()
        persons = detect_persons(yolo, image, args.conf, device) if yolo else []
        timings['person_ms'] = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        faces = detector.detect(image)
        timings['face_detect_ms'] = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        for face in faces:
            face.embedding = recognizer.embed(image, face.kps)
            if gallery is not None:
                face.name, face.similarity = gallery.match(face.embedding, args.match_threshold)
        assign_persons(faces, [box for box, _ in persons])
        timings['recognize_ms'] = (time.perf_counter() - started) * 1000

        record = {
            'ts': datetime.now().isoformat(timespec='milliseconds'),
            'image_size': [width, height],
            'timings_ms': {k: round(v, 1) for k, v in timings.items()},
            'persons': [{'index': i, 'box': list(box), 'conf': round(conf, 3)}
                        for i, (box, conf) in enumerate(persons)],
            'faces': [face.to_dict() for face in faces],
        }
        return record, (faces, persons)
    return process


def annotate(image, faces, persons, renderer):
    annotated = image.copy()
    labels = []
    for index, (box, _) in enumerate(persons):
        cv2.rectangle(annotated, box[:2], box[2:], (150, 150, 150), 1)
        labels.append((f'P{index}', (box[0] + 2, max(0, box[1] - 20)), (150, 150, 150)))
    for face in faces:
        known = face.name and face.name != UNKNOWN
        color = (0, 200, 0) if known else (0, 0, 255)
        cv2.rectangle(annotated, face.box[:2], face.box[2:], color, 2)
        for (kx, ky) in face.kps:
            cv2.circle(annotated, (int(kx), int(ky)), 1, color, -1)
        label = f'{face.name} {face.similarity:.2f}' if known else f'未知 {face.score:.2f}'
        labels.append((label, (face.box[0], max(0, face.box[1] - 24)), color))
    return renderer.draw(annotated, labels)


def iter_still_sources(path):
    if not path.exists():
        raise FileNotFoundError(f'输入路径不存在: {path}')
    suffixes = {p.suffix.lower() for p in path.iterdir()} if path.is_dir() else {path.suffix.lower()}
    if not suffixes & IMAGE_SUFFIXES:
        raise ValueError(f'{path} 不是受支持的图片: {", ".join(sorted(IMAGE_SUFFIXES))}')
    files = sorted(path.glob('*')) if path.is_dir() else [path]
    for file in files:
        if file.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        image = cv2.imread(str(file))
        if image is None:
            print(f'跳过无法读取的图像: {file}')
            continue
        yield file, image


def iter_video_sources(source, stride, max_frames):
    if source.isdigit():
        capture = cv2.VideoCapture(int(source))
    else:
        capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f'无法打开视频源: {source}')
    processed = 0
    frame_index = -1
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            if frame_index % stride:
                continue
            yield frame_index, frame
            processed += 1
            if max_frames and processed >= max_frames:
                break
    finally:
        capture.release()


def is_video_source(input_arg):
    path = Path(input_arg)
    if path.is_dir():
        return False
    if input_arg.startswith(STREAM_PREFIXES) or input_arg.isdigit():
        return True
    return path.is_file() and path.suffix.lower() not in IMAGE_SUFFIXES


def main():
    args = parse_args()
    device = pick_device(args.device)

    detector = FaceDetector(args.face_model, args.face_conf)
    recognizer = Recognizer(args.rec_model)
    gallery = load_or_build_gallery(args, detector, recognizer)
    yolo = None if args.no_person else YOLO(args.weights)

    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.json_out or (args.output / 'results.jsonl')
    json_path.parent.mkdir(parents=True, exist_ok=True)

    process = make_processor(args, detector, recognizer, gallery, yolo, device)
    renderer = TextRenderer()
    viewer = Viewer(enabled=not args.no_display)
    video = is_video_source(args.input)

    print(f'设备: {device} | 人体定位: {"关闭" if yolo is None else args.weights.name} '
          f'| 底库: {"未启用" if gallery is None else f"{len(gallery)} 人"}')

    records = 0
    total_ms = 0.0
    started = time.perf_counter()
    with json_path.open('w', encoding='utf-8') as sink:
        sources = (iter_video_sources(args.input, args.frame_stride, args.max_frames) if video
                   else iter_still_sources(Path(args.input)))
        for name, image in sources:
            record, (faces, persons) = process(image)
            record['source'] = str(name)
            sink.write(json.dumps(record, ensure_ascii=False) + '\n')
            sink.flush()
            records += 1
            total_ms += sum(record['timings_ms'].values())

            annotated = annotate(image, faces, persons, renderer)
            known = sum(1 for f in faces if f.name and f.name != UNKNOWN)
            if video:
                print(f'帧 {name}: 人体 {len(persons)} 人脸 {len(faces)} 已识别 {known} '
                      f'耗时 {sum(record["timings_ms"].values()):.1f}ms')
                if args.save_frames:
                    cv2.imwrite(str(args.output / f'frame_{int(name):06d}.jpg'), annotated)
            else:
                print(f'{name.name}: 人体 {len(persons)} 人脸 {len(faces)} 已识别 {known}')
                cv2.imwrite(str(args.output / f'{name.stem}_annotated.jpg'), annotated)

            key = viewer.show(annotated, 1 if video else 0)
            if key in (ord('q'), 27):
                print('用户中止')
                break
    elapsed = time.perf_counter() - started

    if records:
        print(f'共 {records} 帧，平均 {total_ms / records:.1f}ms/帧，'
              f'吞吐 {records / elapsed:.2f} 帧/秒（含保存与显示）')
    print(f'结果已写入 {json_path}')
    viewer.close()


if __name__ == '__main__':
    main()

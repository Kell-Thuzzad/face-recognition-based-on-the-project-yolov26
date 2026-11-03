"""YuNet 人脸检测 + ArcFace 对齐特征 + 底库 1:N 比对。

坐标一律为输入图像的全分辨率像素值，不做缩放。
"""

from pathlib import Path

import cv2
import numpy as np
from insightface.model_zoo import get_model
from insightface.utils import face_align

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
UNKNOWN = '未知'


class Face:
    __slots__ = ('box', 'score', 'kps', 'embedding', 'name', 'similarity', 'person_index')

    def __init__(self, box, score, kps):
        self.box = box
        self.score = score
        self.kps = kps
        self.embedding = None
        self.name = ''
        self.similarity = 0.0
        self.person_index = None

    @property
    def area(self):
        return (self.box[2] - self.box[0]) * (self.box[3] - self.box[1])

    def to_dict(self):
        return {
            'box': list(self.box),
            'score': round(self.score, 4),
            'name': self.name or UNKNOWN,
            'similarity': round(self.similarity, 4),
            'person_index': self.person_index,
        }


class FaceDetector:
    def __init__(self, model_path, score_threshold=0.6, nms_threshold=0.3, top_k=5000):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f'人脸检测模型不存在: {model_path}')
        self._net = cv2.FaceDetectorYN.create(
            str(model_path), '', (320, 320), score_threshold, nms_threshold, top_k)

    def detect(self, image):
        height, width = image.shape[:2]
        self._net.setInputSize((width, height))
        _, rows = self._net.detect(image)
        faces = []
        for row in [] if rows is None else rows:
            x, y, w, h = row[:4]
            x1 = max(0, int(round(x)))
            y1 = max(0, int(round(y)))
            x2 = min(width, int(round(x + w)))
            y2 = min(height, int(round(y + h)))
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue
            kps = row[4:14].reshape(5, 2).astype(np.float32)
            faces.append(Face((x1, y1, x2, y2), float(row[14]), kps))
        faces.sort(key=lambda face: face.area, reverse=True)
        return faces


class Recognizer:
    def __init__(self, model_path):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f'人脸特征模型不存在: {model_path}')
        model = get_model(str(model_path))
        if getattr(model, 'taskname', None) != 'recognition':
            raise ValueError(f'{model_path.name} 不是人脸特征模型（taskname={getattr(model, "taskname", None)}）')
        self._model = model
        self.input_size = int(model.input_size[0])

    def embed(self, image, kps):
        aligned = face_align.norm_crop(image, landmark=kps, image_size=self.input_size)
        feature = np.asarray(self._model.get_feat(aligned)).ravel()
        return (feature / np.linalg.norm(feature)).astype(np.float32)


class Gallery:
    """底库：每行是一个已归一化的人脸特征，对应一个工号/姓名。"""

    def __init__(self, names, embeddings, model=''):
        self.names = list(names)
        self.embeddings = np.asarray(embeddings, dtype=np.float32).reshape(len(self.names), -1)
        self.model = model

    def __len__(self):
        return len(self.names)

    @classmethod
    def load(cls, npz_path):
        data = np.load(npz_path, allow_pickle=False)
        names = [str(n) for n in data['names']]
        model = str(data['model']) if 'model' in data else ''
        return cls(names, data['embeddings'], model)

    def save(self, npz_path, model=''):
        Path(npz_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, names=np.array(self.names, dtype=str),
                            embeddings=self.embeddings, model=np.array(model, dtype=str))

    @classmethod
    def build(cls, root, detector, recognizer, min_face_pixels=40):
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f'底库目录不存在: {root}')
        entries = sorted([p for p in root.iterdir() if not p.name.startswith('.')],
                         key=lambda p: p.name)

        names, embeddings, skipped = [], [], []
        for entry in entries:
            if entry.is_dir():
                images = sorted(p for p in entry.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
                name = entry.name
            elif entry.suffix.lower() in IMAGE_SUFFIXES:
                images = [entry]
                name = entry.stem
            else:
                continue

            features = []
            for path in images:
                image = cv2.imread(str(path))
                if image is None:
                    continue
                faces = detector.detect(image)
                if not faces or faces[0].area < min_face_pixels * min_face_pixels:
                    continue
                features.append(recognizer.embed(image, faces[0].kps))
            if not features:
                skipped.append(name)
                continue
            names.append(name)
            embeddings.append(np.mean(features, axis=0))
            print(f'  底库 {name}: {len(features)}/{len(images)} 张样本可用')

        if not names:
            raise RuntimeError(f'底库 {root} 中没有注册到任何有效人脸')
        gallery = cls(names, np.stack(embeddings))
        gallery.embeddings /= np.linalg.norm(gallery.embeddings, axis=1, keepdims=True)
        if skipped:
            print(f'  跳过（未检出足够大的人脸）: {", ".join(skipped)}')
        return gallery

    def match(self, embedding, threshold):
        if not len(self):
            return UNKNOWN, 0.0
        similarities = self.embeddings @ embedding
        index = int(np.argmax(similarities))
        best = float(similarities[index])
        return (self.names[index] if best >= threshold else UNKNOWN), best


def assign_persons(faces, person_boxes):
    """把人脸按中心点归属到人体框，用于工位归属与在岗判定。"""
    for face in faces:
        cx = (face.box[0] + face.box[2]) / 2
        cy = (face.box[1] + face.box[3]) / 2
        for index, (x1, y1, x2, y2) in enumerate(person_boxes):
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                face.person_index = index
                break

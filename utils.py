import cv2
import torch


def preprocess_image(image_path):
    """
    预处理图像（例如调整大小等），适用于模型输入
    :param image_path: 输入图像路径
    :return: 预处理后的图像
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像：{image_path}")
        return None

    # 转换为 RGB 并调整大小为模型输入要求的尺寸（假设为640x640）
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 640))

    # 将图像归一化并转换为 float32 类型
    image = image.astype('float32') / 255.0

    # 增加批量维度
    image = image.transpose(2, 0, 1)  # HWC -> CHW
    image = torch.from_numpy(image).unsqueeze(0)

    return image


def detect_faces(model, image):
    """
    使用YOLOv8模型检测图像中的面部
    :param model: YOLOv8模型
    :param image: 预处理后的输入图像
    :return: 检测到的面部信息，包含边界框坐标和置信度
    """
    model.eval()
    with torch.no_grad():
        outputs = model(image)

    print("Model output shape:", outputs.shape)  # 打印模型输出的形状
    print("Model output data:", outputs)  # 打印模型的原始输出数据

    num_anchors = 3
    num_classes = 80
    grid_size = outputs.shape[2]  # 特征图的网格尺寸

    num_predictions = outputs.shape[1]
    num_classes_plus_4 = num_predictions // (num_anchors * grid_size * grid_size)

    detected_faces = []

    # 确认图像的真实尺寸
    image_width, image_height = 178, 218

    for h in range(grid_size):
        for w in range(grid_size):
            for anchor in range(num_anchors):
                start_index = anchor * num_classes_plus_4
                end_index = start_index + num_classes_plus_4

                pred = outputs[0, start_index:end_index, h, w]

                if pred.numel() == 0:
                    continue

                print(f"Predictions at (h={h}, w={w}, anchor={anchor}):", pred)

                confidence = pred[4].item()
                if confidence > 0.1:  # 更低的置信度阈值
                    x_center, y_center, width, height = pred[:4].tolist()
                    class_probs = pred[5:]
                    class_id = class_probs.index(max(class_probs))
                    class_prob = class_probs[class_id]

                    # 根据实际图像尺寸计算边界框坐标
                    x1 = int((x_center - width / 2) * image_width)
                    y1 = int((y_center - height / 2) * image_height)
                    x2 = int((x_center + width / 2) * image_width)
                    y2 = int((y_center + height / 2) * image_height)

                    detected_faces.append({
                        'box': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_prob': class_prob
                    })

    return detected_faces


def draw_faces(image, detected_faces):
    """
    在图像上绘制检测到的面部边界框
    :param image: 输入图像 (numpy 数组)
    :param detected_faces: 检测到的面部信息，包含边界框坐标
    :return: 带有边框的图像
    """
    # 逐个遍历检测到的面部，并绘制边界框
    for face in detected_faces:
        x1, y1, x2, y2 = face['box']  # 获取边界框坐标
        confidence = face['confidence']  # 置信度

        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 绘制置信度文本
        label = f'Confidence: {confidence:.2f}'
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

import cv2
import os

# 加载Haar级联分类器模型
cascPath = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascPath)


def preprocess_image(image_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 如果图像为空则返回
    if image is None:
        print("无法读取图像，请检查图像路径。")
        return None

    return image


def detect_faces(image):
    # 将图像从BGR转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用Haar级联分类器检测面部
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    return faces


def draw_faces(image, faces):
    # 在图像上绘制检测到的面部
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)


def main():
    # 输入图像路径
    input_image_path = "E:\\College Student Innovation\\data set\\a2.PNG"

    # 输出文件夹路径
    output_folder = "E:\\College Student Innovation\\processed_images"

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 对图像进行预处理
    original_image = preprocess_image(input_image_path)

    # 如果预处理后的图像不为空，则进行面部检测
    if original_image is not None:
        # 检测图像中的面部
        faces = detect_faces(original_image)

        # 打印检测到的面部数量
        print(f"检测到的面部数量: {len(faces)}")

        # 在原始图像上绘制检测结果
        draw_faces(original_image, faces)

        # 输出图像文件路径
        output_image_path = os.path.join(output_folder, "processed_image.jpg")

        # 保存带有面部检测结果的图像
        try:
            cv2.imwrite(output_image_path, original_image)
        except Exception as e:
            print(f"保存图像失败: {e}")
            return

        # 显示带有面部检测结果的图像
        cv2.imshow("Face Detection", original_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

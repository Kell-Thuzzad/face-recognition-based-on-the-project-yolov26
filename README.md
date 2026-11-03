# face-recognition-based-on-the-project-yolov8s
人脸识别
本项目用 YOLO26 权重定位人体，再由 MTCNN 在人体框内检出人脸；Haar.py 是独立的 Haar 级联对比分支。
已有程序仅支持对单个图片进行分析，若有需要可自行更改。
使用方法：
安装环境；运行 `python main.py -i 输入图片 -o 输出目录 [--weights weights/yolo26s.pt --conf 0.25 --face-conf 0.9 --device cuda:0]`
有好的修改建议与想法可以交流。

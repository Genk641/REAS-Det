from ultralytics import YOLO
if __name__ == '__main__':
    # 加载模型  cd /root/goshyolo
    #model = YOLO("./ultralytics/cfg/models/v8/REAS-Det-c4-base-3.yaml")  # 从头开始构建新模型
    model = YOLO("network/yolov8n.yaml")
    #model = YOLO("REAS-Det.yaml")  # 加载预训练模型（推荐用于训练）

    # Use the modelp #0.60//0.80
    results = model.train(data="./self_data.yaml", epochs=300, batch=8, workers=8, name='yolov8n')  # 训练模型
    # results = model.val()  # 在验证集上评估模型性能 
    # results = model("https://ultralytics.com/images/bus.jpg")  # 预测图像
    #success = model.export(format="onnx")  # 将模型导出为 ONNX 格式
    #CUDA_VISIBLE_DEVICES=3 python main.py


from ultralytics import YOLO
import torch
import os

def main():
    data_yaml = r"Dataset\data.yaml"   # change this to your real path
    project_dir = r"Dataset\runs"

    os.makedirs(project_dir, exist_ok=True)

    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Training on CPU")

    # Small model is best for CPU
    model = YOLO("yolo11n.pt")   # if unavailable, use "yolov8n.pt"

    model.train(
        data=data_yaml,
        epochs=50,
        imgsz=416,
        batch=4,
        device="cpu",
        workers=0,
        project=project_dir,
        name="esra_cpu_train",
        pretrained=True,
        optimizer="auto",
        patience=10,
        save=True,
        val=True,
        verbose=True
    )

if __name__ == "__main__":
    main()
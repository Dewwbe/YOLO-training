from ultralytics import YOLO

def main():
    model_path = r"D:\Dataset\ESRA\runs\esra_cpu_train\weights\best.pt"
    data_yaml = r"D:\Dataset\ESRA\data.yaml"

    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, split="test", device="cpu")

    print(metrics)

if __name__ == "__main__":
    main()
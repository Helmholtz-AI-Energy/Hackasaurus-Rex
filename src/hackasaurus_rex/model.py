from ultralytics import YOLO


def get_model(version: str = "8", size: str = "m"):
    yolo = YOLO(f"models/yolov{version}{size}.pt")

    return yolo.model

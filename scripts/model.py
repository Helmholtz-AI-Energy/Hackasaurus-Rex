from ultralytics import YOLO

def get_model(size: str = "m"):
    model = YOLO('yolo8n.pt')
    return model



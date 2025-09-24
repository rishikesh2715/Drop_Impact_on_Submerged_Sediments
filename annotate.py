from ultralytics import YOLO

model = YOLO("best.pt")

results = model.pred


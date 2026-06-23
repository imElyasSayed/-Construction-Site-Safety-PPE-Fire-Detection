from ultralytics import YOLO

model = YOLO(r"C:\FireSmokeDataset\yolov8s.pt")

model.train(
    data=r"C:\firesmoke\data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="fire_smoke_v2",
    project=r"C:\firesmoke\runs",
    patience=15,
    save=True,
    val=True,
)

print("\nTraining complete!")
print(r"Best weights: C:\firesmoke\runs\fire_smoke_v2\weights\best.pt")
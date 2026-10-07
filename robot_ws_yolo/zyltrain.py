from ultralytics import YOLO


def main():
    model=YOLO(r"yolo11n.pt")
    model.train(
        data=r"robot_ws.yaml",
        epochs=100,
        imgsz= 960,
        batch=-1,
        cache=False,
        workers=1,

    )

if __name__ =="__main__":
    main()
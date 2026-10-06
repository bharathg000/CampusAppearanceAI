from ultralytics import YOLO
import cv2

model = YOLO("idcard_v2.pt")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera could not be opened")
    exit()

print("Camera started. Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read camera")
        break

    results = model(frame, conf=0.70, verbose=False)

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        print(
            "Class:",
            class_id,
            "Confidence:",
            round(confidence, 2)
        )

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Camera Test", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")

camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()

    if not success:
        print("Could not access camera")
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("CampusAppearanceAI - YOLO Test", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
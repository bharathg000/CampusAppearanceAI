
from flask import Flask, render_template, Response, request
from ultralytics import YOLO
import cv2
import threading
import time

from compliance import check_compliance


app = Flask(__name__)


# =========================================
# LOAD YOLO MODEL
# =========================================

model = YOLO("idcard_v2.pt")


# =========================================
# CAMERA
# =========================================

camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# =========================================
# SHARED DATA
# =========================================

latest_frame = None

latest_id_card = False

frame_lock = threading.Lock()


# =========================================
# CAMERA THREAD
# =========================================

def camera_thread():

    global latest_frame

    while True:

        success, frame = camera.read()

        if success:

            with frame_lock:

                latest_frame = frame.copy()

        else:

            print("Camera read failed")

            time.sleep(0.1)


# =========================================
# YOLO DETECTION THREAD
# =========================================

def detection_thread():

    global latest_id_card

    while True:

        # Get newest camera frame

        with frame_lock:

            if latest_frame is None:

                time.sleep(0.05)

                continue

            frame = latest_frame.copy()


        # =====================================
        # YOLO DETECTION
        # =====================================

        results = model(
            frame,
            conf=0.60,
            imgsz=320,
            verbose=False
        )


        # Assume ID card is not detected

        id_card_detected = False


        # =====================================
        # CHECK ID CARD
        # =====================================

        for box in results[0].boxes:

            class_id = int(box.cls[0])

            confidence = float(box.conf[0])


            # Class 0 = ID Card

            if class_id == 0 and confidence >= 0.70:

                id_card_detected = True

                break


        # =====================================
        # UPDATE STATUS
        # =====================================

        latest_id_card = id_card_detected


        # =====================================
        # WAIT
        # =====================================

        time.sleep(0.5)


# =========================================
# START CAMERA THREAD
# =========================================

camera_worker = threading.Thread(
    target=camera_thread,
    daemon=True
)

camera_worker.start()


# =========================================
# START YOLO THREAD
# =========================================

detection_worker = threading.Thread(
    target=detection_thread,
    daemon=True
)

detection_worker.start()


# =========================================
# HOME PAGE
# =========================================

@app.route("/")
def home():

    return render_template("index.html")


# =========================================
# DETECTION PAGE
# =========================================

@app.route("/detect")
def detect():

    return render_template("detect.html")


# =========================================
# VIDEO STREAM
# =========================================

def generate_frames():

    global latest_id_card

    while True:

        # Get latest camera frame

        with frame_lock:

            if latest_frame is None:

                time.sleep(0.05)

                continue

            frame = latest_frame.copy()


        # =====================================
        # SHOW ID CARD STATUS
        # =====================================

        if latest_id_card:

            status_text = "ID CARD: DETECTED"

            text_color = (0, 255, 0)

        else:

            status_text = "ID CARD: NOT DETECTED"

            text_color = (0, 0, 255)


        cv2.putText(
            frame,
            status_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            text_color,
            2
        )


        # =====================================
        # CONVERT FRAME TO JPEG
        # =====================================

        ret, buffer = cv2.imencode(
            ".jpg",
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                75
            ]
        )


        if not ret:

            continue


        frame_bytes = buffer.tobytes()


        # =====================================
        # SEND FRAME TO WEBSITE
        # =====================================

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# =========================================
# VIDEO ROUTE
# =========================================

@app.route("/video")
def video():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# =========================================
# RESULT PAGE
# =========================================

@app.route("/result")
def result():

    # =====================================
    # ID CARD FROM YOLO
    # =====================================

    id_card = latest_id_card


    # =====================================
    # GET CHECKLIST VALUES
    # =====================================

    shirt_tucked = (
        request.args.get("shirt_tucked", "true").lower()
        == "true"
    )


    formal_trousers = (
        request.args.get("formal_trousers", "true").lower()
        == "true"
    )


    formal_shoes = (
        request.args.get("formal_shoes", "true").lower()
        == "true"
    )


    hair_ok = (
        request.args.get("hair_ok", "true").lower()
        == "true"
    )


    beard_ok = (
        request.args.get("beard_ok", "true").lower()
        == "true"
    )


    # =====================================
    # CHECK COMPLIANCE
    # =====================================

    result, rules, issues = check_compliance(

        id_card=id_card,

        shirt_tucked=shirt_tucked,

        formal_trousers=formal_trousers,

        formal_shoes=formal_shoes,

        hair_ok=hair_ok,

        beard_ok=beard_ok

    )


    # =====================================
    # SHOW RESULT PAGE
    # =====================================

    return render_template(

        "result.html",

        result=result,

        rules=rules,

        issues=issues

    )


# =========================================
# START FLASK
# =========================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=False,

        threaded=True

    )
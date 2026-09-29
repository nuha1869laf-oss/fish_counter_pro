from flask import Flask, render_template, request, jsonify
import os
import cv2
import subprocess
from PIL import Image, ImageDraw
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
import supervision as sv
from imageio_ffmpeg import get_ffmpeg_exe
# =========================================================
# LOAD ENVIRONMENT
# =========================================================
load_dotenv()

# =========================================================
# FLASK APP
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# Create folders if they don't exist
os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)
os.makedirs(
    "static",
    exist_ok=True
)
# =========================================================
# ROBOFLOW CLIENT
# =========================================================

client = InferenceHTTPClient(

    api_url="https://serverless.roboflow.com",

    api_key=os.getenv(
        "ROBOFLOW_API_KEY"
    )
)
# =========================================================
# ROBoflow MODEL
# =========================================================

MODEL_ID = (
    "nuha-ymzrs/"
    "fish-detection-fuc8i-5-rfdetr-small-t1"
)


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# DRAW BOUNDING BOXES ON IMAGE
# =========================================================

def draw_fish_boxes(
    input_path,
    predictions,
    output_path
):

    # Open image
    img = Image.open(
        input_path
    ).convert(
        "RGB"
    )


    draw = ImageDraw.Draw(
        img
    )


    image_width = img.width

    image_height = img.height


    # =====================================================
    # DRAW EACH FISH
    # =====================================================

    for index, prediction in enumerate(
        predictions,
        start=1
    ):

        # Roboflow center coordinates
        x = prediction["x"]

        y = prediction["y"]


        # Bounding box size
        width = prediction["width"]

        height = prediction["height"]


        # Convert center coordinates
        # to corner coordinates

        x1 = max(
            0,
            int(
                x - width / 2
            )
        )


        y1 = max(
            0,
            int(
                y - height / 2
            )
        )


        x2 = min(
            image_width - 1,
            int(
                x + width / 2
            )
        )


        y2 = min(
            image_height - 1,
            int(
                y + height / 2
            )
        )


        # =================================================
        # GREEN BOUNDING BOX
        # =================================================

        draw.rectangle(

            [
                x1,
                y1,
                x2,
                y2
            ],

            outline="lime",

            width=4

        )


        # =================================================
        # FISH LABEL
        # =================================================

        label = (
            f"Fish {index}"
        )


        label_y = max(
            0,
            y1 - 18
        )


        draw.text(

            (
                x1 + 3,
                label_y
            ),

            label,

            fill="lime"

        )


    # =====================================================
    # SAVE RESULT IMAGE
    # =====================================================

    img.save(

        output_path,

        "JPEG",

        quality=95

    )


# =========================================================
# IMAGE UPLOAD
# =========================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload():

    image = request.files.get(
        "fish_image"
    )


    # =====================================================
    # CHECK IMAGE
    # =====================================================

    if not image:

        return """
        <h2>No image selected</h2>
        <a href="/">Go Back</a>
        """


    # =====================================================
    # SAVE ORIGINAL IMAGE
    # =====================================================

    image_path = os.path.join(

        app.config["UPLOAD_FOLDER"],

        image.filename

    )


    image.save(
        image_path
    )


    # =====================================================
    # CONVERT IMAGE TO JPG
    # =====================================================

    converted_path = os.path.join(

        app.config["UPLOAD_FOLDER"],

        "converted_fish.jpg"

    )


    try:

        img = Image.open(
            image_path
        ).convert(
            "RGB"
        )


        img.save(

            converted_path,

            "JPEG",

            quality=95

        )


    except Exception as e:

        return f"""
        <h2>Image conversion error</h2>

        <p>{e}</p>

        <a href="/">Go Back</a>
        """


    # =====================================================
    # ROBOFLOW DETECTION
    # =====================================================

    try:

        result = client.infer(

            converted_path,

            model_id=MODEL_ID

        )


        predictions = result.get(

            "predictions",

            []

        )


        # Remove weak detections to reduce false positives.
        CONFIDENCE_THRESHOLD = 0.50

        predictions = [
            p for p in predictions
            if p.get("confidence", 0) >= CONFIDENCE_THRESHOLD
        ]


        fish_count = len(
            predictions
        )


    except Exception as e:

        return f"""
        <h2>Roboflow Error</h2>

        <p>{e}</p>

        <a href="/">Go Back</a>
        """


    # =====================================================
    # DRAW BOUNDING BOXES
    # =====================================================

    result_image_path = os.path.join(

        "static",

        "fish_result.jpg"

    )


    draw_fish_boxes(

        converted_path,

        predictions,

        result_image_path

    )


    # =====================================================
    # RESULT PAGE
    # =====================================================

    return f"""
    <!DOCTYPE html>

    <html>

    <head>

        <title>
            Fish Detection Result
        </title>

        <style>

            body {{

                margin: 0;

                padding: 40px;

                font-family:
                    Arial,
                    sans-serif;

                background:
                    linear-gradient(
                        135deg,
                        #e0f7fa,
                        #f0f9ff
                    );

                text-align: center;

            }}


            .result-card {{

                max-width: 900px;

                margin: auto;

                background: white;

                padding: 35px;

                border-radius: 25px;

                box-shadow:
                    0 15px 40px
                    rgba(
                        0,
                        100,
                        150,
                        0.15
                    );

            }}


            h1 {{

                color: #075985;

            }}


            .count {{

                font-size: 42px;

                font-weight: bold;

                color: #059669;

                margin: 20px;

            }}


            img {{

                width: 90%;

                max-width: 800px;

                border-radius: 18px;

                margin-top: 20px;

                border:
                    3px solid #0284c7;

            }}


            .back-btn {{

                display: inline-block;

                margin-top: 25px;

                padding: 13px 28px;

                background: #0284c7;

                color: white;

                text-decoration: none;

                border-radius: 12px;

            }}

        </style>

    </head>


    <body>


        <div class="result-card">


            <h1>
                🐟 Fish Detection Result
            </h1>


            <div class="count">

                Fish Count:
                {fish_count}

            </div>


            <p>

                Image:
                {image.filename}

            </p>


            <img

                src="/static/fish_result.jpg?t={fish_count}"

                alt="Fish Detection Result"

            >


            <br>


            <a

                class="back-btn"

                href="/"

            >

                ← Back to Fish Counter

            </a>


        </div>


    </body>

    </html>
    """


# =========================================================
# VIDEO UPLOAD
# =========================================================

@app.route(
    "/upload_video",
    methods=["POST"]
)
def upload_video():

    video = request.files.get(
        "fish_video"
    )


    # =====================================================
    # CHECK VIDEO
    # =====================================================

    if not video:

        return """
        <h2>No video selected</h2>

        <a href="/">Go Back</a>
        """


    # =====================================================
    # SAVE VIDEO
    # =====================================================

    video_path = os.path.join(

        app.config["UPLOAD_FOLDER"],

        video.filename

    )


    video.save(
        video_path
    )


    # =====================================================
    # PROCESS VIDEO
    # =====================================================

    try:

        result = process_video_data(

            video_path

        )


        return render_video_result(

            result["count"],

            result["video"],

            video.filename

        )


    except Exception as e:

        return f"""
        <h2>Video processing error</h2>

        <p>{e}</p>

        <a href="/">Go Back</a>
        """


# =========================================================
# CAMERA PHOTO
# =========================================================

@app.route(
    "/capture_photo",
    methods=["POST"]
)
def capture_photo():

    photo = request.files.get(
        "photo"
    )


    # =====================================================
    # CHECK PHOTO
    # =====================================================

    if not photo:

        return jsonify({

            "success": False,

            "error":
                "No photo received"

        })


    # =====================================================
    # SAVE CAMERA PHOTO
    # =====================================================

    captured_path = os.path.join(

        app.config["UPLOAD_FOLDER"],

        "captured_fish.jpg"

    )


    photo.save(
        captured_path
    )


    try:

        # =================================================
        # ROBOFLOW DETECTION
        # =================================================

        result = client.infer(

            captured_path,

            model_id=MODEL_ID

        )


        predictions = result.get(

            "predictions",

            []

        )


        fish_count = len(
            predictions
        )


        # =================================================
        # DRAW BOUNDING BOXES
        # =================================================

        result_path = os.path.join(

            "static",

            "captured_fish_result.jpg"

        )


        draw_fish_boxes(

            captured_path,

            predictions,

            result_path

        )


        # =================================================
        # RETURN RESULT
        # =================================================

        return jsonify({

            "success":
                True,

            "count":
                fish_count,

            "image":
                "/static/captured_fish_result.jpg"

        })


    except Exception as e:

        print(
            "Camera photo error:",
            e
        )


        return jsonify({

            "success":
                False,

            "error":
                str(e)

        })


# =========================================================
# CAMERA VIDEO
# =========================================================

@app.route(
    "/camera_video",
    methods=["POST"]
)
def camera_video():

    video = request.files.get(
        "camera_video"
    )


    # =====================================================
    # CHECK VIDEO
    # =====================================================

    if not video:

        return jsonify({

            "success":
                False,

            "error":
                "No camera video received"

        })


    # =====================================================
    # SAVE CAMERA VIDEO
    # =====================================================

    video_path = os.path.join(

        app.config["UPLOAD_FOLDER"],

        "camera_recording.webm"

    )


    video.save(
        video_path
    )


    # =====================================================
    # PROCESS CAMERA VIDEO
    # =====================================================

    try:

        result = process_video_data(

            video_path

        )


        return jsonify({

            "success":
                True,

            "count":
                result["count"],

            "video":
                result["video"]

        })


    except Exception as e:

        print(
            "Camera video error:",
            e
        )


        return jsonify({

            "success":
                False,

            "error":
                str(e)

        })


# =========================================================
# VIDEO PROCESSING + TRACKING
# =========================================================

def process_video_data(video_path):
    # =====================================================
    # OPEN VIDEO
    # =====================================================

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception(
            "Could not open video. "
            "Please try a shorter video."
        )

    # =====================================================
    # VIDEO INFORMATION
    # =====================================================

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    if fps <= 0:
        fps = 20

    if width <= 0 or height <= 0:
        cap.release()
        raise Exception(
            "Invalid video dimensions."
        )

    # =====================================================
    # BYTE TRACK
    # =====================================================

    tracker = sv.ByteTrack()

    # IMPORTANT:
    # Do NOT count all unique tracker IDs.
    # ByteTrack can assign a new ID to the same fish
    # after a temporary missed detection.
    #
    # Instead, we count the fish visible in each
    # sampled frame and use the median as the final
    # stable count.

    CONFIDENCE_THRESHOLD = 0.40

    # Process every 5th frame.
    # Roboflow API inference is the slowest part, so this
    # greatly reduces waiting time while keeping tracking/boxes.
    frame_skip = 5

    frame_number = 0

    # Fish count detected in each sampled frame.
    frame_counts = []

    # Store the most recent boxes + tracking IDs.
    last_boxes = []

    # =====================================================
    # TEMPORARY VIDEO
    # =====================================================

    temp_video = os.path.join(
        "static",
        "tracked_fish_temp.avi"
    )

    # =====================================================
    # FINAL MP4 VIDEO
    # =====================================================

    final_video = os.path.join(
        "static",
        "tracked_fish.mp4"
    )

    # =====================================================
    # CREATE TEMP VIDEO
    # =====================================================

    fourcc = cv2.VideoWriter_fourcc(
        *"MJPG"
    )

    out = cv2.VideoWriter(
        temp_video,
        fourcc,
        fps,
        (
            width,
            height
        )
    )

    if not out.isOpened():
        cap.release()
        raise Exception(
            "Could not create temporary video."
        )

    # =====================================================
    # READ VIDEO FRAME BY FRAME
    # =====================================================

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # Show progress in CMD without printing every frame.
        if frame_number % 50 == 0:
            print(
                f"Processing frame {frame_number}..."
            )

        # =================================================
        # AI DETECTION
        # =================================================

        if frame_number % frame_skip == 0:

            try:

                # -----------------------------------------
                # ROBOFLOW
                # -----------------------------------------

                result = client.infer(
                    frame,
                    model_id=MODEL_ID
                )

                # -----------------------------------------
                # Convert to supervision detections
                # -----------------------------------------

                detections = (
                    sv.Detections
                    .from_inference(
                        result
                    )
                )

                # -----------------------------------------
                # CONFIDENCE FILTER
                # -----------------------------------------
                #
                # Remove weak detections.
                # This helps reduce false positives.
                #

                if detections.confidence is not None:

                    confidence_mask = (
                        detections.confidence
                        >= CONFIDENCE_THRESHOLD
                    )

                    detections = (
                        detections[
                            confidence_mask
                        ]
                    )

                # -----------------------------------------
                # BYTE TRACK
                # -----------------------------------------

                detections = (
                    tracker
                    .update_with_detections(
                        detections
                    )
                )

                # -----------------------------------------
                # CURRENT FRAME COUNT
                # -----------------------------------------

                current_count = len(
                    detections
                )

                frame_counts.append(
                    current_count
                )

                # Clear old boxes.
                last_boxes = []

                # -----------------------------------------
                # GET TRACK IDs
                # -----------------------------------------

                if (
                    detections.tracker_id
                    is not None
                ):

                    for i in range(
                        len(
                            detections.xyxy
                        )
                    ):

                        box = (
                            detections
                            .xyxy[i]
                        )

                        track_id = int(
                            detections
                            .tracker_id[i]
                        )

                        # ---------------------------------
                        # BOX COORDINATES
                        # ---------------------------------

                        x1 = max(
                            0,
                            int(
                                box[0]
                            )
                        )

                        y1 = max(
                            0,
                            int(
                                box[1]
                            )
                        )

                        x2 = min(
                            width - 1,
                            int(
                                box[2]
                            )
                        )

                        y2 = min(
                            height - 1,
                            int(
                                box[3]
                            )
                        )

                        # ---------------------------------
                        # SAVE BOX
                        # ---------------------------------

                        last_boxes.append(
                            (
                                x1,
                                y1,
                                x2,
                                y2,
                                track_id
                            )
                        )

            except Exception as e:

                print(
                    "Frame inference error:",
                    e
                )

        # =================================================
        # DRAW BOUNDING BOXES
        # =================================================

        for box_data in last_boxes:

            (
                x1,
                y1,
                x2,
                y2,
                track_id
            ) = box_data

            # -------------------------------------------------
            # GREEN BOX
            # -------------------------------------------------

            cv2.rectangle(
                frame,
                (
                    x1,
                    y1
                ),
                (
                    x2,
                    y2
                ),
                (
                    0,
                    255,
                    0
                ),
                3
            )

            # -------------------------------------------------
            # FISH ID
            # -------------------------------------------------

            cv2.putText(
                frame,
                f"Fish {track_id}",
                (
                    x1,
                    max(
                        y1 - 10,
                        25
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (
                    0,
                    255,
                    0
                ),
                2
            )

        # =================================================
        # FISH COUNT
        # =================================================
        #
        # Do not draw a changing count on the video.
        # The final stable count is calculated after all
        # sampled frames have been processed and is shown
        # on the result page.
        #

        # =================================================
        # WRITE ANNOTATED FRAME
        # =================================================

        out.write(
            frame
        )

    # =====================================================
    # CLOSE VIDEO
    # =====================================================

    cap.release()
    out.release()

    # =====================================================
    # CONVERT AVI TO H264 MP4
    # =====================================================

    print(
        "Converting video to browser-compatible MP4..."
    )

    ffmpeg_exe = get_ffmpeg_exe()

    command = [
        ffmpeg_exe,
        "-y",
        "-i",
        temp_video,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        final_video
    ]

    conversion_result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # =====================================================
    # CHECK FFMPEG RESULT
    # =====================================================

    if conversion_result.returncode != 0:

        print(
            conversion_result.stderr
        )

        if os.path.exists(
            temp_video
        ):
            os.remove(
                temp_video
            )

        raise Exception(
            "Video conversion to MP4 failed."
        )

    # =====================================================
    # DELETE TEMP AVI
    # =====================================================

    if os.path.exists(
        temp_video
    ):
        os.remove(
            temp_video
        )

    # =====================================================
    # FINAL COUNT
    # =====================================================

    if frame_counts:
        final_count = round(
            __import__("statistics")
            .median(frame_counts)
        )
    else:
        final_count = 0

    print(
        "Sampled frame counts:",
        frame_counts
    )

    print(
        "Final Fish Count:",
        final_count
    )

    # =====================================================
    # RETURN RESULT
    # =====================================================

    return {
        "count":
            final_count,

        "video":
            "/static/tracked_fish.mp4"
    }


# =========================================================
# VIDEO RESULT PAGE
# =========================================================

def render_video_result(

    count,

    video_url,

    filename

):

    return f"""
    <!DOCTYPE html>

    <html>

    <head>

        <title>
            Fish Tracking Result
        </title>


        <style>

            body {{

                margin: 0;

                padding: 40px;

                font-family:
                    Arial,
                    sans-serif;

                background:
                    linear-gradient(
                        135deg,
                        #e0f7fa,
                        #f0f9ff
                    );

                text-align: center;

            }}


            .result-card {{

                max-width: 900px;

                margin: auto;

                padding: 35px;

                background: white;

                border-radius: 25px;

                box-shadow:
                    0 15px 40px
                    rgba(
                        0,
                        100,
                        150,
                        0.15
                    );

            }}


            h1 {{

                color: #075985;

            }}


            .count {{

                font-size: 42px;

                font-weight: bold;

                color: #059669;

                margin: 25px;

            }}


            video {{

                width: 90%;

                max-width: 800px;

                border-radius: 18px;

                border:
                    3px solid #0284c7;

                background: black;

            }}


            .back-btn {{

                display: inline-block;

                margin-top: 25px;

                padding: 13px 28px;

                background: #0284c7;

                color: white;

                text-decoration: none;

                border-radius: 12px;

            }}

        </style>

    </head>


    <body>


        <div class="result-card">


            <h1>  🎥 Fish Tracking Result </h1>


            <div class="count">

                🐟 Fish Count:
                {count}

            </div>


            <p>

                Video:
                {filename}

            </p>


            <video

                controls

                playsinline

                preload="metadata"

            >

                <source

                    src="{video_url}?t={count}"

                    type="video/mp4"

                >

                Your browser does not
                support video playback.

            </video>


            <br>
            <a
                class="back-btn"
                href="/"
            >
                ← Back to Fish Counter
            </a>
        </div>
    </body>
    </html>
    """


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(

        debug=True

    )
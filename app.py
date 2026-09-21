import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os
from PIL import Image

# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = "best.pt"

CLASS_NAMES = [
    "Hardhat",
    "Mask",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Person",
    "Safety Cone",
    "Safety Vest",
    "machinery",
    "vehicle"
]

# Classes that indicate PPE violations
NO_HARDHAT = "NO-Hardhat"
NO_VEST = "NO-Safety Vest"

# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="VisionGuard AI",
    page_icon="🦺",
    layout="wide"
)

st.title("🦺 VisionGuard AI")
st.subheader("Construction Safety Monitoring System")

st.write(
    "Upload an image or video and VisionGuard AI will detect "
    "people and safety equipment."
)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error("Could not load the YOLO model.")
    st.code(str(e))
    st.stop()

# =========================================================
# FUNCTIONS
# =========================================================

def analyze_results(result):

    detected_classes = []

    if result.boxes is not None:
        for cls in result.boxes.cls:
            class_id = int(cls)

            if 0 <= class_id < len(CLASS_NAMES):
                detected_classes.append(CLASS_NAMES[class_id])

    people = detected_classes.count("Person")

    no_hardhat = detected_classes.count(NO_HARDHAT)
    no_vest = detected_classes.count(NO_VEST)

    violations = []

    if no_hardhat > 0:
        violations.append("No Hardhat detected")

    if no_vest > 0:
        violations.append("No Safety Vest detected")

    if len(violations) == 0:
        status = "SAFE"
    else:
        status = "UNSAFE"

    return {
        "status": status,
        "people": people,
        "no_hardhat": no_hardhat,
        "no_vest": no_vest,
        "violations": violations,
        "detected_classes": detected_classes
    }


def draw_status(image, analysis):

    if analysis["status"] == "SAFE":
        status_text = "SAFE"
    else:
        status_text = "UNSAFE"

    cv2.rectangle(
        image,
        (10, 10),
        (390, 80),
        (0, 180, 0) if status_text == "SAFE" else (0, 0, 220),
        -1
    )

    cv2.putText(
        image,
        status_text,
        (30, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (255, 255, 255),
        3
    )

    return image


def show_analysis(analysis):

    st.markdown("---")

    if analysis["status"] == "SAFE":
        st.success("🟢 SAFE — No PPE violation detected")
    else:
        st.error("🔴 UNSAFE — PPE violation detected")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "People Detected",
            analysis["people"]
        )

    with col2:
        st.metric(
            "No Hardhat",
            analysis["no_hardhat"]
        )

    with col3:
        st.metric(
            "No Safety Vest",
            analysis["no_vest"]
        )

    if analysis["violations"]:

        st.warning("Detected violations:")

        for violation in analysis["violations"]:
            st.write("❌ " + violation)

    else:

        st.success(
            "✅ All detected PPE classes are compliant."
        )

    if analysis["detected_classes"]:

        with st.expander("Detected Classes"):

            for name in analysis["detected_classes"]:
                st.write("• " + name)


# =========================================================
# UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "mp4",
        "avi",
        "mov",
        "mkv"
    ]
)

# =========================================================
# PROCESS FILE
# =========================================================

if uploaded_file is not None:

    file_name = uploaded_file.name.lower()

    # =====================================================
    # IMAGE
    # =====================================================

    if file_name.endswith(
        (".jpg", ".jpeg", ".png", ".bmp")
    ):

        st.header("📷 Image Analysis")

        file_bytes = uploaded_file.read()

        image_array = np.frombuffer(
            file_bytes,
            np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is None:

            st.error("Could not read the image.")
            st.stop()

        # YOLO prediction
        results = model.predict(
            source=image,
            conf=0.25,
            verbose=False
        )

        result = results[0]

        # Annotated image
        annotated = result.plot()

        # Analysis
        analysis = analyze_results(result)

        annotated = draw_status(
            annotated,
            analysis
        )

        # Convert BGR -> RGB
        annotated_rgb = cv2.cvtColor(
            annotated,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            annotated_rgb,
            caption="VisionGuard Detection",
            use_container_width=True
        )

        show_analysis(analysis)

    # =====================================================
    # VIDEO
    # =====================================================

    elif file_name.endswith(
        (".mp4", ".avi", ".mov", ".mkv")
    ):

        st.header("🎥 Video Analysis")

        # Save uploaded video temporarily
        temp_video = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(file_name)[1]
        )

        temp_video.write(
            uploaded_file.read()
        )

        temp_video.close()

        cap = cv2.VideoCapture(
            temp_video.name
        )

        if not cap.isOpened():

            st.error("Could not open the video.")
            os.unlink(temp_video.name)
            st.stop()

        frame_placeholder = st.empty()
        status_placeholder = st.empty()

        frame_count = 0

        total_people = 0
        total_no_hardhat = 0
        total_no_vest = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            # Process every 3rd frame
            if frame_count % 3 != 0:
                continue

            results = model.predict(
                source=frame,
                conf=0.25,
                verbose=False
            )

            result = results[0]

            analysis = analyze_results(
                result
            )

            total_people = max(
                total_people,
                analysis["people"]
            )

            total_no_hardhat = max(
                total_no_hardhat,
                analysis["no_hardhat"]
            )

            total_no_vest = max(
                total_no_vest,
                analysis["no_vest"]
            )

            annotated = result.plot()

            annotated = draw_status(
                annotated,
                analysis
            )

            annotated_rgb = cv2.cvtColor(
                annotated,
                cv2.COLOR_BGR2RGB
            )

            frame_placeholder.image(
                annotated_rgb,
                caption=f"Frame {frame_count}",
                use_container_width=True
            )

            if analysis["status"] == "SAFE":

                status_placeholder.success(
                    "🟢 SAFE — No PPE violation detected"
                )

            else:

                status_placeholder.error(
                    "🔴 UNSAFE — PPE violation detected"
                )

        cap.release()

        os.unlink(temp_video.name)

        # =================================================
        # FINAL VIDEO SUMMARY
        # =================================================

        st.markdown("---")

        st.header("📊 Video Safety Summary")

        if (
            total_no_hardhat == 0
            and total_no_vest == 0
        ):

            st.success(
                "🟢 SAFE — No PPE violations were detected."
            )

        else:

            st.error(
                "🔴 UNSAFE — PPE violations were detected."
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "People Detected",
                total_people
            )

        with col2:
            st.metric(
                "No Hardhat",
                total_no_hardhat
            )

        with col3:
            st.metric(
                "No Safety Vest",
                total_no_vest
            )

st.markdown("---")

st.caption(
    "VisionGuard AI | Computer Vision PPE Safety Monitoring"
)

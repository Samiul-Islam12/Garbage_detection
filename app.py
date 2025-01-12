import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile

# Load the YOLOv11 model
model = YOLO("best.pt")

# Streamlit App Configuration
st.title("YOLOv11: Real-Time Object Detection")
st.sidebar.title("Options")
mode = st.sidebar.selectbox("Select Mode", ["Webcam", "Image Upload", "Video Upload"])

st.sidebar.write("Use the options above to toggle between modes.")

FRAME_WINDOW = st.empty()  # For displaying webcam frames
uploaded_file = None       # Placeholder for file uploads

# **Function for Image Detection**
def detect_image(image):
    results = model(image, imgsz=640)
    annotated_frame = results[0].plot()
    return annotated_frame

# **Function for Video Detection**
def detect_video(video_path):
    cap = cv2.VideoCapture(video_path)
    output_frames = []
    success, frame = cap.read()

    while success:
        results = model(frame, imgsz=640)
        annotated_frame = results[0].plot()
        output_frames.append(annotated_frame)
        success, frame = cap.read()
    
    cap.release()
    return output_frames

# **Webcam Detection**
if mode == "Webcam":
    run_detection = st.sidebar.checkbox("Run Webcam Detection", value=False)

    if run_detection:
        cap = cv2.VideoCapture(0)  # Open default webcam
        if not cap.isOpened():
            st.error("Webcam not found or unavailable. Please check your device.")
        else:
            st.write("Webcam is running. Disable the checkbox to stop detection.")
            while run_detection:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to read from webcam. Please check your device.")
                    break

                # Detect and display
                annotated_frame = detect_image(frame)
                FRAME_WINDOW.image(annotated_frame, channels="BGR", use_container_width=True)

            cap.release()

# **Image Upload**
elif mode == "Image Upload":
    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.write("Uploaded Image and Detection Result:")
        
        # Convert PIL image to NumPy array for processing
        image_np = np.array(image)

        # Detect objects
        annotated_frame = detect_image(image_np)

        # Display images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        with col2:
            st.image(annotated_frame, caption="Detected Objects", use_container_width=True)

# **Video Upload**
elif mode == "Video Upload":
    uploaded_file = st.sidebar.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)  # Save uploaded file temporarily
        tfile.write(uploaded_file.read())

        st.video(tfile.name, format="video/mp4")
        st.write("Processing video...")

        # Detect objects in the video
        frames = detect_video(tfile.name)
        
        # Create output video
        height, width, _ = frames[0].shape
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (width, height))

        for frame in frames:
            out.write(frame)
        out.release()

        st.video(output_path, format="video/mp4")
        st.success("Video processing completed!")

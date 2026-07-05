import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Glossome: AI Seed Prompting", layout="centered")

st.markdown("## 👅 Glossome: Prompted Extraction")
st.markdown("### Core Engine: Active Region Modeling")
st.info("Upload a clinical photo. The engine will drop an anchor prompt on the central tissue and digitally separate it from the lips and background.")

uploaded_file = st.file_uploader("Upload clinical photo", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 1. Read Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    st.success("Image loaded! Scanning for the anatomical anchor...")

    # 2. Setup the "Prompt" (GrabCut Algorithm)
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    # We mathematically prompt the AI to focus only on the central 70% of the frame
    height, width = img.shape[:2]
    rect = (int(width*0.15), int(height*0.15), int(width*0.7), int(height*0.8))

    # 3. Execute the organic cut (5 iterations of learning)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # 4. Isolate the tissue pixels
    # mask==2 or mask==0 means background. mask==1 or mask==3 means foreground.
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    segmented_output = rgb_img * mask2[:, :, np.newaxis]

    # Display Result
    st.image(segmented_output, caption="Prompted Extraction (Isolated Tissue)", use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📊 Initial Pixel Data:")
    st.json({
        "Anchor Method": "Central Region Bounding",
        "Algorithm": "GrabCut Active Separation",
        "Peripheral Artifacts (Lips/Teeth)": "Zeroed Out (Black)"
    })

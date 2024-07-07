import streamlit as st
import os
from PIL import Image
import cv2
import os
from deepface import DeepFace
from collections import defaultdict
import numpy as np

def process_image(input_image, output_folder):
    temp_input_image_path = "temp_input_image.png"
    input_image.save(temp_input_image_path)
    images_with_boxes = []
    dfs = DeepFace.find(
      img_path = temp_input_image_path,
      db_path = output_folder,
      model_name = 'ArcFace',
      distance_metric = 'cosine',
      detector_backend='mtcnn',
      enforce_detection=False,
    )
    d = dfs[0]
    for i in range(len(d)):
      add = d["identity"][i]
      x, y, w, h = d["target_x"][i], d["target_y"][i], d["target_w"][i], d["target_h"][i]

      image = cv2.imread(add)
      cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
      images_with_boxes.append(image)
    return images_with_boxes


def load_all_images(directory_path):
    try:
        all_images = []
        for filename in os.listdir(directory_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(directory_path, filename)
                img = cv2.imread(img_path)
                all_images.append(img)
        return all_images
    except Exception as e:
        st.error(f"Error loading images from directory: {e}")
        return []




def main():
    st.set_page_config(page_title="Face Similarity App", layout="wide")

    # Sidebar for inputs
    st.sidebar.title("Face Similarity App")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    directory_path = "b_image"
    show_all_images = st.sidebar.checkbox("Show all images in directory")

    st.sidebar.markdown("""
        <style>
            .sidebar .sidebar-content {
                background-color: #f0f2f6;
                padding: 10px;
                border-radius: 10px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Main area
    st.title("Face Similarity App using DeepFace")
    st.markdown("""
        <style>
            .main-content {
                background-color: #ffffff;
                padding: 20px;
                border-radius: 10px;
            }
        </style>
    """, unsafe_allow_html=True)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if directory_path:
            if st.button("Process Image"):
                # Process the image
                processed_images = process_image(image, directory_path)

                if processed_images:
                    st.subheader("Matched Images:")
                    cols = st.columns(3)  # Number of columns in the grid
                    for idx, img in enumerate(processed_images):
                        with cols[idx % 3]:
                            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f'Matched Image {idx+1}', use_column_width=True)
                else:
                    st.write("No matching images found.")

            if show_all_images:
                all_images = load_all_images(directory_path)
                if all_images:
                    st.subheader("All Images in Directory:")
                    cols = st.columns(3)  # Number of columns in the grid
                    for idx, img in enumerate(all_images):
                        with cols[idx % 3]:
                            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f'Image {idx+1}', use_column_width=True)
                else:
                    st.write("No images found in the directory.")

if __name__ == "__main__":
    main()
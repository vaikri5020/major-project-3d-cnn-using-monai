import os
import tempfile

import torch
import streamlit as st

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    ResizeWithPadOrCropd,
    ToTensord
)

from model import get_model


st.title("Brain Tumor Detection")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model():
    model = get_model().to(device)

    model.load_state_dict(
        torch.load(
            "saved_models/brain_tumor_unet.pth",
            map_location=device
        )
    )

    model.eval()
    return model


transform = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),

    NormalizeIntensityd(
        keys=["image"],
        nonzero=True,
        channel_wise=True
    ),

    ResizeWithPadOrCropd(
        keys=["image"],
        spatial_size=(64, 64, 64)
    ),

    ToTensord(keys=["image"])
])


flair = st.file_uploader("Upload FLAIR MRI")
t1 = st.file_uploader("Upload T1 MRI")
t1ce = st.file_uploader("Upload T1CE MRI")
t2 = st.file_uploader("Upload T2 MRI")


if st.button("Predict"):
    if not flair or not t1 or not t1ce or not t2:
        st.error("Please upload all 4 MRI files.")
    else:
        with st.spinner("Predicting..."):
            model = load_model()

            with tempfile.TemporaryDirectory() as temp_dir:
                paths = []

                files = [
                    ("flair.nii", flair),
                    ("t1.nii", t1),
                    ("t1ce.nii", t1ce),
                    ("t2.nii", t2),
                ]

                for filename, file in files:
                    path = os.path.join(temp_dir, filename)

                    with open(path, "wb") as f:
                        f.write(file.read())

                    paths.append(path)

                data = {
                    "image": paths
                }

                data = transform(data)

                image = data["image"]
                image = image.unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(image)

                mask = torch.argmax(output, dim=1)

                unique_values = torch.unique(mask).cpu().numpy().tolist()

                st.success("Prediction completed")

                st.write("Output mask shape:")
                st.write(mask.shape)

                st.write("Detected mask classes:")
                st.write(unique_values)

                if len(unique_values) > 1:
                    st.success("Tumor region detected")
                else:
                    st.warning("No tumor region detected")
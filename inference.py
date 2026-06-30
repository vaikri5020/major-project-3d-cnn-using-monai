import torch

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    ResizeWithPadOrCropd,
    ToTensord
    
)

from model import get_model

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = get_model().to(device)

model.load_state_dict(
    torch.load(
        "saved_models/brain_tumor_unet.pth",
        map_location=device
    )
)

model.eval()

transform = Compose([
    LoadImaged(keys=['image']),
    EnsureChannelFirstd(keys=["image"]),
    NormalizeIntensityd(
        keys=["image"],
        nonzero=True,
        channel_wise=True
    ),
    ResizeWithPadOrCropd(
        keys=["image"],
        spatial_size=(96, 96, 96)
    ),
    ToTensord(keys=["image"])
    
])
patient_dir = r"C:\Users\admin\Downloads\archive\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001"


data = {
    "image": [
        patient_dir + r"\BraTS20_Training_001_flair.nii",
        patient_dir + r"\BraTS20_Training_001_t1.nii",
        patient_dir + r"\BraTS20_Training_001_t1ce.nii",
        patient_dir + r"\BraTS20_Training_001_t2.nii",
    ]
}

data = transform(data)
image=data["image"]

image = image.unsqueeze(0).to(device)

with torch.no_grad():

    prediction = model(image)

mask = torch.argmax(
    prediction,
    dim=1
)

print(mask.shape)
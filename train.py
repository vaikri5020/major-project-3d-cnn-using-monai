import os

import numpy as np
import torch

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    ResizeWithPadOrCropd,
    ToTensord,
    Lambda
)

from monai.data import Dataset
from monai.data import DataLoader,CacheDataset

from monai.losses import DiceLoss

from model import get_model
from dataset import get_data

print("1 import done")

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(torch.cuda.is_available())
print(device)
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

data = get_data()
print("3 get data done")

def convert_label_to_int(data):
    # BraTS labels are 0=background, 1=necrotic/cystic, 2=edema, 4=enhancing tumor.
    # We map them to contiguous class ids [0,1,2,3] for 4 output channels.
    mapping = {0: 0, 1: 1, 2: 2, 4: 3}
    label = data["label"]
    out = np.zeros_like(label, dtype=np.int64)
    for src, dst in mapping.items():
        out[label == src] = dst
    data["label"] = out
    return data

transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityd(keys=["image"]),

     ResizeWithPadOrCropd(
        keys=["image", "label"],
        spatial_size=(64,64,64)
    ),

    
    Lambda(convert_label_to_int),
    ToTensord(keys=["image", "label"])
])
print("4 trandform done")

dataset = CacheDataset(
    data=data[:10],
    transform=transforms,
    cache_rate=0.1,
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)
print("6 dataloader done")

model = get_model().to(device)
print("7 model created")

loss_fn = DiceLoss(
    to_onehot_y=True,
    softmax=True
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

epochs = 2
print("starting training loop")

for epoch in range(epochs):

    model.train()

    epoch_loss = 0
    print("before datset")

    for batch in loader:

        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images)

        # MONAI DiceLoss with to_onehot_y=True expects labels in integer class form.
        loss = loss_fn(
            outputs,
            labels
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    print(
        f"Epoch {epoch+1} Loss: "
        f"{epoch_loss/len(loader):.4f}"
    )

print("saving model")
os.makedirs("saved_models", exist_ok=True)
torch.save(
    model.state_dict(),
    "saved_models/brain_tumor_unet.pth"
)
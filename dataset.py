import os

def get_data():
    root = r"C:\Users\admin\Downloads\archive\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"

    data = []

    for patient in sorted(os.listdir(root)):
        patient_dir = os.path.join(root, patient)

        if not os.path.isdir(patient_dir):
            continue

        image_files = [
            os.path.join(patient_dir, f"{patient}_flair.nii"),
            os.path.join(patient_dir, f"{patient}_t1.nii"),
            os.path.join(patient_dir, f"{patient}_t1ce.nii"),
            os.path.join(patient_dir, f"{patient}_t2.nii"),
        ]

        label_file = os.path.join(patient_dir, f"{patient}_seg.nii")

        if all(os.path.exists(f) for f in image_files) and os.path.exists(label_file):
            data.append({
                "image": image_files,
                "label": label_file
            })

    print("Total samples:", len(data))
    print("First sample:", data[0])

    return data
import os, random
from PIL import Image

def create_sample(src, dst, n=50, size=(64, 64)):
    os.makedirs(dst, exist_ok=True)
    files = [f for f in os.listdir(src) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
    selected = random.sample(files, min(n, len(files)))

    for fname in selected:
        try:
            img = Image.open(os.path.join(src, fname)).convert('L')  # grayscale
            img = img.resize(size)
            img.save(os.path.join(dst, fname))
        except Exception as e:
            print(f"Skipped {fname}: {e}")

base = "data/chest_xray"
splits = ["train", "val", "test"]
classes = ["NORMAL", "PNEUMONIA"]

for split in splits:
    for cls in classes:
        src = f"{base}/{split}/{cls}"
        dst = f"{base}_small/{split}/{cls.lower()}"
        create_sample(src, dst, n=50)

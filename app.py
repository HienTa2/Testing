# =============================
# IMPORTS
# =============================
import sys
import shutil
import os
import urllib.request
from pathlib import Path
from fastai.vision.all import *
from fastdownload import download_url
from fastai.vision.widgets import ImageClassifierCleaner
from duckduckgo_search import DDGS
import gradio as gr

# Ensure correct package path (if needed)
sys.path.append(r"C:\Users\Admin\PycharmProjects\PythonProject\Claims_Parsing\.venv\Lib\site-packages")

# =============================
# IMAGE DATA COLLECTION
# =============================

# Set base path for storing images
path = Path('PETS_images')

# Remove previous dataset and start fresh
if path.exists():
    shutil.rmtree(path)

# Categories to download
categories = ['cats', 'dogs']

# Function to search images using DuckDuckGo
def search_images_duckduckgo(query, max_images=20):
    """Search for images using DuckDuckGo and return a list of image URLs."""
    with DDGS() as ddgs:
        return [r["image"] for r in ddgs.images(query, max_results=max_images)]

# Download images for each category
for category in categories:
    dest = path/category
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Searching images for: {category}")
    urls = search_images_duckduckgo(f'{category} pets', max_images=20)
    print(f"Found {len(urls)} images. Downloading...")

    for idx, url in enumerate(urls):
        try:
            image_path = dest/f'{category}_{idx}.jpg'
            urllib.request.urlretrieve(url, image_path)  # Alternative to download_url
            print(f"Downloaded: {image_path}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")

# Remove any corrupted images
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)

# Display counts of downloaded images
for folder in categories:
    count = len(get_image_files(path/folder))
    print(f"{folder} images:", count)

print("✅ Fastai imported and images processed successfully!")

# =============================
# DATA BLOCK & MODEL TRAINING
# =============================

# Define DataBlock for classification
pets = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=RandomResizedCrop(128, min_scale=0.3)
)

# Create DataLoaders
dls = pets.dataloaders(path, bs=8)  
dls.train.show_batch(max_n=8, unique=True)

# Train the model
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(5)
learn.show_results()

# =============================
# MODEL INTERPRETATION
# =============================

# Confusion Matrix
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

# Show top losses
interp.plot_top_losses(4, nrows=1)

# Clean misclassified images
cleaner = ImageClassifierCleaner(learn)

# List image paths (optional)
for i in range(len(cleaner.fns)):
    print(f"{i}: {cleaner.fns[i]}")

# =============================
# EXPORT & LOAD MODEL
# =============================

# Export trained model
learn.export('model.pkl')

# Load model for inference
learn = load_learner('model.pkl')

# =============================
# SINGLE IMAGE PREDICTION
# =============================


# Load and predict on a sample image
#img_path = 'PETS_images/cats/cats_0.jpg'  
#img = PILImage.create(img_path)
#pred_class, pred_idx, probs = learn.predict(img)
#print(f"Prediction: {pred_class}, Probability: {probs[pred_idx]:.4f}")
###

# =============================
# BATCH IMAGE PREDICTION
# =============================

# Predict all images in a directory
test_folder = Path('PETS_images/cats')

for img_path in test_folder.iterdir():
    img = PILImage.create(img_path)
    pred_class, pred_idx, probs = learn.predict(img)
    print(f"Image: {img_path.name}, Prediction: {pred_class}, Probability: {probs[pred_idx]:.4f}")

# Predict all images in dataset
for img_path in path.rglob("*.jpg"):
    img = PILImage.create(img_path)
    pred_class, pred_idx, probs = learn.predict(img)
    print(f"Image: {img_path}, Prediction: {pred_class}, Probability: {probs[pred_idx]:.4f}")

# =============================
# DEPLOY AS A WEB APP WITH GRADIO
# =============================

# Load trained model
learn = load_learner('model.pkl')

# Define classification function for Gradio
def classify_image(img):
    pred_class, pred_idx, probs = learn.predict(img)
    return {str(pred_class): float(probs[pred_idx])}

# Create Gradio interface
intf = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(),
    allow_flagging="never",
    examples=[
        ['https://raw.githubusercontent.com/HienTa2/Testing/main/PETS_images/dogs/dogs_0.jpg'],
        ['https://raw.githubusercontent.com/HienTa2/Testing/main/PETS_images/cats/cats_1.jpg'],
        ['https://raw.githubusercontent.com/HienTa2/Testing/main/PETS_images/cats/cats_10.jpg'],
        ['https://raw.githubusercontent.com/HienTa2/Testing/main/PETS_images/dogs/dogs_10.jpg']
    ]
)

# Launch Web App
intf.launch(inline=False)

# =============================
# MISCELLANEOUS
# =============================

# Access model parameters (optional)
m = learn.model
ps = list(m.parameters())
print(ps[1])

# Check if settings.ini exists
print("settings.ini exists:", os.path.exists("settings.ini"))

# Export using nbdev (if needed)
from nbdev.doclinks import nbdev_export
nbdev_export()

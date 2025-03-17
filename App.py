# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python (DSproject)
#     language: python
#     name: dsproject
# ---
# +
#|export
import sys
import shutil
from fastai.vision.all import *
from fastdownload import download_url
from fastai.vision.widgets import ImageClassifierCleaner
from duckduckgo_search import DDGS
from pathlib import Path
import urllib.request

# Ensure correct package path
sys.path.append(r"C:\Users\Admin\PycharmProjects\PythonProject\Claims_Parsing\.venv\Lib\site-packages")

# Set base path for storing images
path = Path('PETS')

# Remove everything and start fresh
if path.exists():
    shutil.rmtree(path)

# Categories to download
categories = ['cats', 'dogs']

# Function to search images using DuckDuckGo
def search_images_duckduckgo(query, max_images=20):
    """Search for images using DuckDuckGo and return a list of image URLs."""
    with DDGS() as ddgs:
        return [r["image"] for r in ddgs.images(query, max_results=max_images)]

# Loop through each category and download images
for category in categories:
    dest = path/category
    dest.mkdir(parents=True, exist_ok=True)
    
    print(f"Searching images for: {category}")
    urls = search_images_duckduckgo(f'{category} pets', max_images=20)
    
    print(f"Found {len(urls)} images. Downloading now...")
    
    for idx, url in enumerate(urls):
        try:
            image_path = dest/f'{category}_{idx}.jpg'
            urllib.request.urlretrieve(url, image_path)  # Alternative to download_url
            print(f"Downloaded: {image_path}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")

# Verify images and remove any corrupted ones
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)

# Explicitly confirm counts
for folder in categories:
    count = len(get_image_files(path/folder))
    print(f"{folder} images:", count)

print("✅ Fastai imported and images processed successfully!")


# +
#|export
# Define DataBlock & DataLoader

pets = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=RandomResizedCrop(128, min_scale=0.3)) # RandomResizedCrop gets the same image, but with different size.
   
dls = pets.dataloaders(path, bs=8)  # smaller batch size for safety.
dls.train.show_batch(max_n=8, unique=True)

# +
#|export
# Train the model and fine tune
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(5)

# show the images
learn.show_results()

# +
#|export
# show the confusion matrix

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

# +
#|export
# top losses

interp.plot_top_losses(4, nrows=1)

# +
#|export
# Clean wrong label images

cleaner = ImageClassifierCleaner(learn)
cleaner

# +
#|export
# get the location of the images in case it needs to be removed.

for i in range(len(cleaner.fns)):
    print(f"{i}: {cleaner.fns[i]}")

# +
#|export
#for idx in [2]:  # Example: Delete images at indices 0, 5, and 10
#    os.remove(cleaner.fns[idx])
#    print(f"Deleted: {cleaner.fns[idx]}")

# +
#|export
# Exporting the model

learn.export('model.pkl')

# +
#|export
# load the model to learner

learn = load_learner('model.pkl')

# +
# Load a single image
img_path = 'PETS/cats/cats_0.jpg'  # Update with an actual image path
img = PILImage.create(img_path)


learn.predict(img)
# -

#|export
# Make a prediction
pred_class, pred_idx, probs = learn.predict(img)
print(f"Prediction: {pred_class}, Probability: {probs[pred_idx]:.4f}")

# +
#|export
# Path to single directories
test_folder = Path('PETS/cats')

# Loop through images in the folder
for img_path in test_folder.iterdir():
    img = PILImage.create(img_path)
    pred_class, pred_idx, probs = learn.predict(img)
    print(f"Image: {img_path.name}, Prediction: {pred_class}, Probability: {probs[pred_idx]:.4f}")


# +
#|export
# Define the main directory
pets_path = Path("PETS")

# Loop through all images in PETS (including cats and dogs)
for img_path in pets_path.rglob("*.jpg"):  # Adjust extension if needed
    img = PILImage.create(img_path)  # Load image
    pred_class, pred_idx, probs = learn.predict(img)  # Predict

    print(f"Image: {img_path}, Prediction: {pred_class}, Probability: {probs[pred_idx]:.4f}")

# +
#/export
import gradio as gr

# Load trained model
learn = load_learner('model.pkl')  # Ensure this file exists

# Define the classification function
def classify_image(img):
    pred_class, pred_idx, probs = learn.predict(img)
    return {str(pred_class): float(probs[pred_idx])}

# Create Gradio Interface
intf = gr.Interface(fn=classify_image, 
                    inputs=gr.Image(type="pil"), 
                    outputs=gr.Label(),
                    examples=['PETS/dogs/dogs_0.jpg', 'PETS/cats/cats_1.jpg'])

# Launch Web App
intf.launch(inline=False)

# -

m = learn.model

ps = list(m.parameters())

ps[1]

# ### Export

import os
print("settings.ini exists:", os.path.exists("settings.ini"))


from nbdev.doclinks import nbdev_export
nbdev_export()



# -*- coding: utf-8 -*-
"""microplastic_classify.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1zgriPUgVCe5BHrtR8oa0eWjCnca0HLnd
"""

import os

import glob

from IPython.display import Image,display

from IPython import display

display.clear_output()

import locale
locale.getpreferredencoding = lambda: "UTF-8"
!nvidia-smi

!pip install ultralytics

import ultralytics

ultralytics.checks()

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/datasets
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="Hb30Wlp685D8WdV1ViqW")
project = rf.workspace("yolov8-eruri").project("microplastic_minor")
version = project.version(1)
dataset = version.download("yolov5")

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/datasets

from ultralytics import YOLO


model = YOLO("yolov8m.yaml")


model = YOLO("yolov8m.pt")


results = model.train(data="/content/drive/MyDrive/datasets/microplastic_minor/data.yaml", epochs=70, imgsz=640)

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/datasets

Image(filename=f'/content/drive/MyDrive/datasets/runs/detect/train/confusion_matrix.png',width=900)

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/datasets

Image(filename=f'/content/drive/MyDrive/datasets/runs/detect/train/results.png',width=600)

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/datasets

Image(filename=f'/content/drive/MyDrive/datasets/runs/detect/train/val_batch0_pred.jpg',width=600)

"""Validate Custom Model"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/datasets



from ultralytics import YOLO

#model = YOLO("yolov8m.yaml")


model = YOLO('/content/drive/MyDrive/datasets/runs/detect/train/weights/best.pt')


results = model.val(data='/content/drive/MyDrive/datasets/microplastic_minor/data.yaml')

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/datasets

!yolo task=detect mode=predict model='/content/drive/MyDrive/datasets/runs/detect/train/weights/best.pt' conf=0.25 source='/content/drive/MyDrive/datasets/microplastic_minor/test/images'

Image('/content/drive/MyDrive/datasets/runs/detect/predict/310_jpg.rf.169822e09d598e3797d4e3a486283c49.jpg')

from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load an official model
model = YOLO("/content/drive/MyDrive/datasets/runs/detect/predict/train/weights/best.pt")  # load a custom trained model

# Export the model
model.export(format="onnx")

from google.colab import files

# Change 'best.pt' to your model file name if it's different
files.download('/content/drive/MyDrive/datasets/runs/detect/predict/train/weights/best.pt')

!yolo val model=/content/drive/MyDrive/datasets/runs/detect/predict/train/weights/best.pt data=/content/drive/MyDrive/datasets/microplastic_minor/data.yaml
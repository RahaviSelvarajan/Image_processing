import matplotlib.pyplot as plt
import cv2
import math
import numpy as np
import shutil
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras_ocr

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2) / 2)
    y_mid = int((y1 + y2) / 2)
    return (x_mid, y_mid)


pipeline = keras_ocr.pipeline.Pipeline()


def inpaint_text(img_path, pipeline):
    # read image
    img = keras_ocr.tools.read(img_path)
    # generate (word, box) tuples
    prediction_groups = pipeline.recognize([img])
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1]
        x2, y2 = box[1][2]
        x3, y3 = box[1][3]

        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)

        thickness = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255, thickness)
        img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)

    return (img)

path = 'canex/images/train'

num = 0
for image in os.listdir(path):
    num += 1
    if num % 100 == 0:
        print("Processed Images", num)
    image = os.path.join(path, image)
    processed_image = inpaint_text(image, pipeline)
    cv2.imwrite(image, processed_image)


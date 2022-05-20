from xxlimited import new
import cv2
import os
import pymongo
import torch
import numpy as np
import pandas as pd
import itertools
from skimage import morphology,io
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import colors

# Check if list has consecutive elements
def checkConsecutive(l):
    try:
        return sorted(l) == list(range(min(l), max(l)+1))
    except ValueError:
        return False

# Find the overlapping elements between two lists
def overlapping(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def checkInFirst(a, b):
     # getting count
    count_a = Counter(a)
    count_b = Counter(b)
 
    # checking if element exists in second list
    for key in count_b:
        if key not in count_a:
            return False
        if count_b[key] > count_b[key]:
            return False
    return True

def shortest_list(list):
    list_len = [len(i) for i in list]
    return list_len.index(min(list_len))

# Initialize a dictionary to store new clock positions
annotations = {}
annotations['old'] = {}
annotations['new'] = {}
annotations['class'] = {}
clock = [int(i)+1 for i in range(12)]

# Mongodb
client = pymongo.MongoClient()
mydb = client["Mongo"]
mycol = mydb["top10Categorical"]

# Iterate through all the heatmaps
for num, path in enumerate(os.listdir('/Users/rahaviselvarajan/JACOBB-MITACS internship/Can_Exp')):
    # Get the current annotations
    annot_dict = mycol.find_one({"filename": path})

    # Get the clock positions based on current annotations
    if annot_dict:
        clock_to = annot_dict['clock_to']
        clock_from = annot_dict['clock_at_from']
        annotations['class'][path] = annot_dict['classes']

        if clock_from == None and clock_to == None:
            annotations['old'][path] = None
            annotations['new'][path] = None
            continue
        
        if clock_to == None:
            reference_clock = [clock_from]
            annotations['old'][path] = reference_clock

        elif clock_from == clock_to:
            annotations['old'][path] = list(range(1, 13))
            annotations['new'][path] = list(range(1, 13))
            continue
        
        elif clock_from > clock_to:
            arr_tmp1 = clock[0:clock_to]
            arr_tmp2 = clock[clock_from-1:12]
            reference_clock = arr_tmp1 + arr_tmp2
            annotations['old'][path] = reference_clock
        
        else:
            reference_clock = clock[clock_from - 1:clock_to]
            annotations['old'][path] = reference_clock

        # Heat map
        image_path = os.path.join('/Users/rahaviselvarajan/JACOBB-MITACS internship/Can_Exp', path)
        image = cv2.imread(image_path, 0)

        # Otsu's thresholding after Gaussian filtering
        blurred_image = cv2.GaussianBlur(image, (5,5), 0)
        ret,threshold = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        stencil = np.zeros(threshold.shape).astype(threshold.dtype)
        """cv2.imshow("Thresholded Image", threshold)
        cv2.waitKey(0) """

        # Finding contours
        contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        new_annotations = []

        # Calculate the IoU scores for all masks
        for c, contour in enumerate(contours):
            l = []
            # Area of contour
            area = cv2.contourArea(contour)
            # If the contour area is significant
            if area > 4000:
                
                cv2.fillPoly(stencil, [contour], 255)
                defect = cv2.bitwise_and(threshold, stencil)
                
                for i, m in enumerate(os.listdir('masks')):
                    mask = cv2.imread(os.path.join('masks', m), 0)
                    intersection = np.logical_and(mask, defect)
                    union = np.logical_or(mask, defect)
                    iou_score = np.sum(intersection) / np.sum(union)
                    if iou_score > 0.1:
                        l.append(int(m[-7:-5]))
                new_annotations.append(l)

        new_annotations.sort(key=len)

        if len(new_annotations) > 1:
            flag = True
            index = []
            i = 0
            while flag:
                for j in range(i+1, len(new_annotations)):
                    if set(new_annotations[i]).issubset(set(new_annotations[j])):
                        index.append(i)
                i += 1
                if i == len(new_annotations) - 1:
                    flag = False

            for index in sorted(index, reverse=True):
                del new_annotations[index]

        if len(new_annotations) > 1: 
            index = []   
            for k in range(len(new_annotations)):
                overlap = overlapping(reference_clock, new_annotations[k])
                if len(overlap) == 0:
                    index.append(k)
            
            for index in sorted(index, reverse=True):
                del new_annotations[index]

        if len(reference_clock) == 1 and len(new_annotations) == 1:
            e = []
            new_annot = new_annotations[0]
            if reference_clock == ([12] or [1]):
                padded_clock = [11, 12, 1, 2]
            else:
                padded_clock = [reference_clock[0]-1, reference_clock[0], reference_clock[0]+1]   
            overlap = overlapping(padded_clock, new_annot)    
            if len(new_annot) > 3 or len(overlap) < 2:
                new_annot = reference_clock
            for ele in new_annot:
                if ele not in padded_clock:
                    e.append(ele)
            for element in e:
                new_annot.remove(element)

            new_annotations = new_annot
        
        if len(reference_clock) > 1:
            length = []
            for m in range(len(new_annotations)):
                overlap = overlapping(reference_clock, new_annotations[m])
                length.append(len(overlap))
            if len(length) > 1:
                ind = length.index(max(length))
                new_annotations = new_annotations[ind]
            elif len(overlap) < 2:
                new_annotations = reference_clock
        
        if any(isinstance(el, list) for el in new_annotations):
            new_annotations = list(itertools.chain(*new_annotations))
        
        if len(reference_clock) - len(new_annotations) > 2:
            new_annotations = reference_clock

        if not checkConsecutive(new_annotations):
            new_annotations = reference_clock
        
        annotations['new'][path] = new_annotations

df = pd.DataFrame.from_dict(annotations)
df.to_excel('new_annotations.xlsx')

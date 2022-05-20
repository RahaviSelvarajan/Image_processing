import cv2
import os
import pandas as pd
import numpy as np

df = pd.read_excel('new_annotations.xlsx')

heatmap_dir = '/Users/rahaviselvarajan/JACOBB-MITACS internship/Can_Exp'
image_dir = '/Users/rahaviselvarajan/JACOBB-MITACS internship/Dataset/images'

classes = ['TS', 'IS', 'FC', 'TBI', 'SRI', 'DSF', 'TB', 'TF', 'DAE']

for ind in df.index:
    old = str(df['old'][ind]).strip('][').split(', ')
    new = str(df['new'][ind]).strip('][').split(', ')
    new.sort()
    try:
        if not old == new and len(new) < 4:
            print(df['Unnamed: 0'][ind])
            image_path = os.path.join(image_dir, df['Unnamed: 0'][ind])
            heatmap_path = os.path.join(heatmap_dir, df['Unnamed: 0'][ind])

            image = cv2.imread(image_path)
            heatmap = cv2.imread(heatmap_path, 0)

            w, h, c = image.shape
            print(w, h)
            
            blurred_image = cv2.GaussianBlur(heatmap, (5,5), 0)
            ret,threshold = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            threshold = cv2.resize(threshold, (320, 240))

            stencil = np.zeros(threshold.shape).astype(threshold.dtype)
            final_mask = stencil.copy()

            for item in new:
                if int(item) in [10, 11, 12]:
                    mask_path = 'clock_' + str(item) + '.jpeg'
                else:
                    mask_path = 'clock_0' + str(item) + '.jpeg'
                mask = cv2.imread(os.path.join('masks', mask_path), 0) 
                mask = cv2.resize(mask, (320, 240))
                final_mask = cv2.bitwise_or(final_mask, mask) 
                final_mask = cv2.dilate(final_mask, None, iterations=1)

            defect = cv2.bitwise_and(threshold, final_mask)
            defect = cv2.dilate(defect, None, iterations=1)

            contours, hierarchy = cv2.findContours(defect, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            boxes = []
            for c, contour in enumerate(contours):
                rect = cv2.boundingRect(contour)
                x,y,w,h = rect
                boxes.append([x,y, x+w,y+h])

            boxes = np.asarray(boxes)
            left, top = np.min(boxes, axis=0)[:2]
            right, bottom = np.max(boxes, axis=0)[2:]
            cv2.rectangle(image, (left,top), (right,bottom), (0,255,0), 2)

            txt_path = os.path.join('labels', os.path.splitext(df['Unnamed: 0'][ind])[0]+ '.txt')

            c = [classes.index(str(df['class'][ind])), left/w, top/h, abs(right-left)/w, abs(bottom-top)/h]

            with open(txt_path, 'a+') as file:
                for value in c:
                    file.write("%s " % value)
                file.write("\n")
                file.close()
            
            cv2.imshow('Image with bb', image)
            cv2.imwrite(os.path.join('image_bb', df['Unnamed: 0'][ind]), image)
            cv2.imshow('Heatmap', threshold)
            cv2.waitKey(100)
    except:
        pass
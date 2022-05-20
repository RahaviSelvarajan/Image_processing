from PIL import Image, ImageStat
import numpy as np
import cv2
import math
import os

def brightness( im ):
    im = Image.fromarray(im)
    stat = ImageStat.Stat(im)
    r,g,b = stat.mean
    return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))

# video_path = 'user_991_video_diary_969122.mp4'

# cap = cv2.VideoCapture(video_path)
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))

# bright = []

# out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if ret:
#         bright.append(brightness(frame))
#         cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
#         out.write(frame)
#         cv2.imshow('frame',frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break

# print(sum(bright)/len(bright))
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()

bright = list()
for i, video_file in enumerate(os.listdir('video_diaries')):
    print(i, video_file)
    cap = cv2.VideoCapture(os.path.join('video_diaries', video_file))
    b = list()
    while(True):
        ret, frame = cap.read()
        if ret: 
            b.append(brightness(frame))
        else:
            bright.append(sum(b)/len(b))
            break
    if i%50 == 0:
        print('Processed {0} Videos'.format(i))

print("#videos with brightness less than 50: ", sum(i < 50 for i in bright))
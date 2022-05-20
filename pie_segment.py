import math
from PIL import Image, ImageDraw
import cv2
import numpy as np
def segment_color(i_color, n_colors):
    r = int((192 - 64) / (n_colors - 1) * i_color + 64)
    g = int((224 - 128) / (n_colors - 1) * i_color + 128)
    b = 255
    return (r, g, b)


# Load image; generate ImageDraw
im=Image.open('RS-12@RS-13_RS-12_201909261153_04.jpg').convert('RGB')
h,w =im.size
im = Image.new('RGB', (h+100, w+100))
#size=heat_map.shape
draw = ImageDraw.Draw(im)

# Number of pie segments (must be an even number)
n = 12

# Replace (all-white) edge with defined edge color
edge_color = (255, 128, 0)
pixels = im.load()

for i in range(50,im.height-50):
    pixels[50, i] = edge_color
    pixels[h+50, i] = edge_color
    pixels[i, 50] = edge_color
    pixels[i, h+50] = edge_color

for y in range(im.height):
    for x in range(im.width):
        if pixels[x, y] == (255, 255, 255):
            pixels[x, y] = edge_color

# Draw lines with defined line color
line_color = (0, 255, 0)
d = min(im.width, im.height) - 10
center = (int(im.width/2), int(im.height)/2)
for i in range(int(n/2)):
    angle = 360 / n * i
    x1 = math.cos(angle/180*math.pi) * d/2 + center[0]
    y1 = math.sin(angle/180*math.pi) * d/2 + center[1]
    x2 = math.cos((180+angle)/180*math.pi) * d/2 + center[0]
    y2 = math.sin((180+angle)/180*math.pi) * d/2 + center[1]
    draw.line([(x1, y1), (x2, y2)], line_color)

# Fill pie segments with defined segment colors
for i in range(n):
    angle = 360 / n * i + 360 / n / 2
    x = math.cos(angle/180*math.pi) * 20 + center[0]
    y = math.sin(angle/180*math.pi) * 20 + center[1]
    ImageDraw.floodfill(im, (x, y), segment_color(i, n))
cropped = im.crop((50, 50, h+50, w+50))
pixels = cropped.load()
for c in range(n):
    mask = np.zeros((h,w))
    for y in range(cropped.height):
        for x in range(cropped.width):
            if pixels[x, y] == segment_color(c, n):
                mask[x, y] = 1
    im = Image.fromarray(mask,'L')
    cv2.imwrite('mask_{}.jpeg'.format(c),mask*255)


cropped.save(str(n) + '_pie.png')
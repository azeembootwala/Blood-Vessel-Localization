# This file would just scroll the images/ results stored in the directory
from os import listdir
from os.path import isfile, join
import cv2
folder = "/home/esaote/object_detection_AB/results/good/prediction/"
images = [f for f in listdir(folder) if isfile(join(folder, f))]
#cv2.namedWindow('image',cv2.WINDOW_NORMAL)
for image in images:
    if ".png" in image:
        img = cv2.imread(folder+"/"+image)
        cv2.imshow('image',img)
        k = cv2.waitKey(1500) & 0xFF
        if k == 27: # exits after pressing ESC
            exit()
cv2.destroyAllWindows()

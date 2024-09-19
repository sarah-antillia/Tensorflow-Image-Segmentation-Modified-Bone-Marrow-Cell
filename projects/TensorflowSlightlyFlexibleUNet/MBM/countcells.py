import os
import cv2
import glob
import shutil
import traceback

class CellCounter:

  def __init__(self):
    pass

  def count(self, images_dir, output_dir):
    image_files = glob.glob(images_dir + "/*.jpg")
    for image_file in image_files:
      print("--- image_file {}".format(image_file))
      image = cv2.imread(image_file)
      h, w, c = image.shape
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      gray = cv2.resize(gray, (w*4, h*4))

      
      ret, gray = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

      contours, hierarchy = cv2.findContours(
         gray, cv2.RETR_LIST, 
         #cv2.RETR_EXTERNAL, 
         #cv2.CHAIN_APPROX_NONE)
         #cv2.CHAIN_APPROX_TC89_KCOS)
         cv2.CHAIN_APPROX_TC89_L1)
         #  #cv2.CHAIN_APPROX_NONE) #cv2.CHAIN_APPROX_SIMPLE)
      #print("--- cell count {}".format(contours))
      print("--- count {}".format(len(contours)))
      for i, cnt in enumerate(contours):
         print("--- {}  shape {}".format(i, cnt.shape))
        


if __name__ == "__main__":
  try:
    images_dir = "./mini_test/images"
    output_dir = "./cell_count"
    counter = CellCounter()
    counter.count(images_dir, output_dir)

  except:
    traceback.print_exc()

# Copyright 2024 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# sharpen.py
# 2024/09/19

import os
import glob
import cv2
import numpy as np
import shutil
import traceback

def sharpen(images_dir, sharpen_k, output_dir):
  image_files = glob.glob(images_dir + "/*.jpg")
  for image_file in image_files:
    basename = os.path.basename(image_file)
    image = cv2.imread(image_file)
    if sharpen_k > 0:
      k = sharpen_k
      kernel = np.array([[-k, -k, -k], 
                       [-k, 1+8*k, -k], 
                       [-k, -k, -k]])
    sharpened = cv2.filter2D(image, ddepth=-1, kernel=kernel)
    output_file = os.path.join(output_dir, basename)

    cv2.imwrite(output_file, sharpened)
    print("--- Saved {}".format(output_file))

if __name__ == "__main__":
  try:
     images_dir = "./mini_test/images"
     output_dir = "./sharpened"
     if os.path.exists(output_dir):
       shutil.rmtree(output_dir)
     os.makedirs(output_dir)
     sharpen_k = 1
     sharpen(images_dir, sharpen_k, output_dir)

  except:
    traceback.print_exc()
 

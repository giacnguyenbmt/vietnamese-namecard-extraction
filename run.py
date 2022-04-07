import os
from pathlib import Path

import cv2
import numpy as np

from libs.mmocr.mmocr.utils.ocr import MMOCR
from src.textdetection import prepare
from src.documentdectetion import hough_based_detection as hough
from utils import transform
from utils import fpt_group
from utils import easyocr_group

## VARIABLE
img_path = os.path.join(os.getcwd(), 'data/documentdectetion/ve_1.jpg')
det='PANet_IC15'
text_det_config_dir = os.path.join(str(Path.cwd()), 'libs/mmocr/configs/')

#======================================================#
## PREPARE
det_ckpt = prepare.download_model(det=det)
# Load models into memory
ocr = MMOCR(det=det,
            det_ckpt=det_ckpt,
            recog=None,
            config_dir=text_det_config_dir)



#======================================================#
## MAIN
# Step 1: Document Detection
img = cv2.imread(img_path)

doc_bbox = hough.single_detector(img)
crop_img = transform.four_point_transform(img, doc_bbox)
cv2.imwrite('output/d_crop.jpg', crop_img)


# Step 2: Text Detetion Inference
det_results = ocr.readtext('output/d_crop.jpg', output='output/det_out.jpg', export='output/')
m_pD = np.array(det_results[0]['boundary_result']).astype(int)[:,:8]

# Grouping by FPT
# group_label, order = fpt_group.group_by_block(crop_img, boundary_result)


# Step 3: Text Recognition Inference
...


# Step 4: Post-processing
# Grouping by EasyOCR
# require: "m_pD" is the result of Text Detetion. "Text" is the list of recognized words.

# raw_result = []
# for i in range(len(m_pD)):
#     bbox = m_pD[i].reshape((-1, 2))
#     raw_result.append((bbox.tolist(), Text[i]))
# easyocr_group.get_paragraph(raw_result)
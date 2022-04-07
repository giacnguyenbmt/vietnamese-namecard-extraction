import os
from pathlib import Path

import gdown

textdet_models = {
    'DB_r18': {
        'config':
        'dbnet/dbnet_r18_fpnc_1200e_icdar2015.py',
        'ckpt':
        'dbnet_r18_fpnc_sbn_1200e_vbc583_20221215.pth',
        'id': 
        '1bnr4eGev9MMuyHmieZl6RzfQDxm4uaN7'
    },
    'DB_r50': {
        'config':
        'dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py',
        'ckpt':
        'dbnet_r50_fpnc_sbn_1200e_vbc583_20221215.pth',
        'id': 
        '1ZVAgiuCCCk---hq1zhut4JVZPtyySVVR'
    },
    'MaskRCNN_IC15': {
        'config':
        'maskrcnn/mask_rcnn_r50_fpn_160e_icdar2015.py',
        'ckpt':
        'mask_rcnn_r50_fpn_160e_vbc583_20221215.pth',
        'id': 
        '1OORE_McZJCucjDFN68AzXacC3YdfuDYP'
    },
    'PANet_IC15': {
        'config':
        'panet/panet_r18_fpem_ffm_600e_icdar2015.py',
        'ckpt':
        'panet_r18_fpem_ffm_sbn_600e_vbc583_20221215.pth',
        'id': 
        '1_J3jWRoPvh-em0HGR8kGCzRx-TkQP_Px'
    },
    'PS_IC15': {
        'config':
        'psenet/psenet_r50_fpnf_600e_icdar2015.py',
        'ckpt':
        'psenet_r50_fpnf_600e_vbc583_201215.pth',
        'id': 
        '15Swr8x6hun8CTgFvIpAVyy9OeXfgraFA'
    }
}

def download_model(det='PANet_IC15', det_ckpt_dir='ckpt/'):
    det_ckpt = os.path.join(str(Path.cwd()), det_ckpt_dir, textdet_models[det]['ckpt'])
    if os.path.isdir(det_ckpt_dir) is False:
        os.makedirs(det_ckpt_dir)
    if os.path.isfile(det_ckpt) is False:
        url = 'https://drive.google.com/uc?id={}'.format(textdet_models[det]['id'])
        output = det_ckpt
        gdown.download(url, output, quiet=False)
    return det_ckpt
    




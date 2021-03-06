
import copy

import numpy as np
from .custom import CustomDataset
import mmcv
import os
import platform
import cv2
import logging
from google.cloud import storage

class OIDSegDataset(CustomDataset):
    # 275 leave classes
    CLASSES = (
        '/m/0242l', '/m/03120', '/m/01j51', '/m/029b3', '/m/02zt3', '/m/0kmg4', '/m/0174k2', '/m/01k6s3', '/m/029bxz',
        '/m/03s_tn', '/m/0fx9l', '/m/02f9f_', '/m/02jz0l', '/m/09g1w', '/m/01lsmm', '/m/025dyy', '/m/02d9qx',
        '/m/03m3vtv',
        '/m/05gqfk', '/m/09gtd', '/m/0frqm', '/m/0k1tl', '/m/02w3r3', '/m/034c16', '/m/01_5g', '/m/02d1br', '/m/03v5tg',
        '/m/04ctx', '/m/0cmx8', '/m/01fh4r', '/m/02jvh9', '/m/02p5f1q', '/m/03q5c7', '/m/04dr76w', '/m/04kkgm',
        '/m/054fyh',
        '/m/058qzx', '/m/08hvt4', '/m/099ssp', '/m/04v6l4', '/m/084rd', '/m/02tsc9', '/m/03y6mg', '/m/0h8ntjv',
        '/m/0bt_c3',
        '/m/03m3pdh', '/m/0703r8', '/m/026qbn5', '/m/047j0r', '/m/05kyg_', '/m/0h8n6f9', '/m/046dlr', '/m/06_72j',
        '/m/025nd',
        '/m/02s195', '/m/04yqq2', '/m/01yx86', '/m/06z37_', '/m/0c06p', '/m/0fm3zh', '/m/0162_1', '/m/015qff',
        '/m/02pv19',
        '/m/01pns0', '/m/04h7h', '/m/079cl', '/m/04yx4', '/m/03bt1vf', '/m/01bl7v', '/m/05r655', '/m/01b9xk',
        '/m/01dwsz',
        '/m/01dwwc', '/m/01j3zr', '/m/01f91_', '/m/021mn', '/m/01tcjp', '/m/0fszt', '/m/02g30s', '/m/014j1m',
        '/m/0388q',
        '/m/043nyj', '/m/061_f', '/m/07fbm7', '/m/07j87', '/m/09k_b', '/m/09qck', '/m/0cyhj_', '/m/0dj6p', '/m/0fldg',
        '/m/0hqkz', '/m/0jwn_', '/m/0kpqd', '/m/01fb_0', '/m/09728', '/m/0jy4k', '/m/015wgc', '/m/02zvsm', '/m/052sf',
        '/m/0663v', '/m/0_cp5', '/m/015x4r', '/m/015x5n', '/m/05vtc', '/m/0cjs7', '/m/05zsy', '/m/027pcv', '/m/0fbw6',
        '/m/0fj52s', '/m/0hkxq', '/m/0jg57', '/m/02cvgx', '/m/0cdn1', '/m/06pcq', '/m/06m11', '/m/0ftb8', '/m/012n7d',
        '/m/01bjv', '/m/01x3jk', '/m/04_sv', '/m/076bq', '/m/07cmd', '/m/07jdr', '/m/07r04', '/m/01lcw4', '/m/0h2r6',
        '/m/0pg52', '/m/01btn', '/m/0ph39', '/m/01xs3r', '/m/0cmf2', '/m/09rvcxw', '/m/01bfm9', '/m/01d40f',
        '/m/01gkx_',
        '/m/01n4qj', '/m/01xyhv', '/m/025rp__', '/m/02fq_6', '/m/02jfl0', '/m/02wbtzl', '/m/02h19r', '/m/01cmb2',
        '/m/03grzl',
        '/m/0176mf', '/m/01nq26', '/m/01rkbr', '/m/0gjkl', '/m/04tn4x', '/m/0fly7', '/m/02p3w7d', '/m/01b638',
        '/m/06k2mb',
        '/m/01940j', '/m/01s55n', '/m/0584n8', '/m/080hkjn', '/m/01dy8n', '/m/01f8m5', '/m/05n4y', '/m/05z6w',
        '/m/06j2d',
        '/m/09b5t', '/m/09csl', '/m/09d5_', '/m/09ddx', '/m/0ccs93', '/m/0dbvp', '/m/0dftk', '/m/0f6wt', '/m/0gv1x',
        '/m/0h23m', '/m/0jly1', '/m/01h8tj', '/m/01h44', '/m/01dxs', '/m/0633h', '/m/01yrx', '/m/0306r', '/m/0449p',
        '/m/04g2r', '/m/07dm6', '/m/096mb', '/m/0bt9lr', '/m/0c29q', '/m/0cd4d', '/m/0cn6p', '/m/0dq75', '/m/01x_v',
        '/m/01xq0k1', '/m/03bk1', '/m/03d443', '/m/03fwl', '/m/03k3r', '/m/03qrc', '/m/04c0y', '/m/04rmv', '/m/068zj',
        '/m/06mf6', '/m/071qp', '/m/07bgp', '/m/0898b', '/m/08pbxl', '/m/0bwd_0j', '/m/0cnyhnx', '/m/0dbzx', '/m/02hj4',
        '/m/084zz', '/m/0gd36', '/m/02l8p9', '/m/0pcr', '/m/04m9y', '/m/078jl', '/m/011k07', '/m/0120dh', '/m/09f_2',
        '/m/09ld4', '/m/03fj2', '/m/0by6g', '/m/01xqw', '/m/0342h', '/m/03q5t', '/m/05r5c', '/m/06ncr', '/m/0l14j_',
        '/m/01226z', '/m/02rgn06', '/m/05ctyq', '/m/0wdt60w', '/m/019w40', '/m/03g8mr', '/m/0420v5', '/m/06_fw',
        '/m/04h8sr',
        '/m/0h8my_4', '/m/05_5p_0', '/m/04p0qw', '/m/02zn6n', '/m/0bjyj5', '/m/0d20w4', '/m/01bms0', '/m/01j5ks',
        '/m/01kb5b',
        '/m/05bm6', '/m/07dd4', '/m/0dv5r', '/m/0hdln', '/m/0lt4_', '/m/02gzp', '/m/0gxl3', '/m/06y5r', '/m/04ylt',
        '/m/01c648', '/m/01m2v', '/m/01m4t', '/m/020lf', '/m/03bbps', '/m/03jbxj', '/m/050k8', '/m/0h8lkj8',
        '/m/0bh9flk',
        '/m/01599', '/m/024g6', '/m/02vqfm', '/m/01z1kdw', '/m/07clx', '/m/081qc', '/m/01bqk0', '/m/03c7gz',
        '/m/016m2d',
        '/m/0283dt1', '/m/039xj_', '/m/01jfm_', '/m/083wq', '/m/0dkzw'
    )

    def load_annotations(self, ann_file):
        if 'val' in os.path.basename(ann_file): self.split = 'val'
        elif 'test' in os.path.basename(ann_file): self.split = 'OD_test'      
        else: self.split = 'train'
        return mmcv.load(ann_file)

    def get_ann_info(self, idx):
        logger = logging.getLogger()

        # local path
        if platform.system() == 'Darwin':
            data_dir = '/Users/bo_liu/Documents/open-images/data/'
        else:
            data_dir = '/media/bo/Elements/open-images/data/'

        ann = copy.deepcopy(self.img_infos[idx]['ann'])

        gt_masks = []
        for i, mask_path in enumerate(ann['MaskPath']):
            # for colab/gcp
            if 'gs://oid2019' in self.img_prefix:                
                blob = self.bucket.get_blob('data/train_masks/'+ann['MaskPath'][i])
                if blob is None:
                    logger.info("------ ??? -------")
                  
                msk = cv2.imdecode(np.asarray(bytearray(blob.download_as_string()), dtype=np.uint8),cv2.IMREAD_UNCHANGED)
            else:
                msk = mmcv.imread(data_dir + self.split + '_masks/'+ann['MaskPath'][i],'unchanged')

            msk = (msk > 0).astype('uint8')

            # most images and their masks don't have same size
            msk = mmcv.imresize(msk, (self.img_infos[idx]['width'], self.img_infos[idx]['height']))
            gt_masks.append(msk)
        ann['masks'] = gt_masks
        return ann

       
class OIDSegParentDataset(OIDSegDataset):
  CLASSES = ('/m/0138tl',
             '/m/02crq1',
             '/m/01x3z',
             '/m/06msq',
             '/m/01mqdt',
             '/m/01g317',
             '/m/0dv77',
             '/m/0l515',
             '/m/0c9ph5',
             '/m/0k4j',
             '/m/0k5j',
             '/m/02dl1y',
             '/m/02wv6h6',
             '/m/0174n1',
             '/m/07mhn',
             '/m/0hf58v5',
             '/m/015p6',
             '/m/01dws',
             '/m/09dzg',
             '/m/0ch_cf',
             '/m/018xm',
             '/m/0dv9c',
             '/m/0271t')


import os.path as osp

import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

from .transforms import (ImageTransform, BboxTransform, MaskTransform,
                         SegMapTransform, Numpy2Tensor)
from .utils import to_tensor, random_scale
from .extra_aug import ExtraAugmentation
import cv2
from google.cloud import storage
import logging


class CustomDataset(Dataset):

    CLASSES = None

    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 multiscale_mode='value',
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 with_mask=True,
                 with_crowd=True,
                 with_label=True,
                 with_semantic_seg=False,
                 seg_prefix=None,
                 seg_scale_factor=1,
                 extra_aug=None,
                 resize_keep_ratio=True,
                 test_mode=False):
        # prefix of images path
        self.img_prefix = img_prefix

        self.split = None

        # load annotations (and proposals)
        self.img_infos = self.load_annotations(ann_file)
        if proposal_file is not None:
            self.proposals = self.load_proposals(proposal_file)
        else:
            self.proposals = None
        # filter images with no annotation during training
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.img_infos = [self.img_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]

        # (long_edge, short_edge) or [(long1, short1), (long2, short2), ...]
        self.img_scales = img_scale if isinstance(img_scale,
                                                  list) else [img_scale]
        assert mmcv.is_list_of(self.img_scales, tuple)
        # normalization configs
        self.img_norm_cfg = img_norm_cfg

        # multi-scale mode (only applicable for multi-scale training)
        self.multiscale_mode = multiscale_mode
        assert multiscale_mode in ['value', 'range']

        # max proposals per image
        self.num_max_proposals = num_max_proposals
        # flip ratio
        self.flip_ratio = flip_ratio
        assert flip_ratio >= 0 and flip_ratio <= 1
        # padding border to ensure the image size can be divided by
        # size_divisor (used for FPN)
        self.size_divisor = size_divisor

        # with mask or not (reserved field, takes no effect)
        self.with_mask = with_mask
        # some datasets provide bbox annotations as ignore/crowd/difficult,
        # if `with_crowd` is True, then these info is returned.
        self.with_crowd = with_crowd
        # with label is False for RPN
        self.with_label = with_label
        # with semantic segmentation (stuff) annotation or not
        self.with_seg = with_semantic_seg
        # prefix of semantic segmentation map path
        self.seg_prefix = seg_prefix
        # rescale factor for segmentation maps
        self.seg_scale_factor = seg_scale_factor
        # in test mode or not
        self.test_mode = test_mode

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # transforms
        self.img_transform = ImageTransform(
            size_divisor=self.size_divisor, **self.img_norm_cfg)
        self.bbox_transform = BboxTransform()
        self.mask_transform = MaskTransform()
        self.seg_transform = SegMapTransform(self.size_divisor)
        self.numpy2tensor = Numpy2Tensor()

        # if use extra augmentation
        if extra_aug is not None:
            self.extra_aug = ExtraAugmentation(**extra_aug)
        else:
            self.extra_aug = None

        # image rescale if keep ratio
        self.resize_keep_ratio = resize_keep_ratio

    def __len__(self):
        return len(self.img_infos)

    def load_annotations(self, ann_file):
        return mmcv.load(ann_file)

    def load_proposals(self, proposal_file):
        return mmcv.load(proposal_file)

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann']

    def _filter_imgs(self, min_size=32):
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):

        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = None
            while data is None:
              try:
                data = self.prepare_train_img(idx)
              except:
                  logger = logging.getLogger()
                  logger.info(f"------ self.prepare_train_img(idx) failed, sleep 3 seconds, idx = {idx}")
                  break
                  import time                    
                  time.sleep(3)                 
              
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx):
        import gc
        gc.collect()
        img_info = self.img_infos[idx]
        logger = logging.getLogger()
        if idx%1000==5: 
            logger.info(f"idx={idx}, img_info['filename']={img_info['filename']}")
            print(f"idx={idx}, img_info['filename']={img_info['filename']}")
        # load image

        # for colab
        if 'gs://oid2019' in self.img_prefix:

            self.client = storage.Client.from_service_account_json(
                '/home/jupyter/project-owner-xxxxxxxxxxx.json')
                
            # workaround for connection error
            blob = None
            while blob is None:
                try:
                    self.bucket = self.client.get_bucket('oid2019')
                    blob = self.bucket.get_blob(osp.join(self.img_prefix.replace('gs://oid2019/', ''), img_info['filename']))
                    if blob is None:
                      logger.info(f"self.bucket.get_blob(osp.join(self.img_prefix.replace('gs://oid2019/', ''), img_info['filename'])) failed for {osp.join(self.img_prefix.replace('gs://oid2019/', ''), img_info['filename'])}")
                      blob = self.bucket.get_blob(osp.join(self.img_prefix.replace('gs://oid2019/data/train', 'data/test'), img_info['filename']))
                      if blob is None:
                        logger.info(f"self.bucket.get_blob(osp.join(self.img_prefix.replace('gs://oid2019/data/train', 'data/test'), img_info['filename'])) failed")
                        blob = self.bucket.get_blob(osp.join(self.img_prefix.replace('gs://oid2019/data/train', 'data/val'), img_info['filename']))
                        if blob is None:
                          logger.info(f"self.bucket.get_blob(osp.join(self.img_prefix.replace('gs://oid2019/data/train', 'data/val'), img_info['filename'])) failed")
                except:               
                    logger = logging.getLogger()
                    logger.info(f"------ self.client.get_bucket('oid2019') failed, sleep 2 seconds")
                    import time                    
                    time.sleep(2)                     
                     
            img = cv2.imdecode(np.asarray(bytearray(blob.download_as_string()), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            
        else:
            img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        # load proposals if necessary
        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]
            # TODO: Handle empty proposals properly. Currently images with
            # no proposals are just ignored, but they can be used for
            # training in concept.
            if len(proposals) == 0:
                return None
            if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 5:
                scores = proposals[:, 4, None]
                proposals = proposals[:, :4]
            else:
                scores = None
        
        ann = None
        while ann is None:
          try:
            ann = self.get_ann_info(idx)
          except:
            logger = logging.getLogger()
            logger.info(f"------ self.get_ann_info(idx) failed, sleep 2 seconds")
            ann = 0
            import time                      
            time.sleep(2)        
            
        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']
        if self.with_crowd:
            gt_bboxes_ignore = ann['bboxes_ignore']

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0:
            return None

        # extra augmentation
        if self.extra_aug is not None:
            img = self.extra_aug(img)
#            img, gt_bboxes, gt_labels = self.extra_aug(img, gt_bboxes,
#                                                       gt_labels)

        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        # randomly sample a scale
        img_scale = random_scale(self.img_scales, self.multiscale_mode)

        #        logger.info("img_info['filename'] = " +str(img_info['filename']))

        try:
            img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        except:
            logger = logging.getLogger()
            logger.info("Error! img_info['filename'] = " + str(img_info['filename']))
            raise

        img = img.copy()
        if self.with_seg:
            gt_seg = mmcv.imread(
                osp.join(self.seg_prefix, img_info['file_name'].replace(
                    'jpg', 'png')),
                flag='unchanged')
            gt_seg = self.seg_transform(gt_seg.squeeze(), img_scale, flip)
            gt_seg = mmcv.imrescale(
                gt_seg, self.seg_scale_factor, interpolation='nearest')
            gt_seg = gt_seg[None, ...]
        if self.proposals is not None:
            proposals = self.bbox_transform(proposals, img_shape, scale_factor,
                                            flip)
            proposals = np.hstack(
                [proposals, scores]) if scores is not None else proposals
        gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
                                        flip)
        if self.with_crowd:
            gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                                   scale_factor, flip)
        if self.with_mask:
            logger.info(str(ann['masks']) + str(pad_shape) + str(scale_factor) + str(flip))
            gt_masks = self.mask_transform(ann['masks'], pad_shape, scale_factor, flip)

        ori_shape = (img_info['height'], img_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip)

        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_bboxes=DC(to_tensor(gt_bboxes)))
        if self.proposals is not None:
            data['proposals'] = DC(to_tensor(proposals))
        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
        if self.with_crowd:
            data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
        if self.with_mask:
            data['gt_masks'] = DC(gt_masks, cpu_only=True)
        if self.with_seg:
            data['gt_semantic_seg'] = DC(to_tensor(gt_seg), stack=True)
        return data

    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]

        # for gcp
        if 'gs://oid2019' in self.img_prefix:

            self.client = storage.Client.from_service_account_json(
                '/home/jupyter/project-owner-xxxxxxxxxxx.json')
                
            # workaround for connection error
            blob = None
            while blob is None:
                try:
                    self.bucket = self.client.get_bucket('oid2019')
                    blob = self.bucket.get_blob(osp.join(self.img_prefix.replace('gs://oid2019/', ''), img_info['filename']))
                except:
                    logger = logging.getLogger()
                    logger.info(f"------ self.client.get_bucket('oid2019') failed, sleep 2 seconds")
                    import time                    
                    time.sleep(2)                     
                     
            img = cv2.imdecode(np.asarray(bytearray(blob.download_as_string()), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        else:                                 
            img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))

        width = img.shape[1]
        height = img.shape[0]
        if self.proposals is not None:
            proposal = self.proposals[idx][:self.num_max_proposals]
            if not (proposal.shape[1] == 4 or proposal.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposal.shape))
        else:
            proposal = None

        def prepare_single(img, scale, flip, proposal=None):
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, flip, keep_ratio=self.resize_keep_ratio)
            _img = to_tensor(_img)
            _img_meta = dict(
                ori_shape=(height, width, 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip)
            if proposal is not None:
                if proposal.shape[1] == 5:
                    score = proposal[:, 4, None]
                    proposal = proposal[:, :4]
                else:
                    score = None
                _proposal = self.bbox_transform(proposal, img_shape,
                                                scale_factor, flip)
                _proposal = np.hstack(
                    [_proposal, score]) if score is not None else _proposal
                _proposal = to_tensor(_proposal)
            else:
                _proposal = None
            return _img, _img_meta, _proposal

        imgs = []
        img_metas = []
        proposals = []
        for scale in self.img_scales:
            _img, _img_meta, _proposal = prepare_single(
                img, scale, False, proposal)
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            proposals.append(_proposal)
            if self.flip_ratio > 0:
                _img, _img_meta, _proposal = prepare_single(
                    img, scale, True, proposal)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)
        data = dict(img=imgs, img_meta=img_metas)
        if self.proposals is not None:
            data['proposals'] = proposals
        return data

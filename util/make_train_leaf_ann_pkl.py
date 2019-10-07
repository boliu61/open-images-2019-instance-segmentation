import mmcv
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import funcy

if __name__ == '__main__':

   data_dir = 'open-images/data/'

   # 275 leaf classes
   classes = mmcv.load(data_dir+'seg_anno/list_of_275_leave_labels_seg.pkl')
   # official gt
   train = pd.read_csv(data_dir+'seg_anno/challenge-2019-train-segmentation-masks.csv').set_index('LabelName')
   train = train.loc[train.index.unique().intersection(classes)].reset_index()
   train.set_index('ImageID',inplace=True)
   train.drop(columns=['PredictedIoU','Clicks'],inplace=True)

   label_ids = {name: i + 1 for i, name in enumerate(classes)}

   # what Dun made for OD (incl all val width, height), to look up for w,d
   train_od_ann=mmcv.load(data_dir+'mmdet_anno/train_bbox.pkl')
   d_val_wh = {x['filename']: (x['width'],x['height']) for x in train_od_ann}

   # multi process
   img_lst = train.index.unique()
   i_lst = list(range(len(img_lst)))
   def process_img(i_sublst):
       d = {}
       for i in i_sublst:
           image_id = img_lst[i]
           ann_df = train.loc[[image_id]]
           w,h = d_val_wh[image_id+'.jpg']
           bboxes = ann_df[['BoxXMin', 'BoxYMin', 'BoxXMax', 'BoxYMax']].values * [w,h,w,h]
           labels = ann_df.LabelName.map(label_ids).values
           annotation = {
               'filename': image_id+'.jpg',
               'width': w,
               'height': h,
               'ann': {
                   'bboxes': bboxes.astype(np.float32),
                   'labels': labels.astype(np.int64),
                   'MaskPath': ann_df.MaskPath.values
               }
           }
           d[i] = annotation
       return d

   chunks = funcy.lchunks(int(len(i_lst)/600), i_lst)
   num_processes = 12
   p = Pool(processes=num_processes)
   d_list = list(tqdm(p.imap(process_img, chunks, chunksize=1), total=len(chunks)))
   p.close()
   p.join()

   d_total = {}
   for d in d_list:
       d_total.update(d)

   train_annotations = [d_total[i] for i in range(649312)]

   mmcv.dump(train_annotations, data_dir+'mmdet_anno/seg_train_275_leave_cls_ann.pkl')
   

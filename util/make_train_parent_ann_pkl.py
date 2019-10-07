import mmcv
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import funcy

if __name__ == '__main__':

   data_dir = 'open-images/data/'

   # 23 level 1 parents
   classes = ['/m/0138tl',
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
             '/m/0271t']
   
   # official gt
   train = pd.read_csv(data_dir+'seg_anno/challenge-2019-train-segmentation-masks.csv',
                       usecols=['BoxID', 'BoxXMax', 'BoxXMin', 'BoxYMax', 'BoxYMin','ImageID', 'LabelName', 'MaskPath'])   
   val = pd.read_csv(data_dir+'seg_anno/challenge-2019-validation-segmentation-masks.csv',
                     usecols=['BoxID', 'BoxXMax', 'BoxXMin', 'BoxYMax', 'BoxYMin','ImageID', 'LabelName', 'MaskPath'])
   test = pd.read_csv(data_dir+'seg_anno/test-annotations-object-segmentation.csv',
                      usecols=['BoxID', 'BoxXMax', 'BoxXMin', 'BoxYMax', 'BoxYMin','ImageID', 'LabelName', 'MaskPath'])
   train = pd.concat([train,val,test])
   
   def expand_parent_labels(train_gt):
        all_keyed_child = mmcv.load(data_dir+'seg_all_keyed_child.pkl') # child -> parents
        all_keyed_child = {k:v for (k,v) in all_keyed_child.items() if len(v)>0}
        all_keyed_child_1 = {k:v for (k,v) in all_keyed_child.items() if len(v)==1}
        all_keyed_child_2 = {k:v for (k,v) in all_keyed_child.items() if len(v)==2}
        all_keyed_child_3 = {k:v for (k,v) in all_keyed_child.items() if len(v)==3}        
        
        df1 = train_gt.set_index('LabelName').loc[all_keyed_child_1.keys()].reset_index().copy()
        df1.LabelName = df1.LabelName.apply(lambda x:all_keyed_child_1[x][0])
        
        df2a = train_gt.set_index('LabelName').loc[all_keyed_child_2.keys()].reset_index().copy()
        df2b = train_gt.set_index('LabelName').loc[all_keyed_child_2.keys()].reset_index().copy()
        df2a.LabelName = df2a.LabelName.apply(lambda x:all_keyed_child_2[x][0])
        df2b.LabelName = df2b.LabelName.apply(lambda x:all_keyed_child_2[x][1])
        
        df3a = train_gt.set_index('LabelName').loc[all_keyed_child_3.keys()].reset_index().copy()
        df3b = train_gt.set_index('LabelName').loc[all_keyed_child_3.keys()].reset_index().copy()
        df3c = train_gt.set_index('LabelName').loc[all_keyed_child_3.keys()].reset_index().copy()
        df3a.LabelName = df3a.LabelName.apply(lambda x:all_keyed_child_3[x][0])
        df3b.LabelName = df3b.LabelName.apply(lambda x:all_keyed_child_3[x][1])
        df3c.LabelName = df3c.LabelName.apply(lambda x:all_keyed_child_3[x][2])
        
        train_gt = pd.concat([train_gt,df1,df2a,df2b,df3a,df3b,df3c])
        return train_gt

   train = expand_parent_labels(train)
   train.set_index('LabelName').loc[classes].shape, train.set_index('LabelName').loc[classes].ImageID.nunique()   
   ((1285502, 8), 607601)
   
   train = train.set_index('LabelName').loc[classes].reset_index()
   train.set_index('ImageID',inplace=True)

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
           if image_id+'.jpg' in d_val_wh:
               w,h = d_val_wh[image_id+'.jpg']
           else:
               continue
               h,w = mmcv.imread('test/'+image_id+'.jpg').shape[:2]              
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

   chunks = funcy.lchunks(int(len(i_lst)/6000), i_lst)
   num_processes = 12
   p = Pool(processes=num_processes)
   d_list = list(tqdm(p.imap(process_img, chunks, chunksize=1), total=len(chunks)))
   p.close()
   p.join()

   d_total = {}
   for d in d_list:
       d_total.update(d)

   train_annotations = [d_total[i] for i in range(607451) if i in d_total]

   mmcv.dump(train_annotations, data_dir+'mmdet_anno/seg_train_parent_23_ann.pkl')
   


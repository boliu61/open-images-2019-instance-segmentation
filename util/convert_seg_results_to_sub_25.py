import mmcv
import os
#from mmdet.apis import init_detector, inference_detector, show_result
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import gc
import argparse
import platform

pd.set_option('display.max_columns', 25)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_path')
    parser.add_argument('--expand')    
    parser.add_argument('--parent')    
    parser.add_argument('--eight_digit', action='store_true')
    args = parser.parse_args()
        
    pkl_path = args.pkl_path
    
    data_dir = '/Users/bo_liu/Documents/open-images/data/'
    repo_dir = '/Users/bo_liu/Documents/open-images/open-images/'
    sub_dir = '/Users/bo_liu/Documents/open-images/subs/'
    num_processes = 12
 
    
    if int(args.parent)==1:
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
          print('---- 23 classes ----')
    else:
        from seg_275_leave_classes import CLASSES
        print('---- 275 classes -----')

    all_keyed_child = mmcv.load(data_dir+'seg_all_keyed_child.pkl')
    
    ################## convert to sub

    ################## from Kaggle

    import base64
    from pycocotools import _mask as coco_mask
    import typing as t
    import zlib

    def encode_binary_mask(mask: np.ndarray) -> t.Text:
     """Converts a binary mask into OID challenge encoding ascii text."""
     # check input mask --
     if mask.dtype != np.bool:
       raise ValueError(
           "encode_binary_mask expects a binary mask, received dtype == %s" %
           mask.dtype)

     mask = np.squeeze(mask)
     if len(mask.shape) != 2:
       raise ValueError(
           "encode_binary_mask expects a 2d mask, received shape == %s" %
           mask.shape)

     # convert input mask to expected COCO API input --
     mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
     mask_to_encode = mask_to_encode.astype(np.uint8)
     mask_to_encode = np.asfortranarray(mask_to_encode)

     # RLE encode mask --
     encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

     # compress and base64 encoding --
     binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
     base64_str = base64.b64encode(binary_str)
     return base64_str

    ######## seg_results_pkl_to_sub
    from multiprocessing import Pool
    import funcy
    import pycocotools.mask as maskUtils
            
    print(f'pkl_path = {pkl_path}')        
    results = mmcv.load(pkl_path)
    
    i = int(os.path.basename(pkl_path).split('_')[-1].split('of')[0])
 
    gc.collect()
    gc.collect()
    gc.collect()
    assert len(results)<=4000 and len(results)>=3999
    
    sub = pd.read_csv(data_dir+'sample_empty_submission_seg.csv').\
                set_index('ImageID').iloc[(4000*i):(4000*(i+1))]
    img_lst = sub.index.values        

    # multi process 6 core, 8 min, 50k

    i_lst = list(range(len(img_lst)))
    def process_img(i_sublst):    
        df = pd.DataFrame(columns=['ImageID','ImageWidth', 'ImageHeight', 'PredictionString']).set_index('ImageID')
        for i in i_sublst:
            cnt = 0
    #        for i in tqdm(range(len(results))):
            bb_result, segm_result = results[i]
            bbs = mmcv.concat_list(bb_result)
            scoring_flag = False
            if len(segm_result)==2: ## mask scoring rcnn
                scoring_flag = True
                
            if scoring_flag:
                segms = mmcv.concat_list(segm_result[0])
                probs = mmcv.concat_list(segm_result[1])
            else:
                segms = mmcv.concat_list(segm_result)   
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bb_result)
            ]
            labels = [CLASSES[i] for i in np.concatenate(labels)]
        
            assert len(segms)==len(labels) and len(segms)==len(bbs)  
            if scoring_flag: assert len(segms)==len(probs)
            cnt += len(segms)
            if len(segms)==0:
                df.loc[sub.index.values[i]] = -1,-1,np.nan
            else:
                row = ""         
                h,w = segms[0]['size']                
                if scoring_flag:
                    for proba,seg,label in zip(probs,segms,labels):                    
                        prob = "{:.8f}".format(proba) if args.eight_digit else "{:.6f}".format(proba)
                        mask = maskUtils.decode(seg).astype(np.bool)
                        rle = encode_binary_mask(mask).decode("utf-8")
                        row += (label+' '+prob+' '+rle+' ')
                        if args.expand:
                            for parent in all_keyed_child[label]:
                                row += (parent+' '+prob+' '+rle+' ')                      
                    
    #                sub.loc[sub.index[i]] = w,h,row.strip(' ')
                else:
                    for bb,seg,label in zip(bbs,segms,labels):                    
                        prob = "{:.8f}".format(bb[4]) if args.eight_digit else "{:.6f}".format(bb[4])
                        mask = maskUtils.decode(seg).astype(np.bool)
                        rle = encode_binary_mask(mask).decode("utf-8")
                        row += (label+' '+prob+' '+rle+' ')
                        if args.expand:
                            for parent in all_keyed_child[label]:
                                row += (parent+' '+prob+' '+rle+' ')    
                df.loc[sub.index.values[i]] = w,h,row.strip(' ')
        return df


    chunks = funcy.lchunks(int(len(i_lst)/100), i_lst)
    p = Pool(processes=num_processes)
    df_list = list(tqdm(p.imap(process_img, chunks, chunksize=1), total=len(chunks)))
    p.close()
    p.join()

    print(f"len(df_list) = " +str(len(df_list)))

    df_total = pd.concat(df_list)
    print(f"df_total.shape = " +str(df_total.shape))

    df_total.ImageWidth=df_total.ImageWidth.astype('int')
    df_total.ImageHeight=df_total.ImageHeight.astype('int')
       

    df_total.reset_index().to_csv(pkl_path.replace('.pkl', '.csv').replace('LB_pkl','LB_csv'),index=False)


    ## combining 25 csv
    if i==24:            
        del df_total, results, df_list
        gc.collect()
        gc.collect()
        
        sub = None
        for j in range(25):
            df = pd.read_csv(pkl_path.replace('_24of25',f'_{j}of25').replace('LB_pkl','LB_csv').replace('.pkl', '.csv'))
            if j==0:
                sub = df.copy()
            else:
                sub = pd.concat([sub,df])
        
        sub_filename = pkl_path.split('/')[-2].replace('cascade_mask_rcnn','cmrcnn').replace('_fpn_1x','') + \
                        os.path.basename(pkl_path).replace('LB_res','').replace('_24of25','').replace('.pkl', '.csv')
        sub.to_csv(sub_dir + sub_filename,index=False)        
        
    
    
    
    
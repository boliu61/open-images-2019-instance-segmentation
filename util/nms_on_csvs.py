import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle
import mmcv
import argparse
from pycocotools import mask
from multiprocessing import Pool
import funcy
from functools import partial
import base64
import zlib

pd.set_option('display.max_columns', 30)

def calc_iou(boxA, boxB):
    # input are tuples (xmin, ymin, xmax, ymax)
    
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA) * max(0, yB - yA)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
	boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / (boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou

def iou(msk1,msk2):
    assert msk1.shape==msk2.shape
    assert len(msk1.shape)==2
    i = np.logical_and(msk1,msk2).sum()
    if i==0: return 0
    u = np.logical_or(msk1,msk2).sum()
    return i/u
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path')
    parser.add_argument('--csv_path_1')
    parser.add_argument('--csv_path_2')
    parser.add_argument('--csv_path_3')
    parser.add_argument('--csv_path_4')    
    parser.add_argument('--out_path')
    parser.add_argument('--mask_voting', type=int, default=0)
    parser.add_argument('--thres', type=float)
    parser.add_argument('--iou_thr', type=float,default=0.5)
    parser.add_argument('--single_or_two') # 'single' or 'two' or 'three'
    args = parser.parse_args()
    
    thres = args.thres
    mask_voting = (args.mask_voting==1)
    if mask_voting: print('--- mask voting ----')


    ##### NMS on single model's output

    assert args.single_or_two in ['single','two','three']
    
    if args.single_or_two in ['two','three']:    

        if args.single_or_two == 'two':        
            print('--- NMS ensemble two models ----')            
            csv_path1 = args.csv_path_1
            csv_path2 = args.csv_path_2
            print(os.path.basename(csv_path1))
            print(os.path.basename(csv_path2))
            out_path = args.out_path
            
            df1 = pd.read_csv(csv_path1)
            df2 = pd.read_csv(csv_path2)
            
            if 'Unnamed: 0' in df1.columns: df1.drop(columns=['Unnamed: 0'],inplace=True)
            
    #        assert df1.shape==df2.shape
            assert set(df1.ImageID.unique()) == set(df2.ImageID.unique())
            
            df = pd.concat([df1,df2])        
        else:            
            csv_path1 = args.csv_path_1
            csv_path2 = args.csv_path_2
            csv_path3 = args.csv_path_3  
            csv_path4 = args.csv_path_4
            
            if csv_path4: print('--- NMS ensemble 4 models ----')
            else: print('--- NMS ensemble 3 models ----') 
            
            print(os.path.basename(csv_path1))
            print(os.path.basename(csv_path2))
            print(os.path.basename(csv_path3))            
            if csv_path4: print(os.path.basename(csv_path4))  
            out_path = args.out_path
         
            df1 = pd.read_csv(csv_path1)
            df2 = pd.read_csv(csv_path2)
            df3 = pd.read_csv(csv_path3)        
            if csv_path4: df4 = pd.read_csv(csv_path4)   
            
            if 'Unnamed: 0' in df1.columns: df1.drop(columns=['Unnamed: 0'],inplace=True)
            
            assert set(df1.ImageID.unique()) == set(df2.ImageID.unique())
            assert set(df1.ImageID.unique()) == set(df3.ImageID.unique())
            
            if csv_path4: 
                assert set(df1.ImageID.unique()) == set(df4.ImageID.unique())
                df = pd.concat([df1,df2,df3,df4])
            else:
                df = pd.concat([df1,df2,df3])              
            
    
    if args.single_or_two == 'single':
        print('--- NMS dedupe single model ----')

        csv_path = args.csv_path
        print(os.path.basename(csv_path))
        
        df = pd.read_csv(csv_path)
        
        out_path = csv_path.replace('.csv','_nms_dedup.csv')

    orig_img_ids = df.ImageID.unique()
        
    LB_flag=False
    ## convert LB sub format to val csv format
    if 'Score' not in df and 'PredictionString' in df:
        LB_flag=True
        df = df[pd.notnull(df.PredictionString)]        
        assert df.PredictionString.apply(lambda x:len(x.split(' '))%3).max()==0
        assert df.PredictionString.apply(lambda x:len(x.split(' '))//3).min()>=1
        df_converted = pd.DataFrame(columns=['ImageID', 'ImageWidth', 'ImageHeight','Score','Mask','LabelName'])
        df_converted.LabelName = mmcv.concat_list(df.PredictionString.apply(lambda x:x.split(' ')[0::3]).values)
        scores = mmcv.concat_list(df.PredictionString.apply(lambda x:x.split(' ')[1::3]).values)
        df_converted.Score = [float(x) for x in scores]
        df_converted.Mask = mmcv.concat_list(df.PredictionString.apply(lambda x:x.split(' ')[2::3]).values)            
        df_converted.ImageID = mmcv.concat_list([[df.iloc[i].ImageID] * (len(df.iloc[i].PredictionString.split(' '))//3) for i in range(df.shape[0])])
        df_converted.ImageWidth = mmcv.concat_list([[df.iloc[i].ImageWidth] * (len(df.iloc[i].PredictionString.split(' '))//3) for i in range(df.shape[0])])
        df_converted.ImageHeight = mmcv.concat_list([[df.iloc[i].ImageHeight] * (len(df.iloc[i].PredictionString.split(' '))//3) for i in range(df.shape[0])])
        df = df_converted

    df = df[df.Score > thres].copy()    
    df.reset_index(drop=True,inplace=True)     
    
    msk_nms_thr = args.iou_thr  
    def process_img_lst(img_lst, df):
        if mask_voting: d_updated_msk = {}
        suppresed = []
        for img in img_lst:
            df1_img = df[df.ImageID==img]
            labels = df1_img.LabelName.unique()
            for lbl in labels:    
                # cache the decoded mask
                d1 = {}
                for i1,row1 in df1_img[df1_img.LabelName==lbl].iterrows():                         
                    if LB_flag: mask_rle = zlib.decompress(base64.b64decode(row1.Mask))
                    else: mask_rle = row1.Mask
                    d1[i1] = mask.decode({'size': [row1.ImageHeight, row1.ImageWidth], 'counts': mask_rle})
                    if d1[i1].sum()==0: suppresed.append(i1)
                for i1,row1 in df1_img[df1_img.LabelName==lbl].sort_values('Score',ascending=False).iterrows():
                    if i1 in suppresed: continue                    
                    i1_suppresed = [] # similar masks to i1 that are suppressed
                    for i2,row2 in df1_img[df1_img.LabelName==lbl].sort_values('Score',ascending=False).iterrows():
                        if row2.Score>=row1.Score: continue
                        if i2 in suppresed: continue
                        msk1 = d1[i1]
                        msk2 = d1[i2]    
                        if iou(msk1,msk2) >= msk_nms_thr:
#                            print(img,lbl,i1,i2,row1.Score,row2.Score,iou(msk1,msk2))
                            assert i1 not in suppresed
                            assert i2 not in suppresed
                            assert row1.Score >= row2.Score
                            i1_suppresed.append(i2)
                            suppresed.append(i1 if row1.Score < row2.Score else i2)
                    # mask voting
                    if mask_voting and len(i1_suppresed)>=2 and \
                        df1_img.Score.loc[i1_suppresed].sum() > df1_img.Score.loc[i1]: # otherwise can't overturn the "anchor" mask
                        sum_msk = d1[i1] * df1_img.Score.loc[i1]
                        sum_score = df1_img.Score.loc[i1]
                        for i_sup in i1_suppresed:
                            sum_msk += d1[i_sup] * df1_img.Score.loc[i_sup]
                            sum_score += df1_img.Score.loc[i_sup]
                        wt_avg_msk = (sum_msk/sum_score > 0.5).astype('uint8')
                        if LB_flag:      
                            # compress and base64 encoding --
                            binary_str = zlib.compress(mask.encode(wt_avg_msk)['counts'], zlib.Z_BEST_COMPRESSION)
                            d_updated_msk[i1] = base64.b64encode(binary_str).decode()
                        else:
                            d_updated_msk[i1] = mask.encode(wt_avg_msk)['counts'].decode()

        if mask_voting: return suppresed, d_updated_msk
        else: return suppresed
    
    chunks = funcy.lchunks(int(len(df.ImageID.unique())/100), df.ImageID.unique())
    num_processes = 12
    p = Pool(processes=num_processes)
    if mask_voting: tuple_lst = list(tqdm(p.imap(partial(process_img_lst,df=df), chunks, chunksize=1), total=len(chunks)))
    else: suppresed_lst = list(tqdm(p.imap(partial(process_img_lst,df=df), chunks, chunksize=1), total=len(chunks)))
    p.close()
    p.join()
        
    if mask_voting: 
        suppresed_lst = [x[0] for x in tuple_lst]
        num_updated = sum([len(x[1]) for x in tuple_lst])
        print(f'updated {num_updated} masks by voting')
        d_total = {}
        for tpl in tuple_lst: d_total.update(tpl[1])
        # updating voted mask
        d_voted_msk = pd.DataFrame(d_total.items())
        df.loc[d_voted_msk[0].values,'Mask'] = d_voted_msk[1].values
    num_supr = len(mmcv.concat_list(suppresed_lst))
    print(f'suppressed {num_supr} of {df.shape[0]}, {num_supr/df.shape[0]}')
    
    
    df = df.loc[df.index.difference(mmcv.concat_list(suppresed_lst))]
    
    imgs_dropped = np.setdiff1d(orig_img_ids,df.ImageID.unique())
    
    ## convert back to LB sub format
    if LB_flag:
        sub = pd.DataFrame(columns=['ImageID', 'ImageWidth', 'ImageHeight','PredictionString'])
        sub.ImageID = orig_img_ids
        sub.set_index('ImageID',inplace=True)
        def convert_img_lst(img_lst):
            d = {}
            for img in img_lst:
                rle = [None]* df[df.ImageID==img].shape[0] *3
                rle[0::3] = df[df.ImageID==img].LabelName.values
                rle[1::3] = [str(x) for x in df[df.ImageID==img].Score.values]
                rle[2::3] = df[df.ImageID==img].Mask.values
                d[img] = df[df.ImageID==img][['ImageWidth']].iloc[0].values[0], df[df.ImageID==img][['ImageHeight']].iloc[0].values[0], rle
            return d            
        chunks = funcy.lchunks(int(len(df.ImageID.unique())/100), df.ImageID.unique())
        num_processes = 12
        p = Pool(processes=num_processes)
        d_lst = list(tqdm(p.imap(convert_img_lst, chunks, chunksize=1), total=len(chunks)))
        p.close()
        p.join()            
        d_total = {}
        for d in d_lst: d_total.update(d)
        
        for img,w_h_rle in d_total.items():
            w,h,rle = w_h_rle
            sub.loc[img] = w,h,' '.join(rle)
        for img in imgs_dropped:
            sub.loc[img] = -1,-1,np.nan
        df = sub.reset_index()
        
    if mask_voting: out_path = out_path.replace('.csv','_msk_vote.csv')
    df.to_csv(out_path,index=False)
    
  

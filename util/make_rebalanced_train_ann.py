import os
import pandas as pd
import numpy as np
from glob import glob
import pickle
import shutil
from tqdm import tqdm
import platform
import mmcv
import matplotlib.pyplot as plt
import json
import gc
gc.collect()    

if __name__ == '__main__':
    
    pd.set_option('display.max_columns', 30)
    
    data_dir = '/Users/bo_liu/Documents/open-images/data/'
    repo_dir = '/Users/bo_liu/Documents/open-images/open-images/'
    
        
    #### rebalance training set
    
    mapping = pd.read_csv(data_dir+'seg_anno/challenge-2019-classes-description-segmentable.csv',
                          header=None,names=['code','cat']).set_index('code')
    
    leaves = mmcv.load(data_dir+'list_of_275_leave_labels_seg.pkl')
    val_seg_2844 = mmcv.load(data_dir+'list_of_val_seg_2844.pkl')
    
    
    train_gt = pd.read_csv(data_dir+'seg_anno/challenge-2019-train-segmentation-masks.csv')
    val_gt = pd.read_csv(data_dir+'seg_anno/challenge-2019-validation-segmentation-masks.csv')
    val_gt_non_2844 = val_gt.loc[~val_gt.ImageID.isin(val_seg_2844)].copy()
    
    # add val-val_2844
    train_gt = pd.concat([train_gt,val_gt_non_2844])
    
    # bad image
    train_gt = train_gt.loc[~(train_gt.ImageID=='1d5ef05c8da80e31')]
    
    tr_leaves = train_gt.set_index('LabelName').loc[leaves].reset_index()
    
    
    tr_leaves = tr_leaves.rename(columns={'LabelName':'code'}).merge(mapping.reset_index(),
                                 on='code',how='left')
    tr_leaves = tr_leaves[['code','ImageID','cat']].copy()
    
    tr_leaves_nodup = tr_leaves.drop_duplicates().copy()
    
    
    
    
    # rebalance sampling target for each group
    #rank[0:24] 89k - 6k, downsample to 6k
    #rank[24:64] 6k - 1500, keep all
    #rank[64:241] 1500 - 150, oversample to 1500
    #rank[241:275] 150-13, oversample x10
    
    target = tr_leaves_nodup.cat.value_counts().values
    target[0:24] = 6000
    target[64:241] = 1500
    target[241:] = target[241:]*10
    
    
    
#    plt.figure(figsize=(12,8))
#    plt.plot(range(275),tr_leaves_nodup.cat.value_counts().values,marker='o',markersize=1)
#    plt.plot(range(275),target)
#    #plt.xlim((50,100))
#    plt.ylim((0,6100))
#    plt.xlabel('rank (0 is largest class)')
#    plt.ylabel('num of imgs (not masks) per class')
#    plt.show()
    
    
    ### start sampling
    
    
    for rnd in range(2,9):
    
        sampled = pd.DataFrame(columns=['ImageID','cnt'])
        sampled['ImageID'] = tr_leaves_nodup.ImageID.unique()
        sampled['cnt'] = 0
        sampled.set_index('ImageID',inplace=True)
        
        
        grp4 = tr_leaves_nodup.cat.value_counts().iloc[241:].index.values
        sampled.loc[tr_leaves_nodup.set_index('cat').loc[grp4].ImageID.unique()] = 10
        
        
        grp3 = tr_leaves_nodup.cat.value_counts().index.values[64:241][::-1]
        
        for cat in tqdm(grp3):
        #    if cat=='Toilet paper': break
            target = 1500
            already_sampled = tr_leaves_nodup[tr_leaves_nodup.cat==cat].merge(
                    sampled[sampled.cnt>0].reset_index(),how='inner',on=['ImageID']).ImageID.values
            target -= sampled.loc[already_sampled].cnt.sum()
            if target <0: 
                print(cat)
                print(sampled.loc[already_sampled].cnt.sum())
                continue
            this_cat_imgs = tr_leaves_nodup[tr_leaves_nodup.cat==cat].ImageID.values
            imgs_left = np.setdiff1d(this_cat_imgs, already_sampled)    
            times = target//imgs_left.shape[0]
            rem = target%imgs_left.shape[0]
            sampled.loc[imgs_left] += times
            np.random.seed(100+rnd)
            sampled.loc[np.random.choice(imgs_left, rem, replace=False)] += 1    
            if sampled.cnt.max()>10: break    
            
        sampled.cnt.sort_values(ascending=False)
        sampled.cnt.value_counts()
        
        grp2 = tr_leaves_nodup.cat.value_counts().index.values[24:64]
        for cat in tqdm(grp2):
             this_cat_imgs = tr_leaves_nodup[tr_leaves_nodup.cat==cat].ImageID.values
             sampled.loc[this_cat_imgs,'cnt'] = sampled.loc[this_cat_imgs].cnt.apply(lambda x:max(1,x))
        
        grp1 = tr_leaves_nodup.cat.value_counts().index.values[0:24][::-1]
        for cat in tqdm(grp1):
            this_cat_imgs = tr_leaves_nodup[tr_leaves_nodup.cat==cat].ImageID.values
            target = 6000
            already_sampled = tr_leaves_nodup[tr_leaves_nodup.cat==cat].merge(
                    sampled[sampled.cnt>0].reset_index(),how='inner',on=['ImageID']).ImageID.values
            target -= sampled.loc[already_sampled].cnt.sum()
            imgs_left = np.setdiff1d(this_cat_imgs, already_sampled) 
            if target <0: 
                print(cat)
                print(sampled.loc[already_sampled].cnt.sum())
                target = 1000
            np.random.seed(100+rnd)
            sampled.loc[np.random.choice(imgs_left, target, replace=False)] += 1 
        
        
        ## final sampling results    
        sampled.cnt.sum()
        #Out[296]: 450757
        # sampled 450,757 (incl dup) out of 657,451
        
        sampled.reset_index().to_csv(data_dir+f'seg_balance_rnd{rnd}_{sampled.cnt.sum()}.csv' ,index=False)
        

    
    orig_order = tr_leaves_nodup.cat.value_counts().index.values
    final_cnt = tr_leaves_nodup.merge(sampled.reset_index(),how='left',on='ImageID').groupby('cat').sum()['cnt'].sort_values(ascending=False)
    
    
    plt.figure(figsize=(12,8))
    plt.plot(range(275),tr_leaves_nodup.cat.value_counts().values,marker='o',markersize=1)
    plt.plot(range(275),final_cnt.loc[orig_order].values)
    #plt.xlim((50,100))
    plt.ylim((0,6100))
    plt.xlabel('rank (0 is largest class)')
    plt.ylabel('num of imgs (not masks) per class')
    plt.show()
    
    
    
    ### generating the list of 450757 imgs
    
    
    tr_ann = mmcv.load(data_dir + 'mmdet_anno/seg_train_275_leave_cls_ann.pkl')
    val_ann=mmcv.load(data_dir + 'mmdet_anno/seg_val_275_leave_cls_ann.pkl')
    tr_ann = tr_ann + val_ann
    len(tr_ann)
    tr_ann_filenames = [x['filename'][:-4] for x in tr_ann]
    
    lst_all = []
    
    for rnd in tqdm(range(2,9)):
        sampled = pd.read_csv(glob(data_dir+f'seg_balance_rnd{rnd}_*.csv')[0])
    
        lst = sampled.query("cnt==1").ImageID.unique()
        np.random.seed(20+rnd)
        np.random.shuffle(lst) 
        
        for cnt in range(2,11):
            lst_cnt = sampled[sampled.cnt==cnt].ImageID.unique()
            lst_new = []
            for i in range(cnt):
                n = int(np.ceil(len(lst)/cnt)) # sub lst len (target)
                orig_sub_lst = lst[(i*n):(i*n+n)]
                len_sub_lst = len(orig_sub_lst) + len(lst_cnt) # sub lst len (actual)        
                sub_lst = np.repeat(None,len_sub_lst) # empty sub lst
        
                orig_sub_idx = np.random.choice(len_sub_lst, len(orig_sub_lst), replace=False)
                sub_lst[orig_sub_idx] = orig_sub_lst
        
                lst_cnt_i = lst_cnt.copy()
                np.random.seed(20+ rnd +i)
                np.random.shuffle(lst_cnt_i)         
                cnt_lst_idx = np.setdiff1d(list(range(len_sub_lst)), orig_sub_idx)
                sub_lst[cnt_lst_idx] = lst_cnt_i
                
                lst_new = np.concatenate((lst_new,sub_lst.copy()))    
            lst = lst_new.copy()
            
        lst_all = np.concatenate((lst_all, lst.copy()))     
    
    
    mmcv.dump(lst_all, data_dir + 'mmdet_anno/seg_balance_rnd2to8_filenames_3196859.pkl')
                
    
    

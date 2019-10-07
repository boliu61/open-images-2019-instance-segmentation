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
    
    
        
    #### same as before, but include OID test images and oversample them
    
        
    mapping = pd.read_csv(data_dir+'seg_anno/challenge-2019-classes-description-segmentable.csv',
                          header=None,names=['code','cat']).set_index('code')
    
    leaves = mmcv.load(data_dir+'list_of_275_leave_labels_seg.pkl')
    val_seg_2844 = mmcv.load(data_dir+'list_of_val_seg_2844.pkl')
    
    
    train_gt = pd.read_csv(data_dir+'seg_anno/challenge-2019-train-segmentation-masks.csv')
    val_gt = pd.read_csv(data_dir+'seg_anno/challenge-2019-validation-segmentation-masks.csv')
    val_gt_non_2844 = val_gt.loc[~val_gt.ImageID.isin(val_seg_2844)].copy()
    test_gt = pd.read_csv(data_dir+'seg_anno/test-annotations-object-segmentation.csv',usecols=['MaskPath', 'ImageID', 'LabelName'])
    
    # add val-val_2844
    train_gt = pd.concat([train_gt,val_gt_non_2844])
    
    # bad image
    train_gt = train_gt.loc[~(train_gt.ImageID=='1d5ef05c8da80e31')]
    
    
    tr_leaves = train_gt.set_index('LabelName').loc[leaves].reset_index()
    ts_leaves = test_gt.set_index('LabelName').loc[leaves].reset_index()
    
    tr_leaves = tr_leaves.rename(columns={'LabelName':'code'}).merge(mapping.reset_index(),on='code',how='left')
    tr_leaves = tr_leaves[['code','ImageID','cat']].copy()
    ts_leaves = ts_leaves.rename(columns={'LabelName':'code'}).merge(mapping.reset_index(),on='code',how='left')
    ts_leaves = ts_leaves[['code','ImageID','cat']].copy()
    ts_leaves.dropna(subset=['ImageID'],inplace=True)
    
    tr_leaves_nodup = tr_leaves.drop_duplicates().copy()
    ts_leaves_nodup = ts_leaves.drop_duplicates().copy()
    
    
    cnt_comp = tr_leaves_nodup.cat.value_counts().to_frame().reset_index().merge(
               ts_leaves_nodup.cat.value_counts().to_frame().reset_index(),on='index',how='left')
    cnt_comp['ratio']=cnt_comp.cat_y/cnt_comp.cat_x
    cnt_comp.sort_values('ratio',ascending=False)
    
    # rebalance sampling target for each group
    #rank[0:24] 89k - 6k, downsample to 5k + all test
    #rank[24:64] 6k - 1500, keep all + all test
    #rank[64:241] 1500 - 150, test x 5 (most 1350), then oversample to 1500
    #rank[241:275] 150-13, test x 10, old x 8
    
    
    
    ### start sampling
    
    for rnd in range(9,16):
    
        sampled = pd.DataFrame(columns=['ImageID','cnt'])
        sampled['ImageID'] = np.concatenate((tr_leaves_nodup.ImageID.unique(),ts_leaves_nodup.ImageID.unique()))
        sampled['cnt'] = 0
        sampled.set_index('ImageID',inplace=True)
            
        grp4 = tr_leaves_nodup.cat.value_counts().iloc[241:].index.values
        sampled.loc[tr_leaves_nodup.set_index('cat').loc[grp4].ImageID.unique()] = 8
        sampled.loc[ts_leaves_nodup.set_index('cat').loc[grp4].dropna(subset=['ImageID']).ImageID.unique()] = 10    
        
        grp3 = tr_leaves_nodup.cat.value_counts().index.values[64:241][::-1]
        sampled.loc[ts_leaves_nodup.set_index('cat').loc[grp3].ImageID.unique()] = 5    
        
        for cat in tqdm(grp3):
        #    if cat=='Toilet paper': break
            target = 1500
            already_sampled = tr_leaves_nodup[tr_leaves_nodup.cat==cat].merge(
                    sampled[sampled.cnt>0].reset_index(),how='inner',on=['ImageID']).ImageID.values
            already_sampled = np.concatenate((already_sampled,ts_leaves_nodup[ts_leaves_nodup.cat==cat].merge(
                    sampled[sampled.cnt>0].reset_index(),how='inner',on=['ImageID']).ImageID.values))
                    
            target -= sampled.loc[already_sampled].cnt.sum()
            if target <0: 
                print(cat)
                print(sampled.loc[already_sampled].cnt.sum())
                target = 500
            this_cat_imgs = tr_leaves_nodup[tr_leaves_nodup.cat==cat].ImageID.values
            imgs_left = list(set(this_cat_imgs)-set(already_sampled))   
            times = target//len(imgs_left)
            rem = target%len(imgs_left)
            sampled.loc[imgs_left] += times
            np.random.seed(100+rnd)
            sampled.loc[np.random.choice(imgs_left, rem, replace=False)] += 1    
            if sampled.cnt.max()>10: break    
            
        sampled.cnt.sort_values(ascending=False)
        sampled.cnt.value_counts()
        
        grp2 = tr_leaves_nodup.cat.value_counts().index.values[24:64]
        for cat in tqdm(grp2):
             this_cat_imgs = np.concatenate((tr_leaves_nodup[tr_leaves_nodup.cat==cat].ImageID.values,ts_leaves_nodup[ts_leaves_nodup.cat==cat].ImageID.values))
             sampled.loc[this_cat_imgs,'cnt'] = sampled.loc[this_cat_imgs].cnt.apply(lambda x:max(1,x))
      
        grp1 = tr_leaves_nodup.cat.value_counts().index.values[0:24][::-1]
        for cat in tqdm(grp1):
            this_cat_imgs = ts_leaves_nodup[ts_leaves_nodup.cat==cat].ImageID.values
            sampled.loc[this_cat_imgs,'cnt'] = sampled.loc[this_cat_imgs].cnt.apply(lambda x:max(1,x))        
            
            this_cat_imgs = tr_leaves_nodup[tr_leaves_nodup.cat==cat].ImageID.values
            target = 5000
            already_sampled = tr_leaves_nodup[tr_leaves_nodup.cat==cat].merge(
                    sampled[sampled.cnt>0].reset_index(),how='inner',on=['ImageID']).ImageID.values
            already_sampled = np.concatenate((already_sampled,ts_leaves_nodup[ts_leaves_nodup.cat==cat].merge(
                    sampled[sampled.cnt>0].reset_index(),how='inner',on=['ImageID']).ImageID.values))                
            target -= sampled.loc[already_sampled].cnt.sum()
            imgs_left = list(set(this_cat_imgs)-set(already_sampled))   
            if target <0: 
                print(cat)
                print(sampled.loc[already_sampled].cnt.sum())
                target =500
            np.random.seed(100+rnd)
            sampled.loc[np.random.choice(imgs_left, target, replace=False)] += 1 
            
        ## final sampling results    
        sampled.cnt.sum()
        #Out[296]: 449715
        # sampled 449,715 (incl dup) out of 690,349
        
        sampled.reset_index().to_csv(data_dir+f'seg_incl_test_rnd{rnd}_{sampled.cnt.sum()}.csv' ,index=False)
        
    
    
    orig_order = tr_leaves_nodup.cat.value_counts().index.values
    final_cnt = pd.concat([tr_leaves_nodup,ts_leaves_nodup]).merge(sampled.reset_index(),how='left',on='ImageID').groupby('cat').sum()['cnt'].sort_values(ascending=False)
    
    
#    plt.figure(figsize=(12,8))
#    plt.plot(range(275),tr_leaves_nodup.cat.value_counts().values,marker='o',markersize=1)
#    plt.plot(range(275),final_cnt.loc[orig_order].values)
#    #plt.xlim((50,100))
#    plt.ylim((0,7000))
#    plt.xlabel('rank (0 is largest class)')
#    plt.ylabel('num of imgs (not masks) per class')
#    plt.show()    
        
        
    
    ### generating the list of img names
    
    lst_all = []
    for rnd in tqdm(range(9,16)):
        sampled = pd.read_csv(glob(data_dir+f'seg_incl_test_rnd{rnd}_*.csv' )[0])
    
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
        assert str(lst.shape[0]) in glob(data_dir+f'seg_incl_test_rnd{rnd}_{sampled.cnt.sum()}.csv' )[0]
            
        lst_all = np.concatenate((lst_all, lst.copy()))     
    
    
    #mmcv.dump(lst_all, data_dir + 'mmdet_anno/seg_oversample_test_rnd9to15_filenames_3147572.pkl')
    lst_all = mmcv.load(data_dir + 'mmdet_anno/seg_oversample_test_rnd9to15_filenames_3147572.pkl')
    
    
    all_ann = mmcv.load(data_dir + 'mmdet_anno/seg_train_275_leave_cls_ann.pkl') +\
              mmcv.load(data_dir + 'mmdet_anno/seg_val_275_leave_cls_ann.pkl') +\
              mmcv.load(data_dir + 'mmdet_anno/seg_test_leaves_ann.pkl')
    
    
    d_all_ann = {all_ann[i]['filename'][:-4]:i for i in range(len(all_ann))}
    
    final_ann = [all_ann[d_all_ann[img]] for img in lst_all]
    mmcv.dump(final_ann, data_dir+ 'seg_oversample_test_rnd9to15_ann_3147572.pkl')
    
    
    
    

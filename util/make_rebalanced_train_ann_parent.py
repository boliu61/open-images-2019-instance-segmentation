import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle
import mmcv
import argparse
import platform
from glob import glob 

if __name__ == '__main__':
    
    data_dir = '/Users/bo_liu/Documents/open-images/data/'
    repo_dir = '/Users/bo_liu/Documents/open-images/open-images/'
    sub_dir = '/Users/bo_liu/Documents/open-images/subs/'
    
    
    all_keyed_child = mmcv.load(data_dir+'seg_all_keyed_child.pkl') # child -> parents
    all_keyed_child = {k:v for (k,v) in all_keyed_child.items() if len(v)>0}
    all_keyed_child_1 = {k:v for (k,v) in all_keyed_child.items() if len(v)==1}
    all_keyed_child_2 = {k:v for (k,v) in all_keyed_child.items() if len(v)==2}
    all_keyed_child_3 = {k:v for (k,v) in all_keyed_child.items() if len(v)==3}
    
    
    all_keyed_parent = mmcv.load(data_dir+'seg_all_keyed_parent.pkl') # parent -> children
    
    
    def expand_parent_labels(train_gt):
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
    
    all_parents = [k for (k,v) in all_keyed_parent.items() if len(v)>0]
    
    val_seg_3841 = mmcv.load(data_dir + 'seg_anno/list_of_val_seg_3841.pkl')
    
    
    train_gt = pd.read_csv(data_dir+'seg_anno/challenge-2019-train-segmentation-masks.csv',usecols=['MaskPath', 'ImageID', 'LabelName'])
    val_gt = pd.read_csv(data_dir+'seg_anno/challenge-2019-validation-segmentation-masks.csv',usecols=['MaskPath', 'ImageID', 'LabelName'])
    test_gt = pd.read_csv(data_dir+'seg_anno/test-annotations-object-segmentation.csv',usecols=['MaskPath', 'ImageID', 'LabelName'])
    train_gt = pd.concat([train_gt,val_gt,test_gt])
    
    
    train_gt = train_gt.loc[~(train_gt.ImageID=='1d5ef05c8da80e31')] # 坏图
    train_gt = train_gt.loc[~train_gt.ImageID.isin(val_seg_3841)].copy()
    
    
    train_gt = expand_parent_labels(train_gt)
    
    
    # remove Carnivore and Reptile
    all_23 = [x for x in all_parents if x not in ['/m/01lrl','/m/06bt6']]
    
    train_gt = train_gt.set_index('LabelName').loc[all_23]
    
    mapping = pd.read_csv(data_dir+'seg_anno/challenge-2019-classes-description-segmentable.csv',
                          header=None,names=['code','cat']).set_index('code')
    
    train_gt_cnt = train_gt.index.value_counts().reset_index().\
            rename(columns={'index':'code','LabelName':'cnt'}).\
            merge(mapping.reset_index(),on='code',how='left')
    
    train_gt_nodup = train_gt.reset_index().drop_duplicates(subset=['LabelName','ImageID']).copy()
    
    train_nodup_cnt = train_gt_nodup.\
                        LabelName.value_counts().reset_index().\
                        rename(columns={'index':'code','LabelName':'cnt'}).\
                        merge(mapping.reset_index(),on='code',how='left')
    
    # sampling target
    #0-4: downsample to 30k
    #5-9: keep all, 30k-10k
    #10-24: upsample to 10k
    
#    target = train_nodup_cnt.cnt.values.copy()
#    target[0:5] = 30000
#    target[10:] = 10000
#    
#    plt.figure(figsize=(12,8))
#    plt.plot(range(23),train_nodup_cnt.cnt.values,marker='o',markersize=2)
#    plt.plot(range(23),target,marker='o',markersize=2)
#    #plt.xlim((50,100))
#    plt.ylim((0,100000))
#    plt.xlabel('rank (0 is largest class)')
#    plt.ylabel('num of imgs (not masks) per class')
#    plt.show()
    
    
    ### start sampling
    
    for rnd in range(10):
    
        sampled = pd.DataFrame(columns=['ImageID','cnt'])
        sampled['ImageID'] = train_gt.ImageID.unique()
        sampled['cnt'] = 0
        sampled.set_index('ImageID',inplace=True)
           
        grp3 = train_nodup_cnt.code.values[10:][::-1]
        
        for cat in tqdm(grp3):
    #        if cat=='/m/0ch_cf': break
            target = 10000
            already_sampled = train_gt_nodup.set_index('LabelName').loc[cat].reset_index().merge(
                    sampled[sampled.cnt>0].reset_index(),how='inner',on=['ImageID']).ImageID.values
            target -= sampled.loc[already_sampled].cnt.sum()
            if target <0: 
                print(cat)
                print(sampled.loc[already_sampled].cnt.sum())
                continue
            this_cat_imgs = train_gt_nodup.set_index('LabelName').loc[cat].ImageID.values
            imgs_left = np.setdiff1d(this_cat_imgs, already_sampled)    
            times = target//imgs_left.shape[0]
            rem = target%imgs_left.shape[0]
            sampled.loc[imgs_left] += times
            np.random.seed(100+rnd)
            sampled.loc[np.random.choice(imgs_left, rem, replace=False)] += 1    
            if sampled.cnt.max()>10: break    
    
        sampled.cnt.sort_values(ascending=False)
        sampled.cnt.value_counts()
        
        grp2 = train_nodup_cnt.code.values[5:10]
        for cat in tqdm(grp2):
             this_cat_imgs = train_gt_nodup[train_gt_nodup.LabelName==cat].ImageID.values
             sampled.loc[this_cat_imgs,'cnt'] = sampled.loc[this_cat_imgs].cnt.apply(lambda x:max(1,x))
        
        grp1 = train_nodup_cnt.code.values[:5][::-1]
        for cat in tqdm(grp1):
            if cat=='/m/01g317':break
            this_cat_imgs = train_gt_nodup[train_gt_nodup.LabelName==cat].ImageID.values
            target = 30000
            already_sampled = train_gt_nodup[train_gt_nodup.LabelName==cat].merge(
                    sampled[sampled.cnt>0].reset_index(),how='inner',on=['ImageID']).ImageID.values
            target -= sampled.loc[already_sampled].cnt.sum()
            imgs_left = list(set(this_cat_imgs)-set(already_sampled))
            if target <0: 
                print(cat)
                print(sampled.loc[already_sampled].cnt.sum())
                target = 5000
            np.random.seed(100+rnd)
            sampled.loc[np.random.choice(imgs_left, target, replace=False)] += 1 
        
        ## final sampling results    
        sampled.cnt.sum()
        #Out[296]: 329732
        # sampled 329,732 (incl dup) out of 643,669
        
        sampled.reset_index().to_csv(data_dir+f'seg_balance_23_parents_rnd{rnd}_{sampled.cnt.sum()}.csv' ,index=False)
        
    
    
    orig_order = train_nodup_cnt.code.values
    final_cnt = train_gt_nodup.merge(sampled.reset_index(),how='left',on='ImageID').groupby('LabelName').sum()['cnt'].sort_values(ascending=False)
    
    
    plt.figure(figsize=(12,8))
    plt.plot(range(23),train_nodup_cnt.cnt.values,marker='o',markersize=1)
    plt.plot(range(23),final_cnt.loc[orig_order].values)
    #plt.xlim((50,100))
    plt.ylim((0,100000))
    plt.xlabel('rank (0 is largest class)')
    plt.ylabel('num of imgs (not masks) per class')
    plt.show()
    
    
    
    
    ### generating the list of 329732 imgs
    
    
    all_ann = mmcv.load(data_dir + 'mmdet_anno/seg_all_parent_23_ann_607601.pkl')
    tr_ann_filenames = [x['filename'][:-4] for x in all_ann]
    
    lst_all = []
    
    for rnd in tqdm(range(10)):
        sampled = pd.read_csv(glob(data_dir+f'seg_balance_23_parents_rnd{rnd}_*.csv')[0])
    
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
    
    
    d_all_ann = {all_ann[i]['filename'][:-4]:i for i in range(len(all_ann))}
    
    final_ann = [all_ann[d_all_ann[img]] for img in lst_all]
    mmcv.dump(final_ann, data_dir+ 'seg_bal_23_parents_10rnd_ann_3297102.pkl')
    
    
    
    

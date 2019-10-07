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
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub_csv_pattern')
    parser.add_argument('--parents_only', type=int,default=0)   
    parser.add_argument('--no_expand',type=int,default=0)
    parser.add_argument('--thres', type=float)    
    args = parser.parse_args()
        

    data_dir = '/Users/bo_liu/Documents/open-images/data/'
    repo_dir = '/Users/bo_liu/Documents/open-images/open-images/'
    sub_dir = '/Users/bo_liu/Documents/open-images/subs/'
    
    all_keyed_child = mmcv.load(data_dir+'seg_all_keyed_child.pkl')
        
    sub_csvs = sorted(glob(args.sub_csv_pattern))

    for sub_csv in sub_csvs:
        assert 'of25.csv' in sub_csv or 'of25_msk_vote' in sub_csv
        if 'of25_msk_vote' in sub_csv: 
            k = int(sub_csv.split('of25_msk_vote')[-2].split('_')[-1])
        else: 
            k = int(sub_csv.replace('of25.csv','').split('_')[-1])
        assert k>=0 and k<=24

        import gc;gc.collect()
        sub=pd.read_csv(sub_csv)
        
        thres = args.thres #0.001659
        
        prob_lst = []
        
        for i in tqdm(range(len(sub))):
            string = sub.loc[i,'PredictionString']
            if type(string) is float and np.isnan(string): continue
            new_string = ''
            lst = string.split(' ')
            assert len(lst)%3==0
            for j in range(len(lst)//3):
        #        prob_lst.append(float(lst[3*j+1]))
                if float(lst[3*j+1]) < thres:
                    continue
                if not args.parents_only: new_string += (' ' + lst[3*j] + ' ' + lst[3*j+1] + ' ' + lst[3*j+2])
                if not args.no_expand:
                    for parent in all_keyed_child[lst[3*j]]:
                       new_string +=  (' ' + parent + ' ' + lst[3*j+1] + ' ' + lst[3*j+2])
            sub.loc[i,'PredictionString'] = new_string.strip(' ')            
                
        sub_filename = sub_csv[:-4] + ('' if args.no_expand else '_expand') +f'_thr{args.thres}.csv'
        if args.parents_only:
            sub_filename = sub_filename.replace('.csv','_25cls.csv')   
        sub.to_csv( sub_filename,index=False)        


    ## combining 25 csv
    assert len(sub_csvs)==25    
    if k==9:            
        gc.collect()
        gc.collect()
        
        sub = None
        for sub_csv in sub_csvs:
            df = pd.read_csv(sub_csv[:-4] + ('' if args.no_expand else '_expand') + f'_thr{args.thres}' + ('_25cls.csv' if args.parents_only else '.csv' ))
            if sub is None:
                sub = df.copy()
            else:
                sub = pd.concat([sub,df])
          
        sub_filename = os.path.basename(sub_csv).replace('_9of25','')[:-4] + ('' if args.no_expand else '_expand') + f'_thr{args.thres}.csv'
        if args.parents_only:
            sub_filename = sub_filename.replace('.csv','_25cls.csv')
        sub.to_csv(sub_dir + sub_filename,index=False)  



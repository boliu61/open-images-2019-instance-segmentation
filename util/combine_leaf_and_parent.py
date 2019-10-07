import pandas as pd
from glob import glob
import os
from tqdm import tqdm

pd.set_option('display.max_columns', 25)

if __name__ == '__main__':
    

    csv_pattern1 = 'LB_csv/LB_avg3_2scale_flip_NMS_G8k_2scale_flip_thr0_0.5_*of25.csv'
    csv_pattern2 = 'LB_csv/LB_avg3_2scale_flip_thr0_120_*of25.csv'
    
    
    csv_lst1 = sorted(glob(csv_pattern1))
    csv_lst2 = sorted(glob(csv_pattern2))    
    
    assert len(csv_lst1)==25 and len(csv_lst2)==25
        
    for i in tqdm(range(25)):
        sub1 = pd.read_csv(csv_lst1[i])
        sub2 = pd.read_csv(csv_lst2[i])
        assert sub1.ImageID.equals(sub2.ImageID)
        assert sub2.PredictionString.count()==sub2.shape[0]         
        sub2['tmp'] = sub1.PredictionString.fillna('')
        sub2.PredictionString = sub2.apply(lambda x: 
            (x.PredictionString+' '+x.tmp).strip(' '),axis=1 )

        if i==0:
            sub = sub2.copy()
        else:
            sub = pd.concat([sub,sub2.copy()])
            
    print('-----sub.count()-----\n',sub.count())
    
    sub_name = os.path.basename(csv_lst1[0].replace('_0of25','')).replace('.csv','').replace('LB_','') +\
                '_AND_' +\
                os.path.basename(csv_lst2[0].replace('_0of25','')).replace('.csv','').replace('LB_','')
    
    sub.drop(columns='tmp').to_csv(\
             'subs/' + \
             sub_name + '.csv',chunksize=10000,index=False)
    
    
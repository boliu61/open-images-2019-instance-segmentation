import mmcv
import pandas as pd

if __name__ == '__main__':

   data_dir = 'open-images/data/'

   ### test ann files, split into 25 chunks for parallel inference
   
   test_img_lst = pd.read_csv(data_dir+'sample_empty_submission_seg.csv').ImageID.values
   test_ann = [{'filename': img+'.jpg'} for img in test_img_lst]
   for i in range(25):
       mmcv.dump(test_ann[(4000*i):(4000*i+4000)], data_dir+f'mmdet_anno/test_ann_{i}_of_25.pkl')
   
    
    

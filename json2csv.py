import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

#parser for trails
#use --trial trial_() to run it
parser = argparse.ArgumentParser(description='Read Trial')
parser.add_argument('--trial', type=str,
                    help='Trial Number')

args = parser.parse_args()

#different condition of images
condition = ['big_', 'small_', 'light_', 'dark_']
#classes of product
classList = ['bubly', 'clinique', 'echo', 'lotion', 'micellar', 'parm', 'protein', 'redbull', 'shade', 'skin', 'tory' ]

trial = args.trial
""" Configure Paths"""   
#to generate test_image.txt and calcualte the percentage of the indicated condition
dir_path = os.path.dirname(os.path.realpath("./*"))
def gentestList(con):
    tot_cnt = 0
    con_cnt =0
    labelpath = dir_path+"/dataset_eval/"
    imgpath = dir_path+"/dataset_eval/"
    txt_list = os.listdir(imgpath)
    
    for txt_name in txt_list:
        if not("jpg" in txt_name):
            
            continue
        tot_cnt+=1    
        if not(con in txt_name):
            continue
        con_cnt+=1
        img_filename = txt_name
        img_path = imgpath + img_filename
    
        iopen = open(dir_path+"/test_images.txt", "a")
        iopen.write(img_path+"\n")
    return con_cnt/tot_cnt, tot_cnt

#calaulate the perventahe of each class
def count(con):
    tot_cnt = 0
    con_cnt =0
    labelpath = dir_path+"/dataset_eval/"
    imgpath = dir_path+"/dataset_eval/"
    
    
    txt_list = os.listdir(imgpath)
    
    for txt_name in txt_list:
        if not("jpg" in txt_name):
            
            continue
        tot_cnt+=1    
        if not(con in txt_name):
            continue
        con_cnt+=1

    return con_cnt/tot_cnt



#for con in condition:
#Open txt
def mAPList(condition, p):
    mAPlist = []
    perclist = []
    for con in condition:
        #create test image list for this condition
        if os.path.isfile('test_images.txt'):
            os.remove('test_images.txt')
        perc, _ =gentestList(con)
        #run darknet under this conditions and p value
        os.system(dir_path+'/darknet detector map dataset_eval.data /home/vickicv/Desktop/Versioning_test/Trials/{}/yolov4-tiny-vicki-1.cfg /home/vickicv/Desktop/Versioning_test/Trials/{}/weights/yolov4-tiny-vicki-1_best.weights -iou_thresh 0.{} ->result.txt'.format(trial,trial,p))
        
        f = open ('result.txt','r')
        m = f.readlines()
        for line in m:
            if ('mean average precision (mAP@0.'+ p + ')') in line:
                first = line.split("=")[1]
                mAP = float(first.split(',')[0])*100
                
                mAPlist.append(mAP)
        perclist.append(perc)
    return mAPlist,perclist

def mAPList_class(condition, p):
    mAPlist = []
    perclist = []
    #create test image list for this condition  
    if os.path.isfile('test_images.txt'):
        os.remove('test_images.txt')
    _, tot_cnt =gentestList("")
    os.system(dir_path+'/darknet detector map dataset_eval.data /home/vickicv/Desktop/Versioning_test/Trials/{}/yolov4-tiny-vicki-1.cfg /home/vickicv/Desktop/Versioning_test/Trials/{}/weights/yolov4-tiny-vicki-1_best.weights -iou_thresh 0.{} ->result.txt'.format(trial,trial,p))
    #run darknet under this conditions and p value
    f = open ('result.txt','r')
    m = f.readlines()
    for line in m:
        if 'class_id' in line:
            for i in range(len(condition)):
                if condition[i] in line:
                    mAP = float(line.split('=')[3].split('%')[0])
                    
                #print(mAP)
            mAPlist.append(mAP)

    for i in range(len(condition)):
        perclist.append(count(condition[i]))

    return mAPlist,perclist

def mAP_all(p):
    mAP = 0
    #create test image list for this condition
    if os.path.isfile('test_images.txt'):
        os.remove('test_images.txt')
    _, _ =gentestList("")
    #run darknet under this conditions and p value
    os.system(dir_path+'/darknet detector map dataset_eval.data /home/vickicv/Desktop/Versioning_test/Trials/{}/yolov4-tiny-vicki-1.cfg /home/vickicv/Desktop/Versioning_test/Trials/{}/weights/yolov4-tiny-vicki-1_best.weights -iou_thresh 0.{} ->result.txt'.format(trial,trial,p))
    
    f = open ('result.txt','r')
    m = f.readlines()
    for line in m:
        if ('mean average precision (mAP@0.'+ p + ')') in line:
            first = line.split("=")[1]
            mAP = float(first.split(',')[0])*100

    return mAP

#extract mAP from result.txt for each condition
mAP25,perclist = mAPList(condition, '25')
mAP50,_ = mAPList(condition, '50')
mAP75,_ = mAPList(condition, '75')

#extract mAP from result.txt for each class
mAP_class_25, perclist_class = mAPList_class(classList, '25')
mAP_class_50, _ = mAPList_class(classList, '50')
mAP_class_75, _ = mAPList_class(classList, '75')

path = dir_path+'/output_{}.csv'.format(trial)
#write CSV file
with open(path, 'w', newline = '') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['condition', 'percentage', 'mAP@25' , 'mAP@50', 'mAP@75'])
    writer.writerow(['all', '100', mAP_all('25') , mAP_all('50'), mAP_all('75')])
    for i in range(len(condition)):
        writer.writerow([condition[i], perclist[i] , mAP25[i], mAP50[i], mAP75[i]])
    for i in range(len(classList)):
        writer.writerow([classList[i], perclist_class[i], mAP_class_25[i], mAP_class_50[i], mAP_class_75[i]])

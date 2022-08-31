# Setup
import os,sys
import numpy as np
import cv2
from PIL import Image
from multiprocessing import Pool
import argparse
from IPython import embed #to debug
import imageio
from IPython.display import clear_output

from skimage import io
from skimage.util import img_as_ubyte
import skvideo
skvideo.setFFmpegPath("C:\\ffmpeg\\bin")
import skvideo.io

import scipy.misc

from pathlib import Path

import tqdm # counter

# Functions

def ToImg(raw_flow,bound): # input pixels scaled to 0-255 (bi-bound)
    
    flow=raw_flow
    flow[flow>bound]=bound
    flow[flow<-bound]=-bound
    flow-=-bound
    flow*=(255/float(2*bound))

    return flow.astype(np.uint8) # added .astype(np.uint8) for conversion problems

def save_flows(flows,image,save_dir,save_dir_cls,num,bound): # save optical flow images and rawframes images
  
    flow_x=ToImg(flows[...,0],bound)
    flow_y=ToImg(flows[...,1],bound)
    
    if not os.path.exists(os.path.join(data_root,new_dir,save_dir_cls)): # new_dir is the name of the output save folder within 'data_root', by default it is videos
        os.makedirs(os.path.join(data_root,new_dir,save_dir_cls))

    #save the image
    save_img=os.path.join(data_root,new_dir,save_dir_cls,'img_{:05d}.jpg'.format(num)) # : images prefix: img_
    
    imageio.imwrite(save_img,image)  

    # save flows
    save_x=os.path.join(data_root,new_dir,save_dir_cls,'flow_x_{:05d}.jpg'.format(num)) # optical flow x prefix: flow_x_
    save_y=os.path.join(data_root,new_dir,save_dir_cls,'flow_y_{:05d}.jpg'.format(num)) # optical flow y prefix: flow_y_
    flow_x_img=Image.fromarray(flow_x) 
    flow_y_img=Image.fromarray(flow_y)
    imageio.imwrite(save_x,flow_x_img)
    imageio.imwrite(save_y,flow_y_img)
    return 0

def dense_flow(augs): # extract dense flow images
    '''
    save_dir: destination path (video_name without .ext)
    save_dir_cls: class_name//video_name (without .ext)
    step: number of frames between each two extracted frames
    bound: bi-bound parameter
    '''
    video_name,save_dir,save_dir_cls,step,bound=augs # flow_dir is passed as save_dir (essentially it is the name of the video without avi)
    
    video_path=os.path.join(videos_root,save_dir_cls + '.avi') 
  
    try:
        videocapture=skvideo.io.vread(video_path)
    except:
        print('{} read error! ').format(video_name)
        return 0
    if videocapture.sum()==0:
        print(f'Could not initialize capturing {video_name}.')
        exit()
    print(f'Inizialize capturing of {video_name}.')
    len_frame=len(videocapture)-1
    frame_num=0
    image,prev_image,gray,prev_gray=None,None,None,None
    num0=0
    while True:
        if num0>=len_frame:
            break
        frame=videocapture[num0]
        num0+=1
        if frame_num==0:
            image=np.zeros_like(frame)
            gray=np.zeros_like(frame)
            prev_gray=np.zeros_like(frame)
            prev_image=frame
            prev_gray=cv2.cvtColor(prev_image,cv2.COLOR_RGB2GRAY)
            frame_num+=1 
            step_t=step
            while step_t>1:
                num0+=1
                step_t-=1
            continue

        image=frame
        gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        frame_0=prev_gray
        frame_1=gray
        dtvl1=cv2.optflow.DualTVL1OpticalFlow_create() # tvl1 algorithm used to generate the optical flows
        flowDTVL1=dtvl1.calc(frame_0,frame_1,None)
        save_flows(flowDTVL1,image,save_dir,save_dir_cls,frame_num,bound) # save flows and imges
        prev_gray=gray
        prev_image=image
        frame_num+=1
        step_t=step
        while step_t>1:
            num0+=1
            step_t-=1
    # clear_output(wait=True)


'''
Function that finds:
    The list of all video names
    The length of that list (how many videos there are)
    The list of paths with the classes
'''
def get_video_list():
    cls_names_list = os.listdir(videos_root)
    video_list=[]
    video_cls_path_list=[]
    for cls_names in os.listdir(videos_root):
        cls_path=os.path.join(videos_root,cls_names)
        for video_ in os.listdir(cls_path):
            video_cls_path = os.path.join(cls_names, video_)
            video_list.append(video_)
            video_cls_path_list.append(video_cls_path)
    return video_list,len(video_list),video_cls_path_list


#  step         | 0    | right - left 
#  bound        | 32   | 
# START_IDX (s_): The start index of extracted frames.

def parse_args():
    parser = argparse.ArgumentParser(description="densely extract the video frames and optical flows")
    parser.add_argument('--dataset',default='hmdb51',type=str,help='set the dataset name, to find the data path')
    parser.add_argument('--data_root',default='/n/zqj/video_classification/data',type=str)
    parser.add_argument('--new_dir',default='rawframes',type=str) 
    parser.add_argument('--num_workers',default=6,type=int,help='num of workers to act multi-process')
    parser.add_argument('--step',default=1,type=int,help='gap frames') # (0 for img, non-0 for flow) / step is 1, ie flow of adjacent frames
    parser.add_argument('--bound',default=20,type=int,help='set the maximum of optical flow') # maximum of optical flow (limit the maximum movement of one pixel)
    # in the code we will get the list of all the videos and then a sublist from them
    parser.add_argument('--s_',default=0,type=int,help='start id') # indicates the index in the list from which the sublist starts
    parser.add_argument('--e_',default=6766,type=int,help='end id') # indicates the index in which the sublist ends (default = 6766 or the number of videos)
    parser.add_argument('--mode',default='run',type=str,help='set \'run\' if debug done, otherwise, set debug')
    args = parser.parse_args('') # By putting '' as the argument, the basic arguments are passed
    return args


args=parse_args()
cwd = Path.cwd()
dataset= "hmdb51"
data_root = os.path.join(Path.cwd(),"data",dataset)
videos_root= os.path.join(data_root,'videos')

num_workers=args.num_workers
step=args.step
bound=args.bound
s_=args.s_
e_=args.e_
new_dir = "rawframes"
mode = "run"

#get video list
video_list,len_videos,video_cls_path_list=get_video_list()
video_list=video_list[s_:e_] #select the sublist
video_cls_path_list=video_cls_path_list[s_:e_] #select sublist of save/flow_dir

len_videos=min(e_-s_,6766-s_) # total number of videos selected

flows_dirs=[video.split('.')[0] for video in video_list] # genera una lista dei video senza il '.avi' (nome della cartella)
flows_dirs_cls =[video.split('.')[0] for video in video_cls_path_list] # genera una lista dei video senza il '.avi' per i path

if __name__ == '__main__':
    
    print('get videos list done! ')

    # Parallelizzazione
    pool = Pool(num_workers) # setta il numero di processi paralleli
    
    if mode=='run': #modalità run
        # print('Run mode activated')
        # pool.map(dense_flow,zip(video_list,flows_dirs,flows_dirs_cls,[step]*len(video_list),[bound]*len(video_list)))
        print("mapping ...")
        pool.map(dense_flow,zip(video_list,flows_dirs,flows_dirs_cls,[step]*len(video_list),[bound]*len(video_list)))
    else: #mode=='debug
        # print('Debug mode activated')
        dense_flow((video_list[0],flows_dirs[0],flows_dirs_cls[0],step,bound))
    pool.close()
    pool.join()
    # clear_output(wait=True)
    print('FINISHED')
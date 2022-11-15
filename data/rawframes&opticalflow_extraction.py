#import warnings
#warnings.filterwarnings('ignore')
#warnings.simplefilter('ignore')

# BACKUP 
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

# sys.argv = ['']

def ToImg(raw_flow,bound):
    '''
    this function scale the input pixels to 0-255 with bi-bound
    :param raw_flow: input raw pixel value (not in 0-255)
    :param bound: upper and lower bound (-bound, bound)
    :return: pixel value scale from 0 to 255
    '''
    
    flow=raw_flow
    flow[flow>bound]=bound
    flow[flow<-bound]=-bound
    flow-=-bound
    flow*=(255/float(2*bound))
    
    #flow=raw_flow
    #flow-=-bound
    #flow*=(255/float(2*bound))
    #flow[flow>= 255]=255
    #flow[flow<=0]=0
    
    return flow.astype(np.uint8)

def save_flows(flows,image,save_dir,save_dir_cls,num,bound):
    '''
    To save the optical flow images and raw images
    :param flows: contains flow_x and flow_y
    :param image: raw image
    :param save_dir: save_dir name (always equal to the video id)
    :param save_dir_cls: path of save dire name in form of "class//video_name"
    :param num: the save id, which belongs one of the extracted frames
    :param bound: set the bi-bound to flow images
    :return: return 0
    '''
    
    #rescale to 0~255 with the bound setting
    flow_x=ToImg(flows[...,0],bound)
    flow_y=ToImg(flows[...,1],bound)
    # print(sum(n < 0 for n in flow_x)) # check valori
    if not os.path.exists(os.path.join(data_root,new_dir,save_dir_cls)): #new_dir è il nome della cartella di salvataggio dell'output all'interno di 'data_root', di default è videos
        os.makedirs(os.path.join(data_root,new_dir,save_dir_cls))

    #save the image
    save_img=os.path.join(data_root,new_dir,save_dir_cls,'img_{:05d}.jpg'.format(num)) # prefisso immagini: img_
    # scipy.misc.imsave(save_img,image)
    # cv2.imwrite(save_img, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  
    imageio.imwrite(save_img,image)  

    #save the flows
    save_x=os.path.join(data_root,new_dir,save_dir_cls,'flow_x_{:05d}.jpg'.format(num)) # prefisso optical flow x: flow_x_
    save_y=os.path.join(data_root,new_dir,save_dir_cls,'flow_y_{:05d}.jpg'.format(num)) # prefisso optical flow y: flow_y_
    flow_x_img=Image.fromarray(flow_x) # AGGIUNTO QUESTO PER I PROBLEMI DI CONVERSIONE .astype(np.uint8)
    flow_y_img=Image.fromarray(flow_y)
    imageio.imwrite(save_x,flow_x_img)
    imageio.imwrite(save_y,flow_y_img)
    #io.imsave(save_x,flow_x)
    #io.imsave(save_y,flow_y)
    return 0

def dense_flow(augs):
    '''
    To extract dense_flow images
    :param augs:the detailed augments:
        video_name: the video name which is like: 'v_xxxxxxx',if different ,please have a modify.
        save_dir: the destination path's final direction name. video_name (without .ext)
        save_dir_cls: added by me. class_name//video_name (without .ext)
        step: num of frames between each two extracted frames
        bound: bi-bound parameter
    :return: no returns
    '''
    #print('Dense Flow function activated')
    video_name,save_dir,save_dir_cls,step,bound=augs #flow_dir viene passato come save_dir (essenzialmente è il nome del video senza avi)
    # video_path=os.path.join(videos_root,video_name.split('_')[1],video_name) # CAMBIATO TOLTO LO SPLIT
    video_path=os.path.join(videos_root,save_dir_cls + '.avi') 
    # provide two video-read methods: cv2.VideoCapture() and skvideo.io.vread(), both of which need ffmpeg support

    # videocapture=cv2.VideoCapture(video_path)
    # if not videocapture.isOpened():
    #     print 'Could not initialize capturing! ', video_name
    #     exit()
    
    # print("Parsed argouments:")
    # print(args)
    # print(f'Videos root: {videos_root}')
    
    try:
        videocapture=skvideo.io.vread(video_path)
    except:
        print('{} read error! ').format(video_name)
        return 0
    # if extract nothing, exit!
    if videocapture.sum()==0:
        print(f'Could not initialize capturing {video_name}.')
        exit()
    print(f'Inizialize capturing of {video_name}.')
    len_frame=len(videocapture)-1
    frame_num=0
    image,prev_image,gray,prev_gray=None,None,None,None
    num0=0
    while True:
        #frame=videocapture.read()
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
            frame_num+=1 # TOLTO PERCHÈ AVVIENE ANCHE DOPO QUESTO AUMENTO, CAUSA PROBLEMI PERCHÈ PARTE DA FRAME 2 così
            # to pass the out of stepped frames
            step_t=step
            while step_t>1:
                #frame=videocapture.read()
                num0+=1
                step_t-=1
            continue

        image=frame
        gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        frame_0=prev_gray
        frame_1=gray
        ##default choose the tvl1 algorithm
        # dtvl1=cv2.optflow.createOptFlow_DualTVL1() # vecchia funzione
        dtvl1=cv2.optflow.DualTVL1OpticalFlow_create()
        flowDTVL1=dtvl1.calc(frame_0,frame_1,None)
        # print(f'Salvataggio frame {frame_num}')
        save_flows(flowDTVL1,image,save_dir,save_dir_cls,frame_num,bound) #this is to save flows and img.
        prev_gray=gray
        prev_image=image
        frame_num+=1
        # to pass the out of stepped frames
        step_t=step
        while step_t>1:
            #frame=videocapture.read()
            num0+=1
            step_t-=1
    # clear_output(wait=True)


'''
Funzione che trova 1) La lista di tutti i nomi dei video 2) La lunghezza di tale lista (quanti video ci sono)
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
    #video_list.sort()
    return video_list,len(video_list),video_cls_path_list


#  step         | 0    | right - left (0 for img, non-0 for flow) / step is 1, ie flow of adjacent frames
#  bound        | 32   | maximum of optical flow (limit the maximum movement of one pixel. It's an optional setting.)
# START_IDX (s_): The start index of extracted frames.

def parse_args():
    parser = argparse.ArgumentParser(description="densely extract the video frames and optical flows")
    parser.add_argument('--dataset',default='hmdb51',type=str,help='set the dataset name, to find the data path')
    parser.add_argument('--data_root',default='/n/zqj/video_classification/data',type=str)
    parser.add_argument('--new_dir',default='rawframes',type=str) # cambiato il default prima era 'flows'
    parser.add_argument('--num_workers',default=6,type=int,help='num of workers to act multi-process')
    parser.add_argument('--step',default=1,type=int,help='gap frames')
    parser.add_argument('--bound',default=20,type=int,help='set the maximum of optical flow') # cambiato il default prima era '15'
    # nel codice otterremo la lista di tutti i video e poi una sottolista da essi
    parser.add_argument('--s_',default=0,type=int,help='start id') # indica l'indice nella lista da cui parte la sottolista
    parser.add_argument('--e_',default=6766,type=int,help='end id') # indica l'indice in cui finisce la sottolista (default = 6766 ovvero il numero di video)
    parser.add_argument('--mode',default='run',type=str,help='set \'run\' if debug done, otherwise, set debug')
    args = parser.parse_args('') # mettendo come argomento le '' vengono passati gli argomenti base
    return args


args=parse_args()
cwd = Path.cwd()
dataset= "hmdb51"
data_root = os.path.join(Path.cwd(),"data",dataset)
videos_root= os.path.join(data_root,'videos')

#specify the augments
num_workers=args.num_workers
step=args.step
bound=args.bound
s_=args.s_
e_=args.e_
#new_dir=args.new_dir
new_dir = "rawframes"
#mode= args.mode
mode = "run"

#print("Parsed argouments:")
#print(args)
#print(f'Videos root: {videos_root}')

#get video list
video_list,len_videos,video_cls_path_list=get_video_list()
video_list=video_list[s_:e_] #seleziona la sottolista
video_cls_path_list=video_cls_path_list[s_:e_] #seleziona sottolista di save/flow_dir

len_videos=min(e_-s_,6766-s_) # numero totale di video selezionati

# print(f'find {len_videos} videos.')
#print('find {} videos.').format(len_videos)
flows_dirs=[video.split('.')[0] for video in video_list] # genera una lista dei video senza il '.avi' (nome della cartella)
flows_dirs_cls =[video.split('.')[0] for video in video_cls_path_list] # genera una lista dei video senza il '.avi' per i path

if __name__ == '__main__':

    # example: if the data path not setted from args,just manually set them as belows.
    #dataset='ucf101'
    #data_root='/S2/MI/zqj/video_classification/data'
    #data_root=os.path.join(data_root,dataset)
    
    print('get videos list done! ')

    # Parallelizzazione
    pool = Pool(num_workers) # setta il numero di processi paralleli
    #print([len(video_list),len(flows_dirs),len(flows_dirs_cls),len([step]*len(video_list)),len([bound]*len(video_list))])
    
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
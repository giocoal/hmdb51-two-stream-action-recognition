# SETUP
import csv
import fnmatch
import glob
import json
import os
import os.path as osp

import argparse
import random

import numpy as np
import os
import glob
import fnmatch
import csv
import random

# Functions

def parse_directory(path, key_func=lambda x: x[-11:],
                    rgb_prefix='img_',
                    flow_x_prefix='flow_x_',
                    flow_y_prefix='flow_y_',
                    level=1):

    print('parse frames under folder {}'.format(path))
    if level == 1:
        frame_folders = glob.glob(os.path.join(path, '*'))
    elif level == 2:
        frame_folders = glob.glob(os.path.join(path, '*', '*'))
    else:
        raise ValueError('level can be only 1 or 2')

    def count_files(directory, prefix_list):
        lst = os.listdir(directory)
        cnt_list = [len(fnmatch.filter(lst, x+'*')) for x in prefix_list]
        return cnt_list

    # check RGB
    frame_dict = {}
    for i, f in enumerate(frame_folders):
        all_cnt = count_files(f, (rgb_prefix, flow_x_prefix, flow_y_prefix))
        k = key_func(f)

        x_cnt = all_cnt[1]
        #print(all_cnt[1])
        #print(all_cnt[2])
        y_cnt = all_cnt[2]
        if x_cnt != y_cnt:
            raise ValueError(
                'x and y direction have different number '
                'of flow images. video: ' + f)
        if i % 200 == 0:
            print('{} videos parsed'.format(i))

        frame_dict[k] = (f, all_cnt[0], x_cnt)

    print('frame folder analysis done')
    return frame_dict

def mimic_ucf101(frame_path, anno_dir):

    classes_list = os.listdir(frame_path) # legge i nomi delle cartelle in rawframes (classi)
    classes_list.sort() # ordina in ordine alfabetico 

    classDict = {}
    classIndFile = os.path.join(anno_dir, 'classInd.txt')
    with open(classIndFile, 'w') as f:
        for class_id, class_name in enumerate(classes_list):
            classDict[class_name] = class_id
            cur_line = str(class_id + 1) + ' ' + class_name + '\n'
            f.write(cur_line)


    for split_id in range(1, 4):
        splitTrainFile = os.path.join(anno_dir, 'trainlist%02d.txt' % (split_id))
        with open(splitTrainFile, 'w') as target_train_f:
            for class_name in classDict.keys():
                fname = class_name + '_test_split%d.txt' % (split_id)
                fname_path = os.path.join(anno_dir, fname)
                source_f = open(fname_path, 'r')
                source_info = source_f.readlines()
                for _, source_line in enumerate(source_info):
                    cur_info = source_line.split(' ')
                    video_name = cur_info[0]
                    if cur_info[1] == '1':
                        target_line = class_name + '\\' + video_name + ' ' + str(classDict[class_name] + 1) + '\n'
                        target_train_f.write(target_line)

        splitTestFile = os.path.join(anno_dir, 'testlist%02d.txt' % (split_id))
        with open(splitTestFile, 'w') as target_test_f:
            for class_name in classDict.keys():
                fname = class_name + '_test_split%d.txt' % (split_id)
                fname_path = os.path.join(anno_dir, fname)
                source_f = open(fname_path, 'r')
                source_info = source_f.readlines()
                for _, source_line in enumerate(source_info):
                    cur_info = source_line.split(' ')
                    video_name = cur_info[0]
                    if cur_info[1] == '2':
                        target_line = class_name + '\\' + video_name + ' ' + str(classDict[class_name] + 1) + '\n'
                        target_test_f.write(target_line)

def parse_hmdb51_splits(level):
    
    mimic_ucf101('data/hmdb51/rawframes', 'data/hmdb51/annotations')

    class_ind = [x.strip().split()
                 for x in open('data/hmdb51/annotations/classInd.txt')] #check di quali classi sono presenti in rawframes
    print(class_ind) 

    class_mapping = {x[1]: int(x[0]) - 1 for x in class_ind} # il -1 qui fa diminuire l'indice
                                                             # 0 diventa train / 1 validation (test) / 2 nulla (chiamato test)
    def line2rec(line):
        items = line.strip().split(' ')
        vid = items[0].split('.')[0]
        vid = "data\\hmdb51\\rawframes\\" + '\\'.join(vid.split('\\')[-level:])
        label = class_mapping[items[0].split('\\')[0]]
        return vid, label

    splits = []
    for i in range(1, 4):
        train_list = [line2rec(x) for x in open(
            'data/hmdb51/annotations/trainlist{:02d}.txt'.format(i))]
        test_list = [line2rec(x) for x in open(
            'data/hmdb51/annotations/testlist{:02d}.txt'.format(i))]
        splits.append((train_list, test_list))
    #print(splits)
    return splits
def build_split_list(split, frame_info, shuffle=False):
    def build_set_list(set_list):
        rgb_list, flow_list = list(), list()
        for item in set_list:
            if item[0] not in frame_info: # qui checka che i video nella lista siano nel dizionario dei video
                                          # probabilmente qui c'è un problema con gli slash e per quello non vede
                                          # che item[0] ha dei video nella lista
                continue
            elif frame_info[item[0]][1] > 0: # check che ci sia almeno un frame
                print('OK')
                rgb_cnt = frame_info[item[0]][1] #conteggio frame RGB
                flow_cnt = frame_info[item[0]][2]
                rgb_list.append('{} {} {}\n'.format( 
                    item[0], rgb_cnt, item[1])) #lista dei frame (immagini)
                flow_list.append('{} {} {}\n'.format(
                    item[0], flow_cnt, item[1]))
            else:
                print('OK')
                rgb_list.append('{} {}\n'.format(
                    item[0], item[1]))
                flow_list.append('{} {}\n'.format(
                    item[0], item[1]))
        if shuffle:
            random.shuffle(rgb_list)
            random.shuffle(flow_list)
        return rgb_list, flow_list

    train_rgb_list, train_flow_list = build_set_list(split[0])
    test_rgb_list, test_flow_list = build_set_list(split[1])
    return (train_rgb_list, test_rgb_list), (train_flow_list, test_flow_list)

def parse_args():
    parser = argparse.ArgumentParser(description='Build file list')
    parser.add_argument('--dataset', type=str, default= 'hmdb51')
    parser.add_argument('--frame_path', type=str,
                        help='root directory for the frames', default ="data\\hmdb51\\rawframes\\")
    parser.add_argument('--rgb_prefix', type=str, default='img_')
    parser.add_argument('--flow_x_prefix', type=str, default='flow_x_')
    parser.add_argument('--flow_y_prefix', type=str, default='flow_y_')
    parser.add_argument('--num_split', type=int, default=3)
    parser.add_argument('--subset', type=str, default='train',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--level', type=int, default=2, choices=[1, 2])
    parser.add_argument('--format', type=str,
                        default='rawframes', choices=['rawframes', 'videos'])
    parser.add_argument('--out_list_path', type=str, default='data/')
    parser.add_argument('--shuffle', action='store_true', default=False)
    args_ = parser.parse_args('')
    return args_

args = parse_args()

if args.level == 2:
    def key_func(x): return '/'.join(x.split('/')[-2:])
else:
    def key_func(x): return x.split('/')[-1]

if args.format == 'rawframes':
    frame_info = parse_directory(args.frame_path,
                                    key_func=key_func,
                                    rgb_prefix=args.rgb_prefix,
                                    flow_x_prefix=args.flow_x_prefix,
                                    flow_y_prefix=args.flow_y_prefix,
                                    level=args.level)
else:
    if args.level == 1:
        video_list = glob.glob(osp.join(args.frame_path, '*'))
    elif args.level == 2:
        video_list = glob.glob(osp.join(args.frame_path, '*', '*'))
    frame_info = {osp.relpath(
        x.split('.')[0], args.frame_path): (x, -1, -1) for x in video_list}

if args.dataset == 'hmdb51':
    split_tp = parse_hmdb51_splits(args.level)
assert len(split_tp) == args.num_split

out_path = args.out_list_path + args.dataset
if len(split_tp) > 1:
    for i, split in enumerate(split_tp):
        lists = build_split_list(split_tp[i], frame_info,
                                    shuffle=args.shuffle)
        filename = '{}_train_split_{}_{}.txt'.format(args.dataset,
                                                        i + 1, args.format)
        with open(osp.join(out_path, filename), 'w') as f:
            f.writelines(lists[0][0])
        filename = '{}_val_split_{}_{}.txt'.format(args.dataset,
                                                    i + 1, args.format)
        with open(osp.join(out_path, filename), 'w') as f:
            f.writelines(lists[0][1])
else:
    lists = build_split_list(split_tp[0], frame_info,
                                shuffle=args.shuffle)
    filename = '{}_{}_list_{}.txt'.format(args.dataset,
                                            args.subset,
                                            args.format)
    if args.subset == 'train':
        ind = 0
    elif args.subset == 'val':
        ind = 1
    elif args.subset == 'test':
        ind = 2
    with open(osp.join(out_path, filename), 'w') as f:
        #print(list[0][ind])
        f.writelines(lists[0][ind])

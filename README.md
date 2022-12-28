# VIDEO CLASSIFICATION: HUMAN ACTION RECOGNITION ON HMDB51 DATASET

We use spatial (ResNet-50 finetuned) and temporal stream cnn (stacked Optical Flows) under the Keras framework to perform Video-Based Human Action Recognition on HMDB-51 dataset.

- Check out our [pdf report](https://github.com/giocoal/hmdb51-two-stream-action-recognition/blob/main/Report/Deep%20Learning%20-%20Video%20Action%20Recognition.pdf) 

## References

*  [[1] Two-stream convolutional networks for action recognition in videos](http://papers.nips.cc/paper/5353-two-stream-convolutional)

*  [[2] HMDB: A Large Video Database for Human Motion Recognition](https://serre-lab.clps.brown.edu/wp-content/uploads/2012/08/Kuehne_etal_iccv11.pdf)

*  [[3] A duality based approach for realtime tv-l 1 optical flow](https://www-pequan.lip6.fr/~bereziat/cours/master/vision/papers/zach07.pdf)

## Abstract

HMDB51 is a large database of videos for human action recognition, which is the main task of our project. 
- Official website: https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/

By Human Action Recognition (HAR) we identify that task which involves the classification of one or more actions performed by one or more human subjects. The strong advancement of computer vision and the large amount of video available, have made video classification methods the most common approach to tthe problem.
The specific task was Simple Action Recognition, the training of a model that classifies a single global action (performed by one or more subjects) associated with a short video input.

HMDB-51 this contains 51 distinct action categories, each associated with at least 101 clips, for a total of 6766 annotated clips, extracted mainly from movies and youtube videos. The classes of actions can be grouped into: 1) general facial actions such as smiling 2) facial actions with object manipulation (such as eating) 3) general body movements (such as jumping) 4) body movements involving interaction with objects (such as combing) and 5) finally body movements with human interaction such as (such as hugging).

The data preparation process that preceded the training of the models first involved 1) a split of the dataset into training and test set using the split suggested by the authors which ensures that a) clips from the same video are not present in both the train and test set b) that for each class there are exactly 70 train and 30 test clips and c) and that there is maximization of the relative proportion balance of the meta tags.
2) Frame extraction and optical flow estimation follow. By optical flow we define the apperent motion pattern of objects in a scene between two consecutive frames caused by object or camera motion that coincides with a field of two-dimensional point-shift vectors between one frame and the next. Dual TV-L1, an algorithm for dense optical flows (which then computes the motion vector for each pixel), found in the OpenCV library, was used for estimation, and for each flow the horizontal and vertical component.

## Requirements

- python 3.10.7
- ipython 8.6.0
- imageio 2.22.4
- keras 2.11.0
- matplotlib 3.6.2
- numpy 1.22.1
- pandas 1.3.5
- Pillow 9.3.0
- scipy 1.9.3
- tensorflow 2.10.0
- tqdm 4.64.1
- opencv 4.5.1

## Preparing HMDB-51

### Step 0. Prepare Folders

First of all, create three empty folders: `data/hmdb51/videos`,`data/hmdb51/rawframes` and `Models`.

### Step 1. Prepare Annotations (training/test splits)

Download annotations from the [official website](http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar) and extract them in `data/hmdb51/annotations`

### Step 2. Prepare Videos
Download and extract the dataset from HMDB-51 into the `data/hmdb51/videos` folder:
  * [Download page](http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar)

### Step 3. Extract or Download RGB and Flow Frames (requires CUDA-enabled Open-CV)

#### Option 1: generate them with the dedicated script
If you have plenty of SSD space, then we recommend extracting frames there for better I/O performance.
Run the following script: `data/rawframes&opticalflow_extraction.py`

#### Option 2: download the pre-generated frames
1. Download the .zip file from [Google Drive](https://drive.google.com/drive/folders/1qy5_ukO7_e-XftKfSCjjLkO_Vv2Peqis?usp=sharing)
2. Extract the .zip file content in `data/hmdb51/rawframes`
  
### Step 4. Generate File List

Generate file list in the format of rawframes and video (training and test video lists and path)

Run the following script: `data/video_train_test_split_list_generator.py`

## Model

### Data augmentation

*  Both streams apply the same data augmentation technique such as random cropping and random horizontal flipping. Temporally, we pick the starting frame among those early enough to guarantee a desired number of frames. 

### Spatial-stream cnn (finetuned ResNet-50)

* As mention before, we use ResNet-50 first pre-trained with ImageNet then fine-tuning on our HMDB-51 spatial rgb image dataset. 

* 17 equidistant frames are sampled from a video. 

* Run the following script: `spatial_stream_cnn_finetuned.ipynb`

### Temporal-stream cnn

*  We train the temporal-stream cnn from scratch. In every mini-batch, we randomly select 128 (batch size) videos from training videos and futher randomly select 1 optical flow stack in each video. We follow the reference paper and use 10 x-channels and 10 y-channels for each optical flow stack, resulting in a input shape of (224, 224, 20). 

* Input data of motion cnn is a stack of optical flow images which contained 10 x-channel and 10 y-channel images, So it's input shape is (20, 224, 224) which can be considered as a 20-channel image.

* Multiple workers are utilized in the data generator for faster training.

* Run the following script: `motion_stream.ipynb` 

## Testing

*  We fused the two streams by averaging the softmax scores.

Two evaluation strategies were used:

* [1] **"One frame per video" method** For each batch element, a random video was selected, and for each video/element a single frame is selected and given as input to the spatial stream. The selected frame is also used as the initial frame to obtain the stacked optical flows of 10 consecutive frames, which is given as input to the temporal stream. Softmax scores are averaged. The prediction is compared with the label.
    * Run the following script: `two_stream_fusion_validate.ipynb` 

* [2]  **"Whole video" method**: An arbitrary video is selected. All frames of the video are given as input to the spatial stream, the stream prediction is obtained as the average of the probabilities of all frames. All possible stacked optical flows of the video (the highest frame that can be used as a start frame is total_frame - 10) are given as input to the motion stream, the stream prediction is obtained as the average of the probabilities of all stacked optical flows. The final prediction is obtained as the average of the probabilities of the two streams.
    * Run the following script: `two_stream_fusion_validate_full_video.ipynb` 

## Results
|Network     | One frame | Whole video  |
-------------|:--------------:|:----:|
|Spatial ResNet-50     |39.0%           |45.0% |
|Temporal    |25.8%           |NA% |
|Fusion      |38.0%           |NA% |

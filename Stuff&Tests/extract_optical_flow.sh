#!/usr/bin/env bash

SRC_FOLDER= ""
OUT_FOLDER= ""
NUM_WORKER= $1

echo "Extracting optical flow from videos in folder: C:\\Users\\giorg\\OneDrive - Università degli Studi di Milano-Bicocca\\Laurea Magistrale - Data Science\\directory_progetti\\deep-learning-video-classification\\data\\hmdb51\\videos\\"
"C:\\Users\\giorg\\Documents\\venv\\deepL37\\Scripts\\python.exe" build_of.py "C:\\Users\\giorg\\OneDrive - Università degli Studi di Milano-Bicocca\\Laurea Magistrale - Data Science\\directory_progetti\\deep-learning-video-classification\\data\\hmdb51\\videos\\" "C:\\Users\\giorg\\OneDrive - Università degli Studi di Milano-Bicocca\\Laurea Magistrale - Data Science\\directory_progetti\\deep-learning-video-classification\\data\\hmdb51\\rawframes\\" --num_worker 1 --num_gpu 1
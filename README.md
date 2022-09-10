# VIDEO CLASSIFICATION: HUMAN ACTION RECOGNITION ON HMDB51 DATASET

We use spatial (ResNet-50 Finetuned) and temporal stream cnn under the Keras framework to perform Video-Based Human Action Recognition on HMDB-51 dataset.

Report pdf: [https://github.com/giocoal/hmdb51-two-stream-action-recognition/blob/main/Report/Deep%20Learning%20-%20Video%20Action%20Recognition.pdf]

## References

*  [[1] Two-stream convolutional networks for action recognition in videos](http://papers.nips.cc/paper/5353-two-stream-convolutional)

*  [[2] HMDB: A Large Video Database for Human Motion Recognition](https://serre-lab.clps.brown.edu/wp-content/uploads/2012/08/Kuehne_etal_iccv11.pdf)

## Abstract

HMDB51 is a large database of videos for human action recognition, which is the main task of our project. 
- Official website: https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/

By Human Action Recognition (HAR) we identify that task which involves the classification of one or more actions performed by one or more human subjects. The strong advancement of computer vision and the large amount of video available, have made video classification methods the most common approach to tthe problem.
The specific task was Simple Action Recognition, the training of a model that classifies a single global action (performed by one or more subjects) associated with a short video input.

HMDB-51 this contains 51 distinct action categories, each associated with at least 101 clips, for a total of 6766 annotated clips, extracted mainly from movies and youtube videos. The classes of actions can be grouped into: 1) general facial actions such as smiling 2) facial actions with object manipulation (such as eating) 3) general body movements (such as jumping) 4) body movements involving interaction with objects (such as combing) and 5) finally body movements with human interaction such as (such as hugging).

The data preparation process that preceded the training of the models first involved 1) a split of the dataset into training and test set using the split suggested by the authors which ensures that a) clips from the same video are not present in both the train and test set b) that for each class there are exactly 70 train and 30 test clips and c) and that there is maximization of the relative proportion balance of the meta tags.
2) Frame extraction and optical flow estimation follow. By optical flow we define the apperent motion pattern of objects in a scene between two consecutive frames caused by object or camera motion that coincides with a field of two-dimensional point-shift vectors between one frame and the next. Dual TV-L1, an algorithm for dense optical flows (which then computes the motion vector for each pixel), found in the OpenCV library, was used for estimation, and for each flow the horizontal and vertical component.

### 1. Data Collection

1. Request and add Twitch API keys to the file `Twitch_API_keys.txt`
2. Create a repeated execution task for `Twitch_stream_collection.py` every xx minutes (Win: Task Scheduler, Linux: Crontab)
    - choose the language of the desired streams
    - this script saves the collected stream files in individual json files but it's already supported the upload on MongoDB local server, uncomment the import function in the script (it requires [MongoDB Community Server](https://www.mongodb.com/try/download/community))
3. Run `steam_games_scraping.ipynb` to scrape [SteamDB](https://steamdb.info/graph/) website (if the website asks CAPTCHA clean browser cookies)
4. Download bot dataset from [Twitch Insights](https://twitchinsights.net/bots) using a browser extension (e.g. Table Capture for Chrome) and save it as `Twitch_bot_list.csv`
5. Run `Twitch_social_link.py` to obtain the streamer's social link (this can be run only after Data Processing because it needs the complete streamer list)

### 2. Data Processing

1. Run `DataProcessing.ipynb` selecting the parameters for the analysis in the first block:
    - data source (json files or MongoDB local server)
    - set the time interval acquisition (xx minutes)
    - set parameters and thresholds
2. Run `DataEnrichment.ipynb` to add games info from SteamDB (verify manually the matches)
3. Run `DataExploration.ipynb` and `DataQuality.ipynb` to obtain data insights

### 3. Data Modelling

1. Install [Neo4j Community Server](https://neo4j.com/download-center/#community)
2. Copy the CSVs obtained from the `output_datasets` folder to the neo4j import folder (`neo4j/import/`)
3. Run `graph_neo4j.ipynb` to load data in Neo4j
4. Execute desired queries

### 4. Data Visualization

1. Install Gephi
2. Import `Streamer_dataset_short.csv` and `Streamer-Streamer_dataset_short.csv`
3. Execute some layout algorithms (e.g. Atlas Force), execute statistics analysis to detect communities (e.g. Modularity), edit nodes and edges colors (more details [here](https://github.com/KiranGershenfeld/VisualizingTwitchCommunities))


For additional info on the project read `ProjectReport_ita.pdf` (in Italian)

![Graph visualization on Gephi May 2022](https://github.com/gianscuri/Twitch_Community_Graph/blob/main/DataVisualization/Images/Gephi_graph_dark.png)

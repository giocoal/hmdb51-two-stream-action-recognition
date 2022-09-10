# Twitch community graph

## Abstract

[Twitch.tv](https://www.twitch.tv/) is a **live streaming platform** that allows streamers to broadcast and users to enjoy content in real time. The broadcasts cover various categories related mainly to the world of videogames, entertainment, and the arts.
Thanks to its great success, especially in the last few years, both the revenue opportunities for streamers and companies operating in these sectors have increased.
**Understanding the market and the platform**, however, is crucial to discover the interests of users.
This project therefore aims to **collect and analyze data** about the different streams in order to create an explorable and queryable **graph model** of the communities present thus enabling accurate market analysis.

The project consists in a series of scripts to collect, integrate, analyze and save data from different sources. It is thus a tool that can be run in any time frame to obtain the up-to-date graph of the situation.
The data collection phase is done from **two distinct data sources**: [Twitch](https://www.twitch.tv/) for live information through the use of the official **Web APIs** and from [SteamDB](https://steamdb.info/graph/) for videogames informations through **dynamic scraping techniques**. In the processing phase, the datasets containing the streamers, the different video games streamed, and the related bridge-tables that allow them to be linked are then obtained. The **streamer-game relations** were calculated by analyzing the broadcast categories, while the **streamer-streamer relations** were calculated by evaluating the percentage of common viewers between each pair of streamers.

This repository contains data collected over a **two-week period in May 2022 regarding all Italian broadcasts** on Twitch and data from SteamDB regarding the most played videogames. Approximately 2.5GB of data were collected during this period, which after a detailed analysis allowed the creation of a graph model on the [Neo4j DBMS](https://neo4j.com/) consisting of **4121 nodes and 54931 edges**.

## Execution scheme

![Pipeline](https://github.com/gianscuri/Twitch_Community_Graph/blob/main/DataVisualization/Images/pipeline.png)

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

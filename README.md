# Fuzzy Logic Neural Interface Project 

## Description
This project was born out of the fuzzy sets course given by professor protman over at the university of fribourg switzerland. 

The objective is two fold. 
1. Discover ways of using fuzzy sets to interpret brain signals from a Muse 2 eeg. 
2. Evaluate them against traditional methods. By reusing old reinforcement learning enviroments to play games with your mind. (wip)

All done with a Muse 2 eeg headband. loaned out by the kind people of the [HUMAN-IST INSTITUTE](https://human-ist.unifr.ch/en/)
## Brain Wave Model Training 

To hit the ground running and train your own models do the following.

### Training Installation And Usage

1. Install the dependencies nessary for training  `pip install -r training_requirements.txt` 
2. Get the dataset. (check the read me in the data folder)
3. Have fun running the notebooks in model training.
.

#### After running these notebooks you should see a set of .h5 files populate the models folder. For you to import and intergate into your projects when needed.

## Brainflow Streamer

To test if your eeg will work with our code run the following.

### Brainflow Streamer Installation 

1. Install the dependencies nessary for brainflow streaming `pip install -r brainflow_requirements.txt` 
2. Run brainflow controller script `python brainflow_controller_prototype.py --board-id 39`

The number one thing you need is the id of your eeg. For the muse 2 this is 39. For others you need to check with the [brainflows documentation]([myLib/README.md](https://brainflow.readthedocs.io/en/stable/SupportedBoards.html))  for the supported eegs.

Do note, inspite of what Muse claims. You don't need a special dongle to run with a Muse 2 headset.  

#### What you should see is an animated graph of the 5 basic brainwaves.  




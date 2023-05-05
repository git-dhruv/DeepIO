# Installation

Run the following commands for setting up the repo
```
git clone https://github.com/git-dhruv/DeepIO.git
git submodule update --init --recursive
```

We have provided a conda.yml file and a requirements.txt for the project. 

# Dataset Download
Follow the instructions from [https://github.com/mit-aera/Blackbird-Dataset] (Blackbird) to download the dataset. 
Save the new data in the data directory. 

# Running the pipeline
Run the following pipeline
```
python3 src/DeepAEKF.py
```

import numpy as np
import pandas as pd
import os, sys

from bagpy import bagreader
import logging

class dataloader:

    def __init__(self,location):
        self.folder = os.path.abspath(os.path.expanduser(os.path.expandvars(location)))
        self.cases = os.listdir(self.folder)
        self.setupLogging()        

    def setupLogging(self):
        log_format = "%(filename)s::%(lineno)d::%(message)s"
        logging.basicConfig(level='DEBUG', format=log_format)
    
    def loadCase(self, case):
        for folder in os.listdir(os.path.join(self.folder,case)):
            logging.info(f"Parsing {folder}")
            bagfile = os.path.join(folder,"rosbag.bag")
            
    def runPipeline(self):
        for case in self.cases:
            self.loadCase(case)
        


if __name__ == "__main__":
    tmp = dataloader("../data/clover")
    tmp.runPipeline()
        

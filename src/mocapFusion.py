"""
@author: Anirudh Kailaje, Dhruv Parikh
@date: 4/24/2023
@Description: Not Defined
"""
import numpy as np
import matplotlib.pyplot as plt
from dataloader import *

class OnlineLearingFusion:
    def __init__(self):
        """
        Standard Multi Sensor Fusion Parameters
        """
        self.state = np.zeros(16,1)
        self.covariance = np.zeros(15,15)
        self.Q = np.zeros(7,7)
        self.R = np.zeros_like(self.covariance)

        self.PropogationJacobian = None
        self.MeasurmentJacobian = None
        
    def propogateStep(self):
        raise NotImplementedError
    def measurmentStep(self):
        raise NotImplementedError
    
    def runPipeline(self):
        raise NotImplementedError
    
    def plotSampleOutput(self):
        raise NotImplementedError
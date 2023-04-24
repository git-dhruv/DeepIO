"""
@author: Anirudh Kailaje, Dhruv Parikh
@date: 4/24/2023
@Description: Not Defined
"""
import numpy as np
import matplotlib.pyplot as plt
from dataloader import *
from dynamicsSim import *

class OnlineLearingFusion:
    def __init__(self):
        """
        Standard Multi Sensor Fusion Parameters
        """
        self.state = np.zeros((16,1))
        self.covariance = np.zeros((16,16))
        self.Q = np.zeros((7,7))
        self.R = np.zeros_like(self.covariance)

        self.PropogationJacobian = None
        self.MeasurmentJacobian = None

        self.dynamics = dynamics()


        """        
        a = [(q0^2 - q1^2 - q2^2 - q3^2)ax + 2(q0q1 - q2q3)ay + 2(q0q2 + q1q3)az,
        2(q0q1 + q2q3)ax + (q0^2 - q1^2 + q2^2 - q3^2)ay + 2(q1q2 - q0q3)az,
        2(q0q2 - q1q3)ax + 2(q1q2 + q0*q3)*ay + (q0^2 - q1^2 - q2^2 + q3^2)*az]
        """
    def calcJacobian(self, dt, measurment=1):
        if measurment:
            # self.MeasurmentJacobian = 
            return self.MeasurmentJacobian
        else:
            self.PropogationJacobian = np.array([[],
                                                 [],
                                                 [],
                                                 [],
                                                 [],
                                                 [],
                                                 [],
                                                 []])
            return self.PropogationJacobian
        
    def propogateStep(self, state,rpm,dt):
        self.calcJacobian(dt,measurment=0)
        self.state = self.dynamics.propogateDynamics(state,rpm, dt)
        self.covariance = self.PropogationJacobian@self.covariance@self.PropogationJacobian.T + self.R

    def measurmentStep(self, measurments, dt):
        G = self.calcJacobian(dt,measurment=1)
        
    
    def runPipeline(self):
        #1. Load the Data
        loadDataUtil = dataloader("../data/clover")
        loadDataUtil.runPipeline()
        loadDataUtil.homogenizeData()
        data = loadDataUtil.ConcatData
        #R from imu to ned Frame
        R_imu_to_ned = np.array([[-1,0,0],
                                [0,-1,0],
                                [0,0,1]])
        #2. Run through the Loop
            #3. Propogate Step
            #4. Measurement Update
        #Return all the state vectors
        raise NotImplementedError
    
    def plotSampleOutput(self):
        raise NotImplementedError
if __name__ == '__main__':
    learner = OnlineLearingFusion()
    learner.runPipeline()
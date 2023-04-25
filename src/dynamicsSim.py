import numpy as np
import pandas as pd
from dataloader import *
from scipy.spatial.transform import Rotation


class dynamics:
    def __init__(self, thrustCoef=1.91e-06, torqueCoef=2.6e-7, l=0.13, ixx=4.9*10e-2, iyy=4.9*10e-2, izz=6.9*10e-2, mass=0.915):
        self.m = mass
        self.ixx, self.iyy, self.izz = ixx, iyy, izz
        self.length = l
        self.thrustCoef = 2.27e-8  # thrustCoef
        self.torqueCoef = torqueCoef
        self.G = 9.80665
        # thrustCoef: 1.91e-06
        # torqueCoef: 2.6e-7
        # nRotors: 4
        # mass: 0.873
        # Ixx: 0.0049
        # Iyy: 0.0049
        # Izz: 0.0069
        # armLength: 0.08
        # propHeight: 0.023

    def propogateDynamics(self, x, omega, dt):
        # state vector -> [x,xdot,xddot, euler,angular velocities, biases]
        F, _ = self.rpmConversions(omega.flatten())
        R_b_w = Rotation.from_euler('xyz', x[9:12].flatten()).as_matrix()
        acc = np.array([0, 0, self.m*self.G]).reshape(-1, 1) + \
            R_b_w@np.array([0, 0, np.sum(-F)]).reshape(-1, 1)    
        x[3:6] += (acc*dt)  # Velocities
        x[:3] += x[3:6]*dt  # Position
        x[6:9] = acc  #Accelerations
        x[9:12] += x[12:15]*dt #Angles
        #Angular Velocities, Biases will remain the same
        return x

    def rpmConversions(self, omega):
        F = [i*i*self.thrustCoef for i in omega]
        M = [i*i*self.torqueCoef for i in omega]
        return np.array(F), np.array(M)

    def angularDynamics(self):
        pass

    def lateralDynamics(self):
        pass

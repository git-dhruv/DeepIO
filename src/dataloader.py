"""
@author: Dhruv Parikh
@date: 4/17/2023
@Description: Dataloading Class .. will update soon aka never
"""

import numpy as np
import pandas as pd
import os
import sys

import logging
from os.path import join
from tqdm import tqdm
from utils import *
import glob


class dataloader:

    def __init__(self, location):
        self.folder = os.path.abspath(os.path.expanduser(os.path.expandvars(location)))
        self.cases = os.listdir(self.folder)
        print(f"[LOADER] Found {len(self.cases)} Directories")

        self.relevantCSV = ['gps.csv','wheels.csv','ms25_euler.csv', 'groundtruth_2012-01-08.csv']
        self.gps_df = None      #GPS Converted Data
        self.imu_df = None       #IMU Data
        self.odo_df = None      #Odometry Data
        self.gtruth_df = None
        self.ConcatData = None

    def setupLogging(self):
        log_format = "[%(filename)s]%(lineno)d::%(message)s"
        logging.basicConfig(level='DEBUG', format=log_format)

    def loadCase(self, case):
        
        case = os.path.join(self.folder,case)
        relCSV = [os.path.normpath(os.path.join(case,i)) for i in self.relevantCSV]
        for file in glob.glob(f'{case}\\'+'*.csv'):
            file = os.path.normpath(file)
            if file in relCSV:
                self.parseCSV(file,relCSV)
    
    def parseCSV(self, filename, fileList):
        #@TODO: Dhruv - Figure out which file is it and call the relevant functions
        if filename == fileList[1]:
            self.parseWheel(filename)
        if filename == fileList[0]:
            self.parseGPS(filename)
        if filename == fileList[2]:
            self.parseEuler(filename)
        if filename == fileList[3]:
            self.parseGroundTruth(filename)


    def find_csv_filenames(self, path_to_dir, suffix=".csv" ):
        filenames = os.listdir(path_to_dir)
        return [ filename for filename in filenames if filename.endswith( suffix ) ]
    def parseGroundTruth(self, file):
        if self.gtruth_df is None:    
            self.gtruth_df = pd.read_csv(file)
        else:
            self.gtruth_df = pd.concat([self.gtruth_df, pd.read_csv(file)], ignore_index=True, axis=0)
        
    def parseWheel(self, file):
        if self.odo_df is None:
            self.odo_df = pd.read_csv(file)
        else:
            self.odo_df = pd.concat(
                [self.odo_df, pd.read_csv(file)], ignore_index=True, axis=0)

    def parseEuler(self, file):
        if self.imu_df is None:
            self.imu_df = pd.read_csv(file)
        else:
            self.imu_df = pd.concat([self.imu_df, pd.read_csv(file)], ignore_index = True, axis=0)

    def parseGPS(self, file):
        gps = pd.read_csv(file)
        gps.columns = ['Time', 'Fix', 'NumSat', 'Latitude','Longitude','Altitude', 'Track', 'Speed']
        gps.drop(['NumSat'], axis=1, inplace=True)
        if self.gps_df is None:
            self.gps_df = gps
        else:
            self.gps_df = pd.concat([self.gps_df,gps],ignore_index=True, axis=0)
        

        
        
    def homogenizeData(self):
        """
        This function will homogenize the data by interpolating the data, and creating a single dataframe. 
        Key operations: set Time as index, merge all dataframes, interpolate data with 'linear' method and fill na with the first upcoming reading.
        """
        imucopy = self.imu_data.copy()
        rotorcopy = self.rotor_data.copy()
        mocapcopy = self.mocap_data.copy()
        imucopy.set_index('Time', inplace=True)
        rotorcopy.set_index('Time', inplace=True)
        mocapcopy.set_index('Time', inplace=True)
        self.ConcatData = imucopy.join(
            rotorcopy, how='outer').join(mocapcopy, how='outer')
        self.ConcatData.interpolate(method='linear', inplace=True)
        self.ConcatData.fillna(method='bfill', inplace=True)
        self.ConcatData['Time'] = self.ConcatData.index
        self.ConcatData.reset_index(drop=True, inplace=True)

    def perturbStates(self, pos_noise=np.diag([0.0225, 0.0225, 0.0025]), orientation_noise=np.diag([0.1, 0.1, 0.1])):
        """
        This function will perturb the states by adding noise to the states.
        Inputs:
            - pos_noise: 3x3 covariance matrix for position noise
            - orientation_noise: 3x3 covariance matrix for orientation noise
        Outputs:
            - States a nx7 matrix with the following columns: Time, x, y, z, psi, theta, phi
        """
        if self.ConcatData is None:
            self.homogenizeData()

        time = self.ConcatData['Time'].to_numpy()
        pos = self.ConcatData[['pose.position.x',
                               'pose.position.y', 'pose.position.z']].to_numpy()

        axis_angles = []
        for i in range(len(self.ConcatData)):
            quat = Quaternion(scalar=self.ConcatData['pose.orientation.w'][i], vec=np.array(
                [self.ConcatData['pose.orientation.x'][i], self.ConcatData['pose.orientation.y'][i], self.ConcatData['pose.orientation.z'][i]]))
            axis_angles.append(quat.axis_angle())

        # Add noise
        pos_noise = np.random.multivariate_normal(
            np.zeros(3), pos_noise, len(self.ConcatData))
        orientation_noise = np.random.multivariate_normal(
            np.zeros(3), orientation_noise, len(self.ConcatData))
        pos = pos + pos_noise
        axis_angles = np.array(axis_angles) + orientation_noise

        perturbed_quat = []
        for i in range(len(self.ConcatData)):
            quat = Quaternion()
            quat.from_axis_angle(axis_angles[i])
            perturbed_quat.append(quat.q)
        perturbed_quat = np.array(perturbed_quat)

        data = np.vstack([time, pos[:, 0], pos[:, 1], pos[:, 2], perturbed_quat[:, 1],
                         perturbed_quat[:, 2], perturbed_quat[:, 3], perturbed_quat[:, 0]]).T

        return data

    def convertDataToIndividualNumpy(self):
        """
        Elegance 
        """
        if self.ConcatData is None:
            self.homogenizeData()
        gx, gy, gz = self.ConcatData['angular_velocity.x'].to_numpy(
        ), self.ConcatData['angular_velocity.y'].to_numpy(), self.ConcatData['angular_velocity.z'].to_numpy()
        ax, ay, az = self.ConcatData['linear_acceleration.x'].to_numpy(
        ), self.ConcatData['linear_acceleration.y'].to_numpy(), self.ConcatData['linear_acceleration.z'].to_numpy()
        mcapx, mcapy, mcapz = self.ConcatData['pose.position.x'].to_numpy(
        ), self.ConcatData['pose.position.y'].to_numpy(), self.ConcatData['pose.position.z'].to_numpy()
        rpm0, rpm1, rpm2, rpm3 = self.ConcatData['rpm_0'].to_numpy(), self.ConcatData['rpm_1'].to_numpy(
        ), self.ConcatData['rpm_2'].to_numpy(), self.ConcatData['rpm_3'].to_numpy()
        q0, q1, q2, q3 = self.ConcatData['pose.orientation.w'].to_numpy(), self.ConcatData['pose.orientation.x'].to_numpy(
        ), self.ConcatData['pose.orientation.y'].to_numpy(), self.ConcatData['pose.orientation.z'].to_numpy()
        t = self.ConcatData['Time']
        # Bracket madness
        q = np.vstack((q1, np.vstack((q2, np.vstack((q3, q0))))))
        acc = np.vstack((ax, np.vstack((ay, az))))
        gyro = np.vstack((gx, np.vstack((gy, gz))))
        rpm = np.vstack((rpm0, np.vstack((rpm1, (np.vstack((rpm2, rpm3)))))))
        mocap = np.vstack((mcapx, np.vstack((mcapy, mcapz))))
        return gyro, acc, rpm, mocap, q, t

    def runPipeline(self):
        for case in (self.cases):
            self.loadCase(case)
    def getDataFrames(self):
        return self.gps_df,self.imu_df,self.odo_df,self.gtruth_df

if __name__ == "__main__":
    directory = r'data/'
    mn = dataloader(directory)
    mn.runPipeline()
    
    

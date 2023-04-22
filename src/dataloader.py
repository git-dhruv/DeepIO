"""
@author: Dhruv Parikh
@date: 4/17/2023
@Description: Dataloading Class .. will update soon aka never
"""

import numpy as np
import pandas as pd
import os
import sys

from bagpy import bagreader
import logging
from os.path import join
from tqdm import tqdm
from utils import *


class dataloader:

    def __init__(self, location):
        self.folder = os.path.abspath(
            os.path.expanduser(os.path.expandvars(location)))
        self.cases = os.listdir(self.folder)
        self.imu_data = None
        self.rotor_data = None
        self.mocap_data = None
        self.ConcatData = None
        self.setupLogging()
        logging.info("Dataloader Started")

    def setupLogging(self):
        log_format = "[%(filename)s]%(lineno)d::%(message)s"
        logging.basicConfig(level='DEBUG', format=log_format)

    def loadCase(self, case):
        for folder in os.listdir(os.path.join(self.folder, case)):
            logging.info(f"Loading {folder}")
            folder = join(join(self.folder, case), folder)
            bagfile = os.path.join(folder, "rosbag.bag")
            imu, rotor, mocap = self.parseBagFile(bagfile)
            if self.imu_data is None:
                self.imu_data = imu
                self.rotor_data = rotor
                self.mocap_data = mocap
            else:
                self.imu_data = pd.concat(
                    [self.imu_data, imu], ignore_index=True, axis=0)
                self.rotor_data = pd.concat(
                    [self.rotor_data, rotor], ignore_index=True, axis=0)
                self.mocap_data = pd.concat(
                    [self.mocap_data, mocap], ignore_index=True, axis=0)

    def parseBagFile(self, file):
        imu_topic = '/blackbird/imu'
        rpm_topic = '/blackbird/rotor_rpm'
        mocap_topic = '/blackbird/state'
        relevant_topics = [imu_topic, rpm_topic, mocap_topic]

        relevant_headers = [['Time', 'angular_velocity.x', 'angular_velocity.y',
                             'angular_velocity.z', 'linear_acceleration.x', 'linear_acceleration.y', 'linear_acceleration.z'],
                            ['Time', 'rpm_0', 'rpm_1', 'rpm_2', 'rpm_3'],
                            ['Time', 'pose.position.x', 'pose.position.y', 'pose.position.z', 'pose.orientation.x', 'pose.orientation.y', 'pose.orientation.z',
                             'pose.orientation.w']]
        csvfiles = []

        bagrdr = bagreader(file)
        for topic in bagrdr.topics:
            if topic in relevant_topics:
                data = bagrdr.message_by_topic(topic)
                csvfiles.append(data)
        imu = pd.read_csv(csvfiles[0])
        rotor = pd.read_csv(csvfiles[1])
        mocap = pd.read_csv(csvfiles[2])
        imu, rotor, mocap = imu[relevant_headers[0]
                                ], rotor[relevant_headers[1]], mocap[relevant_headers[2]]
        return imu, rotor, mocap

    def runPipeline(self):
        for case in tqdm(self.cases):
            self.loadCase(case)
        return self.imu_data, self.rotor_data, self.mocap_data

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
            - States a nx8 matrix with the following columns: Time, x, y, z, qw, qx, qy, qz
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

        perturbed_quats = []
        for i in range(len(self.ConcatData)):
            quat = Quaternion()
            quat.from_axis_angle(axis_angles[i])
            perturbed_quats.append(quat.q)
        perturbed_quats = np.array(perturbed_quats)

        data = np.vstack([time, pos[:, 0], pos[:, 1], pos[:, 2], perturbed_quats[:, 0],
                         perturbed_quats[:, 1], perturbed_quats[:, 2], perturbed_quats[:, 3]]).T

        return data


if __name__ == "__main__":
    print("Kindly Check dataloader_demo.ipynb for demo")

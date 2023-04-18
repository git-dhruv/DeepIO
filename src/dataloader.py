"""
@author: Dhruv Parikh
@date: 4/17/2023
@Description: Dataloading Class .. will update soon aka never
"""

import numpy as np
import pandas as pd
import os, sys

from bagpy import bagreader
import logging
from os.path import join
from tqdm import tqdm

class dataloader:

    def __init__(self,location):
        self.folder = os.path.abspath(os.path.expanduser(os.path.expandvars(location)))
        self.cases = os.listdir(self.folder)
        self.imu_data = None
        self.rotor_data = None
        self.mocap_data = None
        self.setupLogging() 
        logging.info("Dataloader Started")       

    def setupLogging(self):
        log_format = "[%(filename)s]%(lineno)d::%(message)s"
        logging.basicConfig(level='DEBUG', format=log_format)
    
    def loadCase(self, case):
        for folder in os.listdir(os.path.join(self.folder,case)):
            logging.info(f"Loading {folder}")
            folder = join(join(self.folder,case),folder)
            bagfile = os.path.join(folder,"rosbag.bag")
            imu,rotor,mocap = self.parseBagFile(bagfile)
            if self.imu_data is None:
                self.imu_data = imu
                self.rotor_data = rotor
                self.mocap_data = mocap
            else:
                self.imu_data = pd.concat([self.imu_data, imu], ignore_index=True, axis=0)
                self.rotor_data = pd.concat([self.rotor_data, rotor], ignore_index=True, axis=0)
                self.mocap_data = pd.concat([self.mocap_data, mocap], ignore_index=True, axis=0)

    def parseBagFile(self, file):
        imu_topic = '/blackbird/imu'
        rpm_topic = '/blackbird/rotor_rpm'
        mocap_topic = '/blackbird/state'
        relevant_topics = [imu_topic,rpm_topic,mocap_topic]

        relevant_headers = [['header.stamp.nsecs', 'angular_velocity.x','angular_velocity.y',
                             'angular_velocity.z','linear_acceleration.x','linear_acceleration.y','linear_acceleration.z'],
                             ['header.stamp.nsecs','rpm_0','rpm_1','rpm_2','rpm_3'],
                             ['header.stamp.nsecs','pose.position.x','pose.position.y','pose.position.z','pose.orientation.x','pose.orientation.y','pose.orientation.z',
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
        imu,rotor,mocap = imu[relevant_headers[0]],rotor[relevant_headers[1]],mocap[relevant_headers[2]]
        return imu,rotor,mocap

                    
    def runPipeline(self):
        for case in tqdm(self.cases):
            self.loadCase(case)
        return self.imu_data,self.rotor_data,self.mocap_data
            
        


if __name__ == "__main__":
    print("Kindly Check dataloader_demo.ipynb for demo")
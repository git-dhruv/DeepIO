"""
@author: Anirudh Kailaje, Dhruv Parikh
@date: 4/24/2023
@Description: Not Defined
Great way to document


(['angular_velocity.x', 'angular_velocity.y', 'angular_velocity.z',
'linear_acceleration.x', 'linear_acceleration.y',
'linear_acceleration.z', 'rpm_0', 'rpm_1', 'rpm_2', 'rpm_3',
'pose.position.x', 'pose.position.y', 'pose.position.z',
'pose.orientation.x', 'pose.orientation.y', 'pose.orientation.z',
'pose.orientation.w', 'Time']

"""
import numpy as np
import matplotlib.pyplot as plt
from dataloader import *
from dynamicsSim import *
from numpy import sin, cos


class OnlineLearingFusion:
    def __init__(self):
        """
        Standard Multi Sensor Fusion Parameters
        """
        self.state = np.zeros((15, 1))
        self.covariance = np.zeros((16, 16))
        self.Q = np.zeros((7, 7))
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
            self.MeasurmentJacobian = np.zeros((15, 15))
            self.MeasurmentJacobian[6:9, 6:9] = np.eye(3)
            self.MeasurmentJacobian[0:3, 0:3] = np.eye(3)
            return self.MeasurmentJacobian
        else:
            jacobian = np.eye(15, 15)
            jacobian[0:3, 3:6] = np.eye(3)*dt
            psi, theta, phi = self.state[6:9].flatten()
            Rdot_phi = np.array([[-sin(phi)*cos(theta), -cos(phi)*cos(psi)-sin(phi)*sin(theta)*sin(psi), cos(phi)*sin(psi)-sin(phi)*sin(theta)*cos(psi)],
                                 [cos(phi)*cos(theta), -sin(phi)*cos(psi)+cos(phi)*sin(theta)
                                  * sin(psi), sin(phi)*sin(psi)+cos(phi)*sin(theta)*cos(psi)],
                                 [-sin(theta),                            cos(theta)*sin(psi), cos(theta)*cos(psi)]])
            Rdot_theta = np.array([[-cos(phi)*sin(theta),                   cos(phi)*cos(theta)*sin(psi), cos(phi)*cos(theta)*cos(psi)],
                                   [-sin(phi)*sin(theta),                   sin(phi) *
                                    cos(theta)*sin(psi), sin(phi)*cos(theta)*cos(psi)],
                                   [-cos(theta),                             -sin(phi)*sin(psi), -sin(phi)*cos(psi)]])
            # Rdot_psi = np.array([[0, sin(phi)*sin(psi)+cos(phi)*sin(theta)*cos(psi), sin(phi)*cos(psi) - cos(phi)*sin(theta)*sin(psi)],
            #                      [0, -cos(phi)*sin(psi)+sin(phi)*sin(theta)*cos(psi), -
            #                       cos(phi)*cos(psi) - sin(phi)*sin(theta)*sin(psi)],
            #                      [0,                              cos(theta)*cos(psi), -cos(theta)*sin(psi)]])
            Rdot_psi = np.array([[0, sin(phi)*sin(psi)+cos(phi)*sin(theta)*cos(psi), sin(phi)*cos(psi) - cos(phi)*sin(theta)*sin(psi)],
                                 [0, -cos(phi)*sin(psi)+sin(phi)*sin(theta)*cos(psi), -
                                  cos(phi)*cos(psi) - sin(phi)*sin(theta)*sin(psi)],
                                 [0,                              cos(theta)*cos(psi), -cos(theta)*sin(psi)]])
            dF_dpsi = (Rdot_psi @ self.state[3:6])*dt
            dF_dtheta = (Rdot_theta @ self.state[3:6])*dt
            dF_dphi = (Rdot_phi @ self.state[3:6])*dt
            jacobian[3:6, 6] = dF_dpsi.flatten()
            jacobian[3:6, 7] = dF_dtheta.flatten()
            jacobian[3:6, 8] = dF_dphi.flatten()

            self.PropogationJacobian = jacobian
            return self.PropogationJacobian

    def propogateStep(self, state, rpm, dt):
        self.calcJacobian(dt, measurment=0)
        self.state = self.dynamics.propogateDynamics(state, rpm, dt)
        self.covariance = self.PropogationJacobian@self.covariance@self.PropogationJacobian.T + self.R

    def measurmentStep(self, measurments, dt):        
        G = self.calcJacobian(dt, measurment=1)

    def runPipeline(self):
        ##--Load the Data--##
        loadDataUtil = dataloader("..\data\clover")
        loadDataUtil.runPipeline()
        loadDataUtil.homogenizeData()
        gyro, acc, rpm, mocap, q, t = loadDataUtil.convertDataToIndividualNumpy()

        ##--Rotate the Motion Capture to IMU frame from Body Frame --#
        # R from imu to ned Frame
        R_imu_to_ned = np.array([[-1, 0, 0],
                                [0, -1, 0],
                                [0, 0, 1]])
        R_imutoBody= np.array([[0, -1, 0],
             [1, 0, 0],
             [0,0,1]])
        mocap = R_imutoBody @ mocap
        for i in range(q.shape[1]):
            shit = R_imutoBody@Rotation.from_quat(q[:,i]).as_matrix()            
            q[:,i] = Rotation.from_matrix(shit).as_quat().flatten()


        ##--Initialization--#
        self.state[:3] = mocap[:,0].flatten()
        self.state[9:12] = Rotation.from_quat(q[:,0].flatten()).as_euler('xyz')

        ##--Loop--#
        for i in range(1,q.shape[1]):
            dt = t[i] - t[i-1]
            self.propogateStep(self.state,rpm[:,i],dt)
            measurementPacket = np.array([float(gyro[0,i]),float(gyro[1,i]),float(gyro[2,i]),
                                          float(acc[0,i]),float(acc[1,i]),float(acc[2,i])])
            self.measurmentStep(measurementPacket, dt)
        return self.state

    def plotSampleOutput(self):
        raise NotImplementedError


if __name__ == '__main__':
    learner = OnlineLearingFusion()
    learner.runPipeline()

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
import tqdm

class OnlineLearingFusion:
    def __init__(self):
        """
        Standard Multi Sensor Fusion Parameters
        """
        self.state = np.zeros((21, 1))
        self.covariance = np.zeros((21, 21))
        self.R = np.eye(6,6)*0.1  #Measurement Noise
        self.Q = np.eye(self.covariance.shape[0]) *10

        self.PropogationJacobian = None
        self.MeasurmentJacobian = None

        self.dynamics = dynamics()

        """        
        a = [(q0^2 - q1^2 - q2^2 - q3^2)ax + 2(q0q1 - q2q3)ay + 2(q0q2 + q1q3)az,
        2(q0q1 + q2q3)ax + (q0^2 - q1^2 + q2^2 - q3^2)ay + 2(q1q2 - q0q3)az,
        2(q0q2 - q1q3)ax + 2(q1q2 + q0*q3)*ay + (q0^2 - q1^2 - q2^2 + q3^2)*az]
        """

    def calcJacobian(self, dt, measurment=1, omega = np.zeros(4)):
        psi, theta, phi = self.state[6:9].flatten()
        Rdot_phi = np.array([[-sin(phi)*cos(theta), -cos(phi)*cos(psi)-sin(phi)*sin(theta)*sin(psi), cos(phi)*sin(psi)-sin(phi)*sin(theta)*cos(psi)],
                                    [cos(phi)*cos(theta), -sin(phi)*cos(psi)+cos(phi)*sin(theta)
                                    * sin(psi), sin(phi)*sin(psi)+cos(phi)*sin(theta)*cos(psi)],
                                    [-sin(theta),                            cos(theta)*sin(psi), cos(theta)*cos(psi)]])
        Rdot_theta = np.array([[-cos(phi)*sin(theta),                   cos(phi)*cos(theta)*sin(psi), cos(phi)*cos(theta)*cos(psi)],
                                [-sin(phi)*sin(theta),                   sin(phi) *
                                cos(theta)*sin(psi), sin(phi)*cos(theta)*cos(psi)],
                                [-cos(theta),                             -sin(phi)*sin(psi), -sin(phi)*cos(psi)]])
        Rdot_psi = np.array([[0, sin(phi)*sin(psi)+cos(phi)*sin(theta)*cos(psi), sin(phi)*cos(psi) - cos(phi)*sin(theta)*sin(psi)],
                                [0, -cos(phi)*sin(psi)+sin(phi)*sin(theta)*cos(psi), -
                                cos(phi)*cos(psi) - sin(phi)*sin(theta)*sin(psi)],
                                [0,                              cos(theta)*cos(psi), -cos(theta)*sin(psi)]])
        
        if measurment:
            self.MeasurmentJacobian = np.zeros((6, 21))
            dg_dpsi = Rdot_psi.T @ self.state[6:9].reshape(-1,1)
            dg_dtheta = Rdot_theta.T @ self.state[6:9].reshape(-1,1)
            dg_dphi = Rdot_phi.T @ self.state[6:9].reshape(-1,1)
            self.MeasurmentJacobian[0:3, 6] = dg_dpsi.flatten()
            self.MeasurmentJacobian[0:3, 7] = dg_dtheta.flatten()
            self.MeasurmentJacobian[0:3, 8] = dg_dphi.flatten()
            dg_dpsi = Rdot_psi.T @ self.state[12:15].reshape(-1,1)
            dg_dtheta = Rdot_theta.T @ self.state[12:15].reshape(-1,1)
            dg_dphi = Rdot_phi.T @ self.state[12:15].reshape(-1,1)
            self.MeasurmentJacobian[3:6, 12] = dg_dpsi.flatten()
            self.MeasurmentJacobian[3:6, 13] = dg_dtheta.flatten()
            self.MeasurmentJacobian[3:6, 14] = dg_dphi.flatten()
            return self.MeasurmentJacobian
        else:
            u, _ = self.dynamics.rpmConversions(omega.flatten())
            jacobian = np.zeros((21, 21))
            jacobian[0:3, 0:3] = np.eye(3)
            jacobian[0:3, 3:6] = np.eye(3)*dt
            jacobian[3:6, 3:6] = np.eye(3)
            jacobian[3:6, 6:9] = np.eye(3)*dt
            
            
            jacobian[6:9, 9] = - Rdot_psi @ np.array([0,0, u.sum()])
            jacobian[6:9, 10] = - Rdot_theta @ np.array([0,0, u.sum()])
            jacobian[6:9, 11] = - Rdot_phi @ np.array([0,0, u.sum()])
            jacobian[9:21, 9:21] = np.eye(12)
            jacobian[9:12, 12:15] = np.eye(3)*dt

            self.PropogationJacobian = jacobian
            return self.PropogationJacobian

    def propogateStep(self, state, rpm, dt):
        J = self.calcJacobian(dt, measurment=0)
        self.state = self.dynamics.propogateDynamics(state, rpm, dt)
        self.covariance = J@self.covariance@J.T + self.Q

    def measurementModel(self):
        #Rotation of IMU wrt NED
        R = Rotation.from_euler('xyz',self.state[9:12].flatten()).as_matrix()
        #Rotation of NED wrt imu
        R = R.T
        gyro = R@self.state[12:15].reshape(-1,1) 
        acc = R@self.state[6:9].reshape(-1,1)
        return np.vstack((acc,gyro))

    def measurmentStep(self, measurments, dt):        
        y = measurments.reshape(-1,1) - self.measurementModel()
        H = self.calcJacobian(dt,measurment=1)
        S = H@self.covariance@H.T + self.R
        K = self.covariance@H.T@np.linalg.inv(S)
        self.state = self.state + K@y
        self.covariance = (np.eye(21) - K@H)@self.covariance

    def runPipeline(self):
        ##--Load the Data--##
        dataDir = r'/Users/dhruv/Desktop/Penn/Sem2/ESE650/FinalProject/DeepIO/data/clover'
        loadDataUtil = dataloader(dataDir)
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
        self.state[:3] = mocap[:,0].reshape(-1,1)
        self.state[9:12] = Rotation.from_quat(q[:,0].flatten()).as_euler('xyz').reshape(-1,1)
        acc[-1,:] = acc[-1,:] - acc[-1,:20].mean()
        print(acc[-1,:2])

        ##--Loop--#
        x = []
        for i in tqdm.tqdm(range(1,q.shape[1]-90000)):
            dt = t[i] - t[i-1]
            self.propogateStep(self.state,rpm[:,i],dt)
            measurementPacket = np.array([float(acc[0,i]),float(acc[1,i]),float(acc[2,i]),
                                          float(gyro[0,i]),float(gyro[1,i]),float(gyro[2,i])])
            self.measurmentStep(measurementPacket, dt)
            x.append(float(self.state[0]))

        plt.plot(x); plt.plot(mocap[0,:]);
        plt.show()

        return self.state

    def plotSampleOutput(self):
        raise NotImplementedError


if __name__ == '__main__':
    learner = OnlineLearingFusion()
    learner.runPipeline()

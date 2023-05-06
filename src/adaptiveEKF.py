"""
@author: Anirudh Kailaje, Dhruv Parikh
@date: 4/24/2023
@Description: EKF for quadrotor state estimation by fusing IMU and Motion Capture Data


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
from copy import deepcopy
from estimate_rot import ukfPipeline
from complementary import *
import tensorflow as tf

class OnlineLearningFusion:
    def __init__(self,dataDir = r'data\clover'):
        """
        Standard Multi Sensor Fusion Parameters
            - self.state: State of the quadrotor defined as [x, y, z, vx, vy, vz, ax, ay, az, psi, theta, phi, p, q, r, bax, bay, baz, bwx, bwy, bwz]
        """
        self.MotionCap = 2
        self.IMU = 1 #1 for IMU, 2 for Motion Capture, Dynamics otherwise Will be used for calcJacobian, measurement model

        self.state = np.zeros((21, 1))
        self.covariance = np.zeros((21, 21))

        self.R = np.eye(6,6)*1.0  #Measurement Noise
        self.R_imu = deepcopy(self.R)*100

        self.Q = np.eye(self.covariance.shape[0])*10

        self.PropogationJacobian = None
        self.MeasurmentJacobian = None

        self.dynamics = dynamics()

        self.loadDataUtil = dataloader(dataDir)
        self.loadDataUtil.runPipeline()
        self.loadDataUtil.homogenizeData()


    def calcJacobian(self, dt, measurment= 1, omega = np.zeros(4)):
        """
        Returns the desired jacobian for the EKF
        Inputs:
            - dt: Time Step
            - measurment: 1 for IMU, 2 for Motion Capture, Dynamics otherwise
            - omega: motor speeds of the quadrotor
        Outputs:
            - None; Computes the jacobian and stores it in the class in self.MeasurmentJacobian or self.PropogationJacobian
        """
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
        
        if measurment == self.IMU: #Returns the meassurement jacobian for the IMU 
            self.MeasurmentJacobian = np.zeros((6, 21))
            dg_dpsi = Rdot_psi.T @ self.state[6:9].reshape(-1,1)
            dg_dtheta = Rdot_theta.T @ self.state[6:9].reshape(-1,1)
            dg_dphi = Rdot_phi.T @ self.state[6:9].reshape(-1,1)
            self.MeasurmentJacobian[0:3, 8] = dg_dpsi.flatten()
            self.MeasurmentJacobian[0:3, 7] = dg_dtheta.flatten()
            self.MeasurmentJacobian[0:3, 6] = dg_dphi.flatten()
            dg_dpsi = Rdot_psi.T @ self.state[15:18].reshape(-1,1)
            dg_dtheta = Rdot_theta.T @ self.state[15:18].reshape(-1,1)
            dg_dphi = Rdot_phi.T @ self.state[15:18].reshape(-1,1)
            
            self.MeasurmentJacobian[0:3, 17] = dg_dpsi.flatten()
            self.MeasurmentJacobian[0:3, 16] = dg_dtheta.flatten()
            self.MeasurmentJacobian[0:3, 15] = dg_dphi.flatten()

            dg_dpsi = Rdot_psi.T @ self.state[12:15].reshape(-1,1)
            dg_dtheta = Rdot_theta.T @ self.state[12:15].reshape(-1,1)
            dg_dphi = Rdot_phi.T @ self.state[12:15].reshape(-1,1)
            self.MeasurmentJacobian[3:6, 14] = dg_dpsi.flatten()
            self.MeasurmentJacobian[3:6, 13] = dg_dtheta.flatten()
            self.MeasurmentJacobian[3:6, 12] = dg_dphi.flatten()
            
            dg_dpsi = Rdot_psi.T @ self.state[18:].reshape(-1,1)
            dg_dtheta = Rdot_theta.T @ self.state[18:].reshape(-1,1)
            dg_dphi = Rdot_phi.T @ self.state[18:].reshape(-1,1)
            self.MeasurmentJacobian[3:6, 18] = dg_dpsi.flatten()
            self.MeasurmentJacobian[3:6, 19] = dg_dtheta.flatten()
            self.MeasurmentJacobian[3:6, 20] = dg_dphi.flatten()
            return self.MeasurmentJacobian
        
        elif measurment == self.MotionCap: #Returns measurement jacobian for motion capture input
            self.MeasurmentJacobian = np.zeros((6, 21))
            self.MeasurmentJacobian[0:3, 0:3] = np.eye(3)
            self.MeasurmentJacobian[3:6, 9:12] = np.eye(3)
            return self.MeasurmentJacobian

        else: #Returns the dynamics Jacobian
            u, _ = self.dynamics.rpmConversions(omega.flatten())
            jacobian = np.zeros((21, 21))
            jacobian[0:3, 0:3] = np.eye(3)
            jacobian[0:3, 3:6] = np.eye(3)*dt
            jacobian[3:6, 3:6] = np.eye(3)
            jacobian[3:6, 6:9] = np.eye(3)*dt
            # jacobian[0:3, 6:9] = np.eye(3)*dt*dt*0.5
            
            
            jacobian[6:9, 11] = - Rdot_psi @ np.array([0,0, u.sum()/0.9])
            jacobian[6:9, 10] = - Rdot_theta @ np.array([0,0, u.sum()/0.9])
            jacobian[6:9, 9] = - Rdot_phi @ np.array([0,0, u.sum()/0.9])
            jacobian[9:21, 9:21] = np.eye(12)
            jacobian[9:12, 12:15] = 0.0 #np.eye(3)*dt

            self.PropogationJacobian = jacobian
            return self.PropogationJacobian

    def propogateStep(self, state, rpm, dt):
        J = self.calcJacobian(dt, measurment=0)
        self.state = self.dynamics.propogateDynamics(state, rpm, dt)
        self.covariance = J@self.covariance@J.T + self.Q

    def measurementModel(self, packet_num = 1):
        #x,y,z,vx,vy,vz,acc,acc,acc,r,p,y,wx,wy,wz,acb,ac

        if packet_num == 2: #Motion capture
            return np.vstack((self.state[0:3], self.state[9:12]))
        else:            #IMU
            #Rotation of IMU wrt NED
            R = Rotation.from_euler('xyz',self.state[9:12].flatten()).as_matrix()
            #Rotation of NED wrt imu
            R = R.T
            gyro = R@(self.state[12:15].reshape(-1,1)+self.state[18:].reshape(-1,1)).reshape(-1,1) 
            acc = R@(self.state[6:9].reshape(-1,1)+self.state[15:18].reshape(-1,1))
            return np.vstack((acc,gyro))
        
    def getValidRotation(self, R):
        for _ in range(3):
            R = np.round(R,2)
            U,_,Vt = np.linalg.svd(R)
            Smod = np.eye(3)
            Smod[-1,-1] = np.linalg.det(U@Vt)
            R = U@Smod@Vt
        return R

    def measurmentStep(self, measurments, dt, packet_num = 1, Adapt = True, beta = 0.7, alpha = 0.9):  
        if packet_num==1:
            Rot = self.getValidRotation(Rotation.from_euler('xyz', self.state[9:12].flatten()).as_matrix()).T
            # measurments[:3] = (measurments[:3].reshape(-1,1) - Rot@self.grav.reshape(-1,1)).flatten()

            #Normalizing the accelerometer
            # measurments[:3] = 9.81*measurments[:3]/np.linalg.norm(measurments[:3])
            R = deepcopy(self.R_imu)
        else:
            R = deepcopy(self.R)

        #Innovation calculation
        y = measurments.reshape(-1,1) - self.measurementModel(packet_num= packet_num)
        #Measurement Jacobian
        H = self.calcJacobian(dt,measurment=packet_num)

        #Used for AdaptiveEKF, S is the covariance of mear
        S = H@self.covariance@H.T + R

        #Kalman Gain
        K = self.covariance@H.T@np.linalg.inv(S)

        """
        if packet_num==self.IMU:
            K = CNN(measurments,self.covariance,self.state)
        """

        old_state = self.state.copy()
        self.state = self.state + K@y
        q = Rotation.from_euler('xyz', self.state[9:12].flatten()).as_quat()
        q = q/np.linalg.norm(q)

        self.state[9:12] = Rotation.from_quat(q).as_euler('xyz').reshape(-1,1)

        #Apadtive EKF
        if Adapt:
            #Angle Naive Residual Calculation
            residual = measurments.reshape(-1,1) - self.measurementModel(packet_num=packet_num)
            if packet_num == self.MotionCap:
                #Nearest rotation residuals -> doesnt matter the directions
                residual[3:] = np.arctan2(np.sin(residual[3:]),np.cos(residual[3:]))

            # if packet_num==1:
            #     residual[:3] = residual[:3]*np.clip(np.linalg.norm(measurments[:3])/10,1,None)

            R = beta*R + (1-beta)*(residual@residual.T+H@self.covariance@H.T)
            
            #Recalculating State Update
            H = self.calcJacobian(dt,measurment=packet_num)
            S = H@self.covariance@H.T + R
            K = self.covariance@H.T@np.linalg.inv(S)
            self.state = old_state + K@y
            q = Rotation.from_euler('xyz', self.state[9:12].flatten()).as_quat()
            q = q/np.linalg.norm(q)

            self.state[9:12] = Rotation.from_quat(q).as_euler('xyz').reshape(-1,1)
            ####Adaptive Q part Doesnt work for some reason####
            # if True:
            #     #There are two formulas, 2nd one works this one might not
            #     self.Q = alpha*self.Q + (1-alpha)*(K@y@y.T@K.T)
                
        #Covariance Update
        self.covariance = (np.eye(21) - K@H)@self.covariance

        if packet_num== self.IMU:
            self.R_imu = deepcopy(R)
        else:
            self.R = deepcopy(R)




        if np.all(np.linalg.eigvals(self.Q) < 0) or np.all(np.linalg.eigvals(self.R) < 0):
            print("AdaptiveEKF Q or R is not positive definite")


    def runPipeline(self,  
                    Adapt = True, sensor_biases = np.array([1000.0, 130, -150.0]), IMU_step = 20, MotionCap_step = 1000, beta = 0.3, alpha = 0.9):
        #____________________________Load Data____________________________#
        
        OrientationNN = tf.keras.models.load_model(r'SavedModels\GRUModelPredictingOrientation.h5')
        PositionNN = tf.keras.models.load_model(r'SavedModels\GRUModelPredictingPosition.h5')
        
        
        gyro, acc, rpm, mocap, q, t = self.loadDataUtil.convertDataToIndividualNumpy()
        perturbedMocap = self.loadDataUtil.perturbStates()[:,1:]
        
        Gtruth = mocap.copy()
        mocap = perturbedMocap[:,:3]
        quats = perturbedMocap[:,3:]
        

        

        #____________Rotate the Motion Capture to IMU frame from Body Frame_______#
        # R from imu to ned Frame
        R_imu_to_ned = np.array([[-1, 0, 0],
                                 [0, -1, 0],
                                 [0,  0, 1]])
        R_imutoBody= np.array([ [0, -1, 0],
                                [1,  0, 0],
                                [0,  0, 1]])
        
        mocap = R_imutoBody @ mocap.T
        Gtruth = R_imutoBody@ Gtruth
        eulers = []

        # # plt.plot(gyro[0,:])
        # # plt.plot(gyro[1,:])
        # plt.plot(acc[-1,:])
        # # plt.plot(mocap[0,:])
        # # plt.plot(mocap[1,:])
        # plt.plot(mocap[2,:])
        # # plt.legend(["Acc X","Acc Y","Acc Z","mocap X", "Mocap Y"])
        # plt.show()



        for i in range(quats.shape[0]):
            R_temp = R_imutoBody@Rotation.from_quat(quats[i, :]).as_matrix()
            quats[i, :] = Rotation.from_matrix(R_temp).as_quat().flatten()
            quat = Quaternion(scalar = quats[i, -1], vec = quats[i, 0:3])
            eulers.append(quat.euler_angles())
        eulers = np.array(eulers)
        


        #_________________Initialize State_________________________#
        # acc = -acc
        self.state += 1e-15
        self.state[:3] = mocap[:,0].reshape(-1,1)
        self.state[9:12] = eulers[0,:].reshape(-1,1)
        # self.state[9:12] = Rotation.from_quat(q[:,0].flatten()).as_euler('xyz').reshape(-1,1)
        self.state[18:] = gyro[:,:20].mean(axis=1).reshape(-1,1)
        self.state[15:18] = acc[:3,:20].mean(axis=1).reshape(-1,1)
        # self.state[-3:] = sensor_biases.reshape(-1,1)
        self.grav = np.array([0,0,9.81]).reshape(-1,1)


        ##--Loop--#
        self.estimates = []
        self.estimates.append(self.state)
        self.covariances = []
        self.covariances.append(self.covariance)
        self.groundtruth = np.vstack((Gtruth, eulers.T)).T
        self.groundtruth = self.groundtruth[:25000, :]



        for i in tqdm.tqdm(range(1,25000)):
            dt = t[i] - t[i-1]
            self.propogateStep(self.state,rpm[:,i],dt)
            if i%IMU_step == 0:

                measurementPacket = np.array([float(acc[0,i]),float(acc[1,i]),float(acc[2,i]),
                                            float(gyro[0,i]),float(gyro[1,i]),float(gyro[2,i])])
                measurementPacket2 = np.array([mocap[0,i],mocap[1,i],mocap[2,i],eulers[i,0],eulers[i,1],eulers[i,2]])
                self.measurmentStep(measurementPacket, dt, packet_num = self.IMU, Adapt = Adapt, beta =beta)
                
                if i%MotionCap_step == 0:
                    self.measurmentStep(measurementPacket2, dt, packet_num = self.MotionCap, Adapt = Adapt, beta = beta)
                
            self.estimates.append(self.state)
            self.covariances.append(self.covariance)
        
        self.estimates = np.array(self.estimates).reshape(-1,21)
        self.covariances = np.array(self.covariances).reshape(-1,21,21)

        return self.estimates, self.covariances, self.groundtruth, gyro, acc, perturbedMocap, eulers


if __name__ == '__main__':
    learner = OnlineLearningFusion()
    learner.runPipeline()

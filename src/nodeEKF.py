"""
@author: Anirudh Kailaje, Dhruv Parikh
@date: 4/24/2023
@Description: EKF for quadrotor state estimation by fusing IMU and Noisy Capture Data. 
Version: 5.0 - Probably Final
"""



import numpy as np
import matplotlib.pyplot as plt
from dataloader import *
from dynamicsSim import *
from numpy import sin, cos
import tqdm
from copy import deepcopy
import torch
import torch.nn as nn

#This is not recommended 
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class euler_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(9, 30, dtype=torch.float64)
        self.layer2 = nn.Linear(30, 30, dtype=torch.float64)
        self.layer3 = nn.Linear(30, 3, dtype=torch.float64)
        self.tanh = nn.Tanh()

    def forward(self, sensor_reading):
        x = self.tanh(self.layer1(sensor_reading))
        x = self.tanh(self.layer2(x))
        x = self.layer3(x)
        return x


class OnlineLearningFusion:
    def __init__(self,dataDir = r'C:\Users\dhruv\Desktop\Penn\Sem2\ESE650\FinalProject\DeepIO\data\clover'):
        """
        Standard Multi Sensor Fusion Parameters
            - self.state: State of the quadrotor defined as [x, y, z, vx, vy, vz, ax, ay, az, psi, theta, phi, p, q, r, bax, bay, baz, bwx, bwy, bwz]
        """
        #Loading Model first since it takes a lot of time
        self.model = euler_model()
        #load dict
        self.model.load_state_dict(torch.load('src/NN_Euler.pt',map_location=torch.device('cpu')))
        self.model.eval()

        #Flags for Combining various sensors and dynamics 
        self.MotionCap = 2
        self.IMU = 1 #1 for IMU, 2 for Motion Capture, Dynamics otherwise Will be used for calcJacobian, measurement model
        self.NN = 10

        #State and Covariance
        self.state = np.zeros((21, 1))
        self.covariance = np.zeros((21, 21))

        self.R = np.eye(3,3)  #Mocap Measurment Noise
        self.R_imu = deepcopy(np.eye(3,3)*1.0)*100  #NN IMU Measrument Noise

        self.Q = np.eye(self.covariance.shape[0])*1e10  #State Update through IMU

        #Nobody should use these shit variables
        self.PropogationJacobian = None
        self.MeasurmentJacobian = None

        #Dynamics Class
        self.dynamics = dynamics()

        #Load Data Class
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
        elif measurment == self.NN:
            J = np.zeros((3, 21))
            J[:,9:12] = np.eye(3)
            return J
        
        elif measurment == self.MotionCap: #Returns measurement jacobian for motion capture input
            J = np.zeros((3, 21))
            J[:,0:3] = np.eye(3)
            return J

        else: #Returns the dynamics Jacobian
            """
            Thought:
            We will fuse acceleration from previous time step. Since we can avoid Rotation matrix in Jacobian
            """
            jacobian = np.zeros((21, 21))
            #Position Jac
            jacobian[0:3, 0:3] = np.eye(3)          #Previous Pos
            jacobian[0:3, 3:6] = np.eye(3)*dt       #Previos Vel
            jacobian[0:3, 6:9] = 0.5*np.eye(3)*dt*dt       #Previos Acc
            #Velocity Jac
            jacobian[3:6, 3:6] = np.eye(3)          #Previous Vel
            jacobian[3:6, 6:9] = np.eye(3)*dt       #Acc
            
            #Acceleration Jacobian
            dg_dpsi = Rdot_psi @ self.state[6:9].reshape(-1,1)
            dg_dtheta = Rdot_theta @ self.state[6:9].reshape(-1,1)
            dg_dphi = Rdot_phi @ self.state[6:9].reshape(-1,1)
            jacobian[6:9,6:9] = np.eye(3)
            jacobian[6:9,15:18] = -np.eye(3)

            # x,y,z,vx,vy,vz,acc,acc,acc,r,p,y,wx,wy,wz,acb,ac
            #Acceleration Biases in acceleration
            dg_dpsi = Rdot_psi @ self.state[15:18].reshape(-1,1)
            dg_dtheta = Rdot_theta @ self.state[15:18].reshape(-1,1)
            dg_dphi = Rdot_phi @ self.state[15:18].reshape(-1,1)

            #Angles
            jacobian[9:12,9:12] = np.eye(3)
            jacobian[9:12,12:15] = np.eye(3)*dt
            #Angular Velocity
            jacobian[12:15,12:15] = np.eye(3)            
            #Gyro Biases in omega
            jacobian[12:15, -1] = -1#-dg_dpsi.flatten()
            jacobian[12:15, -2] = -1#-dg_dtheta.flatten()
            jacobian[12:15, -3] = -1#-dg_dphi.flatten()

            #Rest all are zero
            return jacobian

    def propogateStep2(self, state, rpm, dt):
        J = self.calcJacobian(dt, measurment=0)
        self.state = self.dynamics.propogateDynamics(state, rpm, dt)
        self.covariance = J@self.covariance@J.T + self.Q

    def propogateStep(self, state, imu, dt):
        """
        Propogate Step requires IMU measurement packet, previous step and dt
        """
        #propogate acc-  abias + gdt
        
        self.state = self.dynamics.propogateIMUDynamics(state,imu,dt)
        J = self.calcJacobian(dt, measurment=-1)
        self.covariance = J@self.covariance@J.T + self.Q

    def measurementModel(self, packet_num = 1):
        #x,y,z,vx,vy,vz,acc,acc,acc,r,p,y,wx,wy,wz,acb,ac
        if packet_num==self.NN:
            return self.state[9:12].reshape(-1,1)
        
        if packet_num == 2: #Motion capture
            return self.state[:3].reshape(-1,1)
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
        """
        Least Squares Optimization for Rotation matrices in closest rotation for rounding errors
        """
        for _ in range(3):
            R = np.round(R,3)
            U,_,Vt = np.linalg.svd(R)
            Smod = np.eye(3)
            Smod[-1,-1] = np.linalg.det(U@Vt)
            R = U@Smod@Vt
        return R

    def measurmentStep(self, measurments, dt, packet_num = 1, Adapt = True, beta = 0.7, alpha = 0.3):  
        if packet_num==self.NN:
            # Rot = self.getValidRotation(Rotation.from_euler('xyz', self.state[9:12].flatten()).as_matrix()).T
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
        if False:
            #There are two formulas, 2nd one works this one might not
            self.Q = alpha*self.Q + (1-alpha)*(K@y@y.T@K.T)
                
        #Covariance Update
        self.covariance = (np.eye(21) - K@H)@self.covariance

        if packet_num== self.NN:
            self.R_imu = deepcopy(R)
        else:
            self.R = deepcopy(R)




        if np.all(np.linalg.eigvals(self.Q) < 0) or np.all(np.linalg.eigvals(self.R) < 0):
            print("AdaptiveEKF Q or R is not positive definite")


    def runPipeline(self,  
                    Adapt = False, sensor_biases = np.array([1000.0, 130, -150.0]), IMU_step = 20, MotionCap_step = 1000, beta = 0.5, alpha = 0.3):
        #____________________________Load Data____________________________#
        
        
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
        for i in range(quats.shape[0]):
            R_temp = R_imutoBody@Rotation.from_quat(quats[i, :]).as_matrix()
            quats[i, :] = Rotation.from_matrix(R_temp).as_quat().flatten()
            quat = Quaternion(scalar = quats[i, -1], vec = quats[i, 0:3])
            eulers.append(quat.euler_angles())
        eulers = np.array(eulers)
        


        #_________________Initialize State_________________________#
        
        self.state += 1e-15
        self.state[:3] = mocap[:,0].reshape(-1,1)
        self.state[9:12] = eulers[0,:].reshape(-1,1)
        # self.state[9:12] = Rotation.from_quat(q[:,0].flatten()).as_euler('xyz').reshape(-1,1)
        self.state[18:] = gyro[:,:20].mean(axis=1).reshape(-1,1)
        self.state[15:17] = acc[:2,:20].mean(axis=1).reshape(-1,1)
        # self.state[-2:] = sensor_biases.reshape(-1,1)
        self.grav = -np.array([0,0,9.81]).reshape(-1,1)


        ##--Loop--#
        self.estimates = []
        self.estimates.append(self.state)
        self.covariances = []
        self.covariances.append(self.covariance)
        self.groundtruth = np.vstack((Gtruth, eulers.T)).T
        self.groundtruth = self.groundtruth[:25000, :]

        for i in tqdm.tqdm(range(1,25000)):


            #---------------- Propogation ---------------#
            #Time Difference
            dt = t[i] - t[i-1]

            #Preparing IMU packet for State Propogation
            imuPacket = np.array([float(acc[0,i]),float(acc[1,i]),float(acc[2,i]),
                                            float(gyro[0,i]),float(gyro[1,i]),float(gyro[2,i])])

            #To remove non linearity in jacobian we need to iterate 10 times in single time frame
            for j in range(10):
                self.propogateStep(self.state,imuPacket,dt/10)




            #---------------- Neural Network ---------------#
            #Sensor Packet for Neural Network Propogation
            neuralPacket = np.array([float(acc[0,i]),float(acc[1,i]),float(acc[2,i]),
                float(gyro[0,i]),float(gyro[1,i]),float(gyro[2,i]),0,0,-9.8])
            #Converting to tensor
            neuralInput = torch.tensor(neuralPacket,dtype=torch.float64)
            #Model Forward
            nnPacket = self.model(neuralInput).cpu().detach().numpy()
            # nnPacket = 0.6*self.state[9:12].reshape(-1,1) + 0.4*nnPacket.reshape(-1,1)

            #Measurment step
            self.measurmentStep(nnPacket, dt, packet_num = self.NN, Adapt =Adapt, beta = beta)

            #----------- Motion Capture ------------#               
            if i%MotionCap_step == 0 or (i<5000):
                #Packet for Motion Capture
                measurementPacket2 = np.array([mocap[0,i],mocap[1,i],mocap[2,i]])         
                #Measurement Update Step
                self.measurmentStep(measurementPacket2, dt, packet_num = self.MotionCap, Adapt = 1-Adapt, beta = beta)


            #----------- Loggers -----------------#                
            self.estimates.append(self.state)
            self.covariances.append(self.covariance)
        
        self.estimates = np.array(self.estimates).reshape(-1,21)
        self.covariances = np.array(self.covariances).reshape(-1,21,21)

        return self.estimates, self.covariances, self.groundtruth, gyro, acc, perturbedMocap, eulers


if __name__ == '__main__':
    #Define the class
    learner = OnlineLearningFusion()
    #Run Pipeline
    estimate, covariance, ground_truth, gyro, acc, perturbedMocap, eulers = learner.runPipeline(
    Adapt=1, IMU_step=20, MotionCap_step=1000, sensor_biases=np.array([1000.0, 130, -150.0]), beta=0.1)
    #Plots 
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))

    axs[0, 0].plot(estimate[:, 0], label='estimate_x')
    axs[0, 0].plot(ground_truth[:, 0], label='ground_truth_x')
    axs[0, 0].legend()

    axs[0, 1].plot(estimate[:, 1], label='estimate_y')
    axs[0, 1].plot(ground_truth[:, 1], label='ground_truth_y')
    axs[0, 1].legend()

    axs[0, 2].plot(estimate[:, 2], label='estimate_z')
    axs[0, 2].plot(ground_truth[:, 2], label='ground_truth_z')
    axs[0, 2].legend()

    axs[1, 0].plot(estimate[:, 9]*180/np.pi, label='estimate_psi')
    axs[1, 0].plot(ground_truth[:, 3]*180/np.pi,
                label='ground_truth_psi', alpha=0.8)
    axs[1, 0].legend()

    axs[1, 1].plot(estimate[:, 10]*180/np.pi, label='estimate_theta')
    axs[1, 1].plot(ground_truth[:, 4]*180/np.pi, label='ground_truth_theta')
    axs[1, 1].legend()

    axs[1, 2].plot(estimate[:, 11]*180/np.pi, label='estimate_phi')
    axs[1, 2].plot(ground_truth[:, 5]*180/np.pi, label='ground_truth_phi')
    axs[1, 2].legend()

    plt.show()
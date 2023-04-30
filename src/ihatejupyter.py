# %%
from dataloader import *
from adaptiveEKF import *
learner = OnlineLearningFusion(r'data\clover')


# %%
from random import random
leastsquares = []
betas = []
for i in range(1):
    beta = random()
    estimate, covariance, ground_truth, gyro, acc, perturbedMocap, eulers = learner.runPipeline(
        Adapt=True, IMU_step=20, MotionCap_step=1000, sensor_biases=np.array([1000.0, 130, -150.0]),beta=0.2594063010484614)
    err = np.linalg.norm(np.sum(np.abs(ground_truth[:,:3] - estimate[:,:3]),axis=0))
    leastsquares.append(err)
    betas.append(beta)

print(leastsquares, beta)

# %%
# %matplotlib widget
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


# %%
eulers.shape


# %%
eulers[0, 0]


# %%


# %%


# %%


# %%


# %%


# %%


# %%




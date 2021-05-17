import matplotlib.pyplot as plt
from pilco.rewards import ExponentialReward
import tensorflow as tf
import numpy as np
with tf.Session(graph=tf.Graph()) as sess:
    R = ExponentialReward(state_dim=2, t=np.array([0.,0.]),W=np.ones((2,2)))
    muRs = lambda th:R.compute_reward(np.array([np.sin(th),np.cos(th)]), s=np.eye(2))
    left = [i/36*2*np.pi for i in range(36)]
    height = np.array([muRs(th)[0].eval()[0][0] for th in left])
plt.xlabel('rad')
plt.ylabel('mean of reward')
plt.plot(left, height,label='ones')
plt.legend()
plt.savefig(/script)

import json

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)

filename = "recorded_acs.json"


buffer = json.load(open(filename))

data = np.array(buffer)

timestamps = data[:, 0]


freq = 50
dt = 1 / freq


# def print_stats(base_dq, joint_dq):
    # print("All joint stat:")
    # print(" mean: {0:.3f}".format(np.mean(joint_dq[:, :])))
    # print(" std: {0:.3f}".format(np.std(joint_dq[:, :])))
    # print(" max: {0:.3f}".format(np.max(joint_dq[:, :])))
    # print(" min: {0:.3f}".format(np.min(joint_dq[:, :])))

    # for i in range(12):
    #     print("Joint #{0} stat:".format(i))
    #     print(" mean: {0:.3f}".format(np.mean(joint_vel[:, i])))
    #     print(" std: {0:.3f}".format(np.std(joint_vel[:, i])))
    #     print(" max: {0:.3f}".format(np.max(joint_vel[:, i])))
    #     print(" min: {0:.3f}".format(np.min(joint_vel[:, i])))



obs = data[:, 1:]

joint_pos = obs[:, 9:9+12]
joint_vel = obs[:, 21:21+12]


obs_scale_dof_vel = 0.05
joint_vel /= obs_scale_dof_vel

def print_stats(base_ang):
    base_g = base_ang
    nominal_g = np.array([0, 0, -1])


    angles = np.arccos(np.abs(np.dot(base_g, nominal_g)))

    angles_1 = angles[:-1]
    angles_2 = angles[1:]

    d_angles = np.abs(angles_2 - angles_1)
    d_angles /= dt

    d_angles_1 = d_angles[:-1]
    d_angles_2 = d_angles[1:]

    dd_angles = np.abs(d_angles_2 - d_angles_1)
    dd_angles /= dt


    print("Base angvel stat:")
    print(" mean: {0:.4f}".format(np.mean(d_angles)))
    print(" std: {0:.4f}".format(np.std(d_angles)))
    print(" max: {0:.4f}".format(np.max(d_angles)))
    print(" min: {0:.4f}".format(np.min(d_angles)))
    print("")

    print("Base angacc stat:")
    print(" mean: {0:.4f}".format(np.mean(dd_angles)))
    print(" std: {0:.4f}".format(np.std(dd_angles)))
    print(" max: {0:.4f}".format(np.max(dd_angles)))
    print(" min: {0:.4f}".format(np.min(dd_angles)))
    print("")



print("1 - 2 s")
print_stats(obs[freq*1:freq*2, 0:3])
print("2 - 3 s")
print_stats(obs[freq*2:freq*3, 0:3])
print("3 - 4 s")
print_stats(obs[freq*3:freq*4, 0:3])
print("stay")
print_stats(obs[int(freq*5.6):, 0:3])

# plt.plot(timestamps, joint_pos[:, 3], label="q[3]")
# plt.plot(timestamps, joint_pos[:, 4], label="q[4]")
# plt.plot(timestamps, joint_pos[:, 5], label="q[5]")
# # plt.plot(timestamps, obs[:, 6])

# plt.axvline(x=2, ls="--{0:.3f}".format(color="#000{0:.3f}".format(lw=1)
# plt.legend()

# plt.show()





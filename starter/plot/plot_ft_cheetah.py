import csv
import matplotlib.pyplot as plt
import numpy as np
# log_csv_path = "log/MUJOCO_CHEETAH_VEL_8_withnoise_v4_finetune/HalfCheetah-v3/1/forward_3_6_finetune_init_log.csv"
# log_file = open(log_csv_path, 'r')
# init_3_6_data = []
# reader = csv.DictReader(log_file)
# for data in reader:
#     init_3_6_data.append(list(data.values()))
# init_3_6_data = np.array(init_3_6_data, dtype=float)

# log_csv_path = "log/MUJOCO_CHEETAH_VEL_8_withnoise_v4_finetune/HalfCheetah-v3/1/forward_7_2_finetune_init_log.csv"
# log_file = open(log_csv_path, 'r')
# init_7_2_data = []
# reader = csv.DictReader(log_file)
# for data in reader:
#     init_7_2_data.append(list(data.values()))
# init_7_2_data = np.array(init_7_2_data, dtype=float)

# log_csv_path = "log/MUJOCO_CHEETAH_VEL_8_withnoise_v4_finetune/HalfCheetah-v3/1/forward_9_5_finetune_init_log.csv"
# log_file = open(log_csv_path, 'r')
# init_9_5_data = []
# reader = csv.DictReader(log_file)
# for data in reader:
#     init_9_5_data.append(list(data.values()))
# init_9_5_data = np.array(init_9_5_data, dtype=float)




log_csv_path = "log/MUJOCO_CHEETAH_VEL_8_withnoise_v4_finetune/HalfCheetah-v3/1/forward_3_6_finetune_random_log.csv"
log_file = open(log_csv_path, 'r')
random_3_6_data = []
reader = csv.DictReader(log_file)
for data in reader:
    random_3_6_data.append(list(data.values()))
random_3_6_data = np.array(random_3_6_data, dtype=float)

log_csv_path = "log/MUJOCO_CHEETAH_VEL_8_withnoise_v4_finetune/HalfCheetah-v3/1/forward_7_2_finetune_random_log.csv"
log_file = open(log_csv_path, 'r')
random_7_2_data = []
reader = csv.DictReader(log_file)
for data in reader:
    random_7_2_data.append(list(data.values()))
random_7_2_data = np.array(random_7_2_data, dtype=float)

log_csv_path = "log/MUJOCO_CHEETAH_VEL_8_withnoise_v4_finetune/HalfCheetah-v3/1/forward_9_5_finetune_random_log.csv"
log_file = open(log_csv_path, 'r')
random_9_5_data = []
reader = csv.DictReader(log_file)
for data in reader:
    random_9_5_data.append(list(data.values()))
random_9_5_data = np.array(random_9_5_data, dtype=float)



















# epochs = init_3_6_data[:, 0]
# reward_init_3_6 = init_3_6_data[:, 1]
# velocity_init_3_6 = init_3_6_data[:, 2]

# reward_init_7_2 = init_7_2_data[:, 1]
# velocity_init_7_2 = init_7_2_data[:, 2]

# reward_init_9_5 = init_9_5_data[:, 1]
# velocity_init_9_5 = init_9_5_data[:, 2]


reward_random_3_6 = random_3_6_data[:, 1]
velocity_random_3_6 = random_3_6_data[:, 2]

reward_random_7_2 = random_7_2_data[:, 1]
velocity_random_7_2 = random_7_2_data[:, 2]

reward_random_9_5 = random_9_5_data[:, 1]
velocity_random_9_5 = random_9_5_data[:, 2]



plt.figure(figsize=(7,5))
# plt.plot(reward_init_3_6, lw = 1.5, label = 'velocity_init_3.6')
# plt.plot(reward_init_7_2, lw = 1.5, label = 'velocity_init_7.2')
# plt.plot(reward_init_9_5, lw = 1.5, label = 'velocity_init_9.5')

plt.plot(reward_random_3_6, lw = 1.5, label = 'velocity_random_3.6')
plt.plot(reward_random_7_2, lw = 1.5, label = 'velocity_random_7.2')
plt.plot(reward_random_9_5, lw = 1.5, label = 'velocity_random_9.5')

plt.grid(True)
plt.legend(loc = 0)
plt.axis('tight')
plt.xlabel('Epoch')
plt.ylabel('Reward')
plt.title('HalfCheetah-Vel finetune')
plt.ylim((-5000, 0))
plt.xlim((0, 10))
plt.savefig('./fig/random_reward_8.jpg')

plt.figure(figsize=(7,5))
# plt.plot(velocity_init_3_6, lw = 1.5, label = 'velocity_init_3.6')
# plt.plot(velocity_init_7_2, lw = 1.5, label = 'velocity_init_7.2')
# plt.plot(velocity_init_9_5, lw = 1.5, label = 'velocity_init_9.5')

plt.plot(velocity_random_3_6, lw = 1.5, label = 'velocity_random_3.6')
plt.plot(velocity_random_7_2, lw = 1.5, label = 'velocity_random_7.2')
plt.plot(velocity_random_9_5, lw = 1.5, label = 'velocity_random_9.5')

plt.grid(True)
plt.legend(loc = 0)
plt.axis('tight')
plt.xlabel('Epoch')
plt.ylabel('Velocity')
plt.title('HalfCheetah-Vel finetune')
plt.ylim((0, 10))
plt.xlim((0, 10))
plt.savefig('./fig/random_velocity_8.jpg')
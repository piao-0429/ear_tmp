import csv
import matplotlib.pyplot as plt
import numpy as np

id = "AR_HOPPER_VEL_10_scale_reward_finetune"
env = "Hopper-v3"
seed = 1
log_dir = "log/"+id+"/"+env+"/"+str(seed)


log_csv_path = log_dir+"/vel_1_5_finetune_random_log.csv"
log_file = open(log_csv_path, 'r')
random_0_3_data = []
reader = csv.DictReader(log_file)
for data in reader:
    random_0_3_data.append(list(data.values()))
random_0_3_data = np.array(random_0_3_data, dtype=float)

log_csv_path = log_dir+"/vel_3_5_finetune_random_log.csv"
log_file = open(log_csv_path, 'r')
random_0_7_data = []
reader = csv.DictReader(log_file)
for data in reader:
    random_0_7_data.append(list(data.values()))
random_0_7_data = np.array(random_0_7_data, dtype=float)

log_csv_path = log_dir+"/vel_7_5_finetune_random_log.csv"
log_file = open(log_csv_path, 'r')
random_1_5_data = []
reader = csv.DictReader(log_file)
for data in reader:
    random_1_5_data.append(list(data.values()))
random_1_5_data = np.array(random_1_5_data, dtype=float)






reward_random_0_3 = random_0_3_data[:, 1]
velocity_random_0_3 = random_0_3_data[:, 2]

reward_random_0_7 = random_0_7_data[:, 1]
velocity_random_0_7 = random_0_7_data[:, 2]

reward_random_1_5 = random_1_5_data[:, 1]
velocity_random_1_5 = random_1_5_data[:, 2]



plt.figure(figsize=(7,5))


plt.plot(reward_random_0_3, lw = 1.5, label = 'velocity_random_0.3')
plt.plot(reward_random_0_7, lw = 1.5, label = 'velocity_random_0.7')
plt.plot(reward_random_1_5, lw = 1.5, label = 'velocity_random_1.5')

plt.grid(True)
plt.legend(loc = 0)
plt.axis('tight')
plt.xlabel('Epoch')
plt.ylabel('Reward')
plt.title('Hopper-Vel finetune')
plt.ylim((-100, 200))
plt.xlim((0, 10))
plt.savefig('./fig/random_hopper_reward_10.jpg')

plt.figure(figsize=(7,5))

plt.plot(velocity_random_0_3, lw = 1.5, label = 'velocity_random_0.3')
plt.plot(velocity_random_0_7, lw = 1.5, label = 'velocity_random_0.7')
plt.plot(velocity_random_1_5, lw = 1.5, label = 'velocity_random_1.5')

plt.grid(True)
plt.legend(loc = 0)
plt.axis('tight')
plt.xlabel('Epoch')
plt.ylabel('Velocity')
plt.title('Hopper-Vel finetune')
plt.ylim((0, 2.5))
plt.xlim((0, 10))
plt.savefig('./fig/random_hopper_velocity_10.jpg')
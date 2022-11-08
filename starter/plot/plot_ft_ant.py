import csv
import matplotlib.pyplot as plt
import numpy as np
log_csv_path = "log/MUJOCO_ANT_DIR_24_withnoise_v4_finetune/Ant-v3/2/direction_77_finetune_init_log.csv"
log_file = open(log_csv_path, 'r')
init_77_data = []
reader = csv.DictReader(log_file)
for data in reader:
    init_77_data.append(list(data.values()))
init_77_data = np.array(init_77_data, dtype=float)

log_csv_path = "log/MUJOCO_ANT_DIR_24_withnoise_v4_finetune/Ant-v3/2/direction_82_finetune_init_log.csv"
log_file = open(log_csv_path, 'r')
init_82_data = []
reader = csv.DictReader(log_file)
for data in reader:
    init_82_data.append(list(data.values()))
init_82_data = np.array(init_82_data, dtype=float)

log_csv_path = "log/MUJOCO_ANT_DIR_24_withnoise_v4_finetune/Ant-v3/2/direction_97_finetune_init_log.csv"
log_file = open(log_csv_path, 'r')
init_97_data = []
reader = csv.DictReader(log_file)
for data in reader:
    init_97_data.append(list(data.values()))
init_97_data = np.array(init_97_data, dtype=float)

log_csv_path = "log/MUJOCO_ANT_DIR_24_withnoise_v4_finetune/Ant-v3/2/direction_201_finetune_init_log.csv"
log_file = open(log_csv_path, 'r')
init_201_data = []
reader = csv.DictReader(log_file)
for data in reader:
    init_201_data.append(list(data.values()))
init_201_data = np.array(init_201_data, dtype=float)

log_csv_path = "log/MUJOCO_ANT_DIR_24_withnoise_v4_finetune/Ant-v3/2/direction_133_finetune_init_log.csv"
log_file = open(log_csv_path, 'r')
init_133_data = []
reader = csv.DictReader(log_file)
for data in reader:
    init_133_data.append(list(data.values()))
init_133_data = np.array(init_133_data, dtype=float)


log_csv_path = "log/MUJOCO_ANT_DIR_24_withnoise_v4_finetune_/Ant-v3/2/direction_77_finetune_random_log.csv"
log_file = open(log_csv_path, 'r')
random_77_data = []
reader = csv.DictReader(log_file)
for data in reader:
    random_77_data.append(list(data.values()))
random_77_data = np.array(random_77_data, dtype=float)

log_csv_path = "log/MUJOCO_ANT_DIR_24_withnoise_v4_finetune_/Ant-v3/2/direction_82_finetune_random_log.csv"
log_file = open(log_csv_path, 'r')
random_82_data = []
reader = csv.DictReader(log_file)
for data in reader:
    random_82_data.append(list(data.values()))
random_82_data = np.array(random_82_data, dtype=float)

log_csv_path = "log/MUJOCO_ANT_DIR_24_withnoise_v4_finetune_/Ant-v3/2/direction_97_finetune_random_log.csv"
log_file = open(log_csv_path, 'r')
random_97_data = []
reader = csv.DictReader(log_file)
for data in reader:
    random_97_data.append(list(data.values()))
random_97_data = np.array(random_97_data, dtype=float)

log_csv_path = "log/MUJOCO_ANT_DIR_24_withnoise_v4_finetune_/Ant-v3/2/direction_201_finetune_random_log.csv"
log_file = open(log_csv_path, 'r')
random_201_data = []
reader = csv.DictReader(log_file)
for data in reader:
    random_201_data.append(list(data.values()))
random_201_data = np.array(random_201_data, dtype=float)

log_csv_path = "log/MUJOCO_ANT_DIR_24_withnoise_v4_finetune_/Ant-v3/2/direction_133_finetune_random_log.csv"
log_file = open(log_csv_path, 'r')
random_133_data = []
reader = csv.DictReader(log_file)
for data in reader:
    random_133_data.append(list(data.values()))
random_133_data = np.array(random_133_data, dtype=float)


















epochs = init_77_data[:, 0]
reward_init_77 = init_77_data[:, 1]
direction_init_77 = init_77_data[:, 2]

reward_init_82 = init_82_data[:, 1]
direction_init_82 = init_82_data[:, 2]

reward_init_97 = init_97_data[:, 1]
direction_init_97 = init_97_data[:, 2]

reward_init_201 = init_201_data[:, 1]
direction_init_201 = init_201_data[:, 2]

reward_init_133 = init_133_data[:, 1]
direction_init_133 = init_133_data[:, 2]

reward_random_77 = random_77_data[:, 1]
direction_random_77 = random_77_data[:, 2]

reward_random_82 = random_82_data[:, 1]
direction_random_82 = random_82_data[:, 2]

reward_random_97 = random_97_data[:, 1]
direction_random_97 = random_97_data[:, 2]

reward_random_201 = random_201_data[:, 1]
direction_random_201 = random_201_data[:, 2]

reward_random_133 = random_133_data[:, 1]
direction_random_133 = random_133_data[:, 2]

plt.figure(figsize=(7,5))
plt.plot(reward_init_77, lw = 1.5, label = 'direction_init_77')
plt.plot(reward_init_82, lw = 1.5, label = 'direction_init_82')
plt.plot(reward_init_97, lw = 1.5, label = 'direction_init_97')
plt.plot(reward_init_133, lw = 1.5, label = 'direction_init_133')
plt.plot(reward_init_201, lw = 1.5, label = 'direction_init_201')
# plt.plot(reward_random_77, lw = 1.5, label = 'direction_random_77')
# plt.plot(reward_random_82, lw = 1.5, label = 'direction_random_82')
# plt.plot(reward_random_97, lw = 1.5, label = 'direction_random_97')
# plt.plot(reward_random_133, lw = 1.5, label = 'direction_random_133')
# plt.plot(reward_random_201, lw = 1.5, label = 'direction_random_201')
plt.grid(True)
plt.legend(loc = 0)
plt.axis('tight')
plt.xlabel('Epoch')
plt.ylabel('Reward')
plt.title('Ant-Dir finetune')
plt.ylim((0, 8000))
plt.xlim((0, 10))
plt.savefig('./fig/init_reward_24.jpg')

plt.figure(figsize=(7,5))
plt.plot(direction_init_77, lw = 1.5, label = 'direction_init_77')
plt.plot(direction_init_82, lw = 1.5, label = 'direction_init_82')
plt.plot(direction_init_97, lw = 1.5, label = 'direction_init_97')
plt.plot(direction_init_133, lw = 1.5, label = 'direction_init_133')
plt.plot(direction_init_201, lw = 1.5, label = 'direction_init_201')
# plt.plot(direction_random_77, lw = 1.5, label = 'direction_random_77')
# plt.plot(direction_random_82, lw = 1.5, label = 'direction_random_82')
# plt.plot(direction_random_97, lw = 1.5, label = 'direction_random_97')
# plt.plot(direction_random_133, lw = 1.5, label = 'direction_random_133')
# plt.plot(direction_random_201, lw = 1.5, label = 'direction_random_201')

plt.grid(True)
plt.legend(loc = 0)
plt.axis('tight')
plt.xlabel('Epoch')
plt.ylabel('Direction')
plt.title('Ant-Dir finetune')
plt.ylim((0, 360))
plt.xlim((0, 10))
plt.savefig('./fig/init_direction_24.jpg')
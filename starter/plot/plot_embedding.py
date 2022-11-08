import matplotlib
import matplotlib.pyplot as plt
import csv
import numpy as np

task_list = ["direction_0", "direction_30", "direction_60", "direction_90", "direction_120", "direction_150", "direction_180", "direction_210", "direction_240", "direction_270", "direction_300", "direction_330"]
task_list = ["direction_0", "direction_15", "direction_30", "direction_45", "direction_60", "direction_75", "direction_90", "direction_105", "direction_120", "direction_135", "direction_150", "direction_165",  "direction_180", "direction_195", "direction_210", "direction_225", "direction_240", "direction_255", "direction_270", "direction_285", "direction_300", "direction_315", "direction_330", "direction_345"]
task_num = len(task_list)
embed_dir = "embedding/MUJOCO_ANT_DIR_24_withnoise_v4/Ant-v3/2"
additional_embed_dir = "log/MUJOCO_ANT_DIR_24_withnoise_v4_finetune/Ant-v3/2/model"

for i in range(task_num):
    embed_csv_path = embed_dir + '/' + task_list[i] + '.csv'
    # embed_file = open(embed_csv_path, "r")
    embedding = np.loadtxt(embed_csv_path, delimiter=',')
    if i == 0:
        embeddings = embedding
    else:
        embeddings = np.vstack((embeddings, embedding))
    

additional_task_list = ["direction_77", "direction_82", "direction_97", "direction_133", "direction_201"]
additional_task_num = len(additional_task_list)

for i in range(additional_task_num):
    embed_csv_path = additional_embed_dir + '/' + additional_task_list[i] + '_finetune_random_finish.csv'
    # embed_file = open(embed_csv_path, "r")
    embedding = np.loadtxt(embed_csv_path, delimiter=',')
    if i == 0:
        additional_embeddings = embedding
    else:
        additional_embeddings = np.vstack((additional_embeddings, embedding))

x = embeddings[:, 0]
y = embeddings[:, 1]
z = embeddings[:, 2]
add_x = additional_embeddings[:, 0]
add_y = additional_embeddings[:, 1]
add_z = additional_embeddings[:, 2]


# 绘制散点图
ax = plt.gca(projection='3d') 
ax = plt.gca() 
ax.scatter(x, y, z, c='b')
 
ax.scatter(add_x, add_y, add_z, c='r')
# 添加坐标轴(顺序是Z, Y, X)
ax.set_zlabel('dim_2')
ax.set_ylabel('dim_1')
ax.set_xlabel('dim_0')


for i in range(task_num):
    ax.text(x[i]*1.01, y[i]*1.01, z[i]*1.01,task_list[i][10:], fontsize=10, color = 'b')
for i in range(additional_task_num):
    ax.text(add_x[i]*1.01, add_y[i]*1.01, add_z[i]*1.01,task_list[i][10:], fontsize=10, color = 'r')

plt.savefig("./plot_embedding.png")



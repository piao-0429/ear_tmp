import os
import glob
import time
from datetime import datetime
from PIL import Image
import csv
import sys

sys.path.append(".") 
import torch
import os
import time
import os.path as osp
import numpy as np
from torchrl.utils import get_args
from torchrl.utils import get_params
from torchrl.env import get_env


import torchrl.policies as policies
import torchrl.networks as networks
import gym
from mujoco_py import GlfwContext
# GlfwContext(offscreen=True)  # Create a window to init GLFW.




args = get_args()
params = get_params(args.config)
env=gym.make(params['env_name'])

# task_list=["forward_1","forward_2","forward_3","forward_4","forward_5","forward_6","forward_7","forward_8","forward_9","forward_10"]
task_list = ["dir_0", "dir_30", "dir_60", "dir_90", "dir_120", "dir_150", "dir_180", "dir_210", "dir_240", "dir_270", "dir_300", "dir_330"]
task_num=len(task_list)
representation_shape= params['representation_shape']
embedding_shape=params['embedding_shape']
params['p_state_net']['base_type']=networks.MLPBase
params['task_net']['base_type']=networks.MLPBase
params['p_action_net']['base_type']=networks.MLPBase
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
pf_state = networks.Net(
	input_shape=env.observation_space.shape[0], 
	output_shape=representation_shape,
	**params['p_state_net']
)

pf_task=networks.NormNet(
	input_shape=task_num, 
	output_shape=embedding_shape,
	**params['task_net'],
	norm = 5
)

pf_action=policies.ActionRepresentationGuassianContPolicy(
	input_shape = representation_shape + embedding_shape,
	output_shape = 2 * env.action_space.shape[0],
	**params['p_action_net'] 
)
experiment_id = str(args.id)
model_dir="log/"+experiment_id+"/"+params['env_name']+"/"+str(args.seed)+"/model/"

# pf_state.load_state_dict(torch.load(model_dir + "model_pf_state_finish.pth", map_location='cpu'))
# pf_task.load_state_dict(torch.load(model_dir + "model_pf_task_finish.pth", map_location='cpu'))
# pf_action.load_state_dict(torch.load(model_dir + "model_pf_action_finish.pth", map_location='cpu'))

# pf_state.load_state_dict(torch.load(model_dir + "model_pf_state_8060.pth", map_location='cpu'))
# pf_task.load_state_dict(torch.load(model_dir + "model_pf_task_8060.pth", map_location='cpu'))
# pf_action.load_state_dict(torch.load(model_dir + "model_pf_action_8060.pth", map_location='cpu'))

pf_state.load_state_dict(torch.load(model_dir + "model_pf_state_best.pth", map_location='cpu'))
pf_task.load_state_dict(torch.load(model_dir + "model_pf_task_best.pth", map_location='cpu'))
pf_action.load_state_dict(torch.load(model_dir + "model_pf_action_best.pth", map_location='cpu'))

############################# save images for gif ##############################


def save_gif_images(env_name, max_ep_len):

	print("============================================================================================")
	device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	if args.cuda:
		torch.backends.cudnn.deterministic=True

	# make directory for saving gif images
	gif_images_dir = "gif_images" + '/'
	if not os.path.exists(gif_images_dir):
		os.makedirs(gif_images_dir)

	gif_images_dir = gif_images_dir + '/' + experiment_id + '/'
	if not os.path.exists(gif_images_dir):
		os.makedirs(gif_images_dir)

	# make environment directory for saving gif images
	gif_images_dir = gif_images_dir + '/' + env_name + '/'
	if not os.path.exists(gif_images_dir):
		os.makedirs(gif_images_dir)

	gif_images_dir_list=[]
	
	for i in range(len(task_list)):
		# gif_images_dir_list[i]=gif_images_dir+"/"+cls_list[i]+"/"
		gif_images_dir_list.append(gif_images_dir+"/"+task_list[i]+"/")
		if not os.path.exists(gif_images_dir_list[i]):
			os.makedirs(gif_images_dir_list[i])

	# make directory for gif
	gif_dir = "gifs" + '/'
	if not os.path.exists(gif_dir):
		os.makedirs(gif_dir)

	gif_dir = gif_dir + '/' + experiment_id + '/'
	if not os.path.exists(gif_dir):
		os.makedirs(gif_dir)

	# make environment directory for gif
	gif_dir = gif_dir + '/' + env_name  + '/'
	if not os.path.exists(gif_dir):
		os.makedirs(gif_dir)

	gif_dir_list=[]
	
	for i in range(len(task_list)):
        # gif_dir_list[i]=gif_dir+"/"+cls_list[i]+"/"
		gif_dir_list.append(gif_dir+"/"+task_list[i]+"/")
		if not os.path.exists(gif_dir_list[i]):
			os.makedirs(gif_dir_list[i])

	if params["save_embedding"]:
		embed_dir = "embedding"+'/'
		if not os.path.exists(embed_dir):
			os.makedirs(embed_dir)
		embed_dir = embed_dir + '/' + experiment_id + '/'
		if not os.path.exists(embed_dir):
			os.makedirs(embed_dir)
		embed_dir = embed_dir + '/' + env_name  + '/'
		if not os.path.exists(embed_dir):
			os.makedirs(embed_dir)
		embed_dir = embed_dir + '/' + str(args.seed) + '/'
		if not os.path.exists(embed_dir):
			os.makedirs(embed_dir)	

	if params["save_velocity"]:
		velocity_dir = "velocity"+'/'
		if not os.path.exists(velocity_dir):
			os.makedirs(velocity_dir)
		velocity_dir = velocity_dir + '/' + experiment_id + '/'
		if not os.path.exists(velocity_dir):
			os.makedirs(velocity_dir)
		velocity_dir = velocity_dir + '/' + env_name  + '/'
		if not os.path.exists(velocity_dir):
			os.makedirs(velocity_dir)
		velocity_dir = velocity_dir + '/' + str(args.seed) + '/'
		if not os.path.exists(velocity_dir):
			os.makedirs(velocity_dir)	

		average_v_csv_path = velocity_dir+ "/average_velocity.csv"
		average_v_file = open(average_v_csv_path,"w")
		average_v_writer = csv.writer(average_v_file)
		average_v_writer.writerow(["task","v_mean","v_std"])

	for i in range(len(task_list)):
		if params["save_embedding"]:
			embed_csv_path = embed_dir + '/' + task_list[i] + ".csv"
			embed_file = open(embed_csv_path, "w")
			embed_writer = csv.writer(embed_file)
		# if params["save_velocity"]:
		# 	velocity_csv_path = velocity_dir+ '/' + task_list[i] + ".csv"
		# 	velocity_file = open(velocity_csv_path,'w')
		# 	velocity_writer = csv.writer(velocity_file)

		ob=env.reset()
		task_input = torch.zeros(len(task_list))
		task_input[i] = 1
		task_input=task_input.unsqueeze(0).to("cpu")
		with torch.no_grad():
			for t in range(1, max_ep_len+1):
				representation = pf_state.forward(torch.Tensor( ob ).to("cpu").unsqueeze(0))
				embedding = pf_task.forward(task_input)
				out=pf_action.explore(representation,embedding)
				act=out["action"]
				act = act.detach().cpu().numpy()
				next_ob, _, done, info = env.step(act)
				# if params["save_velocity"]:
				# 	x_velocity = info['x_velocity']
				# 	velocity_writer.writerow([x_velocity])
				# img = env.render(mode = 'rgb_array')
				# img = Image.fromarray(img)
				# img.save(gif_images_dir_list[i] + '/' + experiment_id + '_' + task_list[i] + str(t).zfill(6) + '.jpg')
				ob=next_ob
				if done:
					break
			x = info['x_position']
			y = info['y_position']
			dir = np.arctan(y/x)/ np.pi * 180
			if x<0 and y>0:
				dir+=180
			elif x<0 and y<0:
				dir+=180
			elif x>0 and y<0:
				dir+=360
			
			print("task", i, "direction:", dir)

		if params["save_embedding"]:
			embedding = embedding.squeeze(0)
			embedding = embedding.detach().cpu().numpy()
			embed_writer.writerow(embedding)
			embed_file.close()
		# if params["save_velocity"]:
		# 	velocity_file.close()
		# 	velocity_file = open(velocity_csv_path,'r')
		# 	velocity_list = np.loadtxt(velocity_file)
		# 	velocity_list = velocity_list[100:]
		# 	average_v_writer.writerow([task_list[i], np.mean(velocity_list), np.std(velocity_list)])


	env.close()











######################## generate gif from saved images ########################

def save_gif(env_name):

	print("============================================================================================")

	gif_num = args.seed    
	experiment_id=str(args.id)

	# adjust following parameters to get desired duration, size (bytes) and smoothness of gif
	total_timesteps = 1000
	step = 1
	frame_duration = 60

	# input images
	# gif_images_dir = "gif_images/" + env_name + '/*.jpg'
	gif_images_dir = "gif_images/" + experiment_id + '/' + env_name +"/"
	gif_images_dir_list=[]
	for i in range(len(task_list)):
		gif_images_dir_list.append(gif_images_dir+"/"+task_list[i]+"/*.jpg")

	# output gif path
	gif_dir = "gifs"
	if not os.path.exists(gif_dir):
		os.makedirs(gif_dir)

	gif_dir = gif_dir + '/' + experiment_id + '/' + env_name
	if not os.path.exists(gif_dir):
		os.makedirs(gif_dir)
	gif_path_list=[]
	for i in range(len(task_list)):
		gif_path_list.append(gif_dir+"/"+task_list[i]+"/"+experiment_id+'_'+task_list[i]+ '_gif_' + str(gif_num) + '.gif')
	
	img_paths_list=[]
	for i in range(len(task_list)):

		img_paths_list.append(sorted(glob.glob(gif_images_dir_list[i]))) 
		img_paths_list[i] = img_paths_list[i][:total_timesteps]
		img_paths_list[i] = img_paths_list[i][::step]

		img, *imgs = [Image.open(f) for f in img_paths_list[i]]
		img.save(fp=gif_path_list[i], format='GIF', append_images=imgs, save_all=True, optimize=True, duration=frame_duration, loop=0)
		print("saved gif at : ", gif_path_list[i])



if __name__ == '__main__':
	env_name = params["env_name"]
	max_ep_len = 1000           
	save_gif_images(env_name,  max_ep_len)
	# save_gif(env_name)



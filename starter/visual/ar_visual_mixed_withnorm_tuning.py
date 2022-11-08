from audioop import avg
from cmath import inf
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
import torch.nn.functional as F
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
task_list=["direction_7_5_mixed","direction_22_5_mixed","direction_37_5_mixed","direction_52_5_mixed","direction_67_5_mixed","direction_82_5_mixed","direction_97_5_mixed","direction_112_5_mixed","direction_127_5_mixed", "direction_142_5_mixed","direction_157_5_mixed","direction_172_5_mixed","direction_187_5_mixed","direction_202_5_mixed","direction_217_5_mixed","direction_232_5_mixed","direction_247_5_mixed","direction_262_5_mixed","direction_277_5_mixed","direction_292_5_mixed","direction_307_5_mixed", "direction_322_5_mixed","direction_337_5_mixed","direction_352_5_mixed"]
task_list=["direction_15_mixed","direction_45_mixed","direction_75_mixed","direction_105_mixed","direction_135_mixed","direction_165_mixed","direction_195_mixed","direction_225_mixed","direction_255_mixed", "direction_285_mixed","direction_315_mixed","direction_345_mixed"]
task_list = ["vel_1.5", "vel_2.5", "vel_3.5", "vel_4.5", "vel_5.5", "vel_6.5", "vel_7.5", "vel_8.5", "vel_9.5", "vel_"]
task_list=["direction_7_5_mixed","direction_22_5_mixed","direction_37_5_mixed","direction_52_5_mixed","direction_67_5_mixed","direction_82_5_mixed","direction_97_5_mixed","direction_112_5_mixed","direction_127_5_mixed", "direction_142_5_mixed","direction_157_5_mixed","direction_172_5_mixed","direction_187_5_mixed","direction_202_5_mixed","direction_217_5_mixed","direction_232_5_mixed","direction_247_5_mixed","direction_262_5_mixed","direction_277_5_mixed","direction_292_5_mixed","direction_307_5_mixed", "direction_322_5_mixed","direction_337_5_mixed","direction_352_5_mixed"]

task_num=len(task_list)
representation_shape= params['representation_shape']
embedding_shape=params['embedding_shape']
params['p_state_net']['base_type']=networks.MLPBase
params['task_net']['base_type']=networks.MLPBase
params['p_action_net']['base_type']=networks.MLPBase
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
pf_state = networks.NormNet(
	input_shape=env.observation_space.shape[0], 
	output_shape=representation_shape,
	**params['p_state_net']
)

pf_task = networks.NormNet(
	input_shape=task_num,
	output_shape=embedding_shape,
	**params['task_net']
)

pf_action=policies.ActionRepresentationGuassianContPolicy(
	input_shape = representation_shape + embedding_shape,
	output_shape = 2 * env.action_space.shape[0],
	**params['p_action_net'] 
)
experiment_id = str(args.id)
experiment_id_v2 = experiment_id + "_mixed"
model_dir="log/"+experiment_id+"/"+params['env_name']+"/"+str(args.seed)+"/model/"

# pf_state.load_state_dict(torch.load(model_dir + "model_pf_state_finish.pth", map_location='cpu'))
# pf_action.load_state_dict(torch.load(model_dir + "model_pf_action_finish.pth", map_location='cpu'))

# pf_state.load_state_dict(torch.load(model_dir + "model_pf_state_8060.pth", map_location='cpu'))
# pf_action.load_state_dict(torch.load(model_dir + "model_pf_action_8060.pth", map_location='cpu'))

pf_state.load_state_dict(torch.load(model_dir + "model_pf_state_best.pth", map_location='cpu'))
pf_action.load_state_dict(torch.load(model_dir + "model_pf_action_best.pth", map_location='cpu'))
pf_task.load_state_dict(torch.load(model_dir + "model_pf_task_best.pth", map_location='cpu'))

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

	gif_images_dir = gif_images_dir + '/' + experiment_id_v2 + '/'
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

	gif_dir = gif_dir + '/' + experiment_id_v2 + '/'
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

	task_inputs = torch.eye(task_num)
	pre_embeddings = pf_task.forward(task_inputs).unsqueeze(1)

	pre_embeddings_ = torch.cat((pre_embeddings[1:], pre_embeddings[0].unsqueeze(0)), dim=0)
	# a=0.50
	# embeddings = F.normalize(a* pre_embeddings + (1-a) * pre_embeddings_, dim=-1)


	for i in range(task_num):
		a_m = 0
		err_m = 100
		label = i*15 + 7.5
		for a in range(1,100):
			 
			a = a/100
			embedding = F.normalize(a * pre_embeddings[i] + (1-a) * pre_embeddings_[i], dim=-1)
			ob=env.reset()
			sum_vel = 0
			with torch.no_grad():
				for t in range(1, max_ep_len+1):
					representation = pf_state.forward(torch.Tensor( ob ).to("cpu").unsqueeze(0))
					out=pf_action.explore(representation,embedding)
					act=out["action"]
					act = act.detach().cpu().numpy()
					next_ob, _, done, info = env.step(act)
				
					if t > 100:
						sum_vel += info['x_velocity']
					ob=next_ob
					if done:
						break
				if params['env_name'] == 'Ant-v3':
					x = info['x_position']
					y = info['y_position']
					dir = np.arctan(y/x)/ np.pi * 180
					if x<0 and y>0:
						dir+=180
					elif x<0 and y<0:
						dir+=180
					elif x>0 and y<0:
						dir+=360
					err = abs(dir-label)
					if err < err_m:
						err_m = err
						a_m = a
						l_m = dir
					# print(task_list[i], dir)
     
				else:
					# print(task_list[i], sum_vel/ (t - 100))
					avg_v = sum_vel/ (t - 100)
					err = abs(avg_v - label)
					if err < err_m:
						err_m = err
						a_m = a
						l_m = avg_v
		print(task_list[i], a_m, l_m)



	env.close()











######################## generate gif from saved images ########################

def save_gif(env_name):

	print("============================================================================================")

	gif_num = args.seed    
	experiment_id=str(args.id)

	# adjust following parameters to get desired duration, size (bytes) and smoothness of gif
	total_timesteps = 200
	step = 1
	frame_duration = 60

	# input images
	gif_images_dir = "gif_images/" + experiment_id_v2 + '/' + env_name +"/"
	gif_images_dir_list=[]
	for i in range(len(task_list)):
		gif_images_dir_list.append(gif_images_dir+"/"+task_list[i]+"/*.jpg")

	# output gif path
	gif_dir = "gifs"
	if not os.path.exists(gif_dir):
		os.makedirs(gif_dir)

	gif_dir = gif_dir + '/' + experiment_id_v2 + '/' + env_name
	if not os.path.exists(gif_dir):
		os.makedirs(gif_dir)
	gif_path_list=[]
	for i in range(len(task_list)):
		gif_path_list.append(gif_dir+"/"+task_list[i]+"/"+experiment_id_v2+'_'+task_list[i]+ '_gif_' + str(gif_num) + '.gif')
	
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
	max_ep_len = 200    
	save_gif_images(env_name,  max_ep_len)
	# save_gif(env_name)



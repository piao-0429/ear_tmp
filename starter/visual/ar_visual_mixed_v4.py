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
task_list=["dir_15_mixed","dir_45_mixed","dir_75_mixed","dir_105_mixed","dir_135_mixed","dir_165_mixed","dir_195_mixed","dir_225_mixed","dir_255_mixed", "dir_285_mixed","dir_315_mixed","dir_345_mixed"]
task_num=len(task_list)
representation_shape= params['representation_shape']
embedding_shape=params['embedding_shape']
embedding4q_shape=params['embedding4q_shape']
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

	if params["save_embedding"]:
		embed_dir = "embedding"+'/'
		if not os.path.exists(embed_dir):
			os.makedirs(embed_dir)
		embed_dir = embed_dir + '/' + experiment_id_v2 + '/'
		if not os.path.exists(embed_dir):
			os.makedirs(embed_dir)
		embed_dir = embed_dir + '/' + env_name  + '/'
		if not os.path.exists(embed_dir):
			os.makedirs(embed_dir)
		embed_dir = embed_dir + '/' + str(args.seed) + '/'
		if not os.path.exists(embed_dir):
			os.makedirs(embed_dir)	

		embed4q_dir = "embedding4q"+'/'
		if not os.path.exists(embed4q_dir):
			os.makedirs(embed4q_dir)
		embed4q_dir = embed4q_dir + '/' + experiment_id_v2 + '/'
		if not os.path.exists(embed4q_dir):
			os.makedirs(embed4q_dir)
		embed4q_dir = embed4q_dir + '/' + env_name  + '/'
		if not os.path.exists(embed4q_dir):
			os.makedirs(embed4q_dir)
		embed4q_dir = embed4q_dir + '/' + str(args.seed) + '/'
		if not os.path.exists(embed4q_dir):
			os.makedirs(embed4q_dir)	

	if params["save_velocity"]:
		velocity_dir = "velocity"+'/'
		if not os.path.exists(velocity_dir):
			os.makedirs(velocity_dir)
		velocity_dir = velocity_dir + '/' + experiment_id_v2 + '/'
		if not os.path.exists(velocity_dir):
			os.makedirs(velocity_dir)
		velocity_dir = velocity_dir + '/' + env_name  + '/'
		if not os.path.exists(velocity_dir):
			os.makedirs(velocity_dir)
		velocity_dir = velocity_dir + '/' + str(args.seed) + '/'
		if not os.path.exists(velocity_dir):
			os.makedirs(velocity_dir)	

		average_v_csv_path = velocity_dir+ "/average_velocity.csv"
		average_v_file = open(average_v_csv_path,"a")
		average_v_writer = csv.writer(average_v_file)
		average_v_writer.writerow(["task","v_mean","v_std"])

	pre_embeddings=[]
	pre_embedding=torch.Tensor([0.91391444,-1.0774294,0.29897562]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)
	pre_embedding=torch.Tensor([0.82457685,-0.4906419,0.5033239]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)
	pre_embedding=torch.Tensor([-1.6671668,-1.9543726,0.09565737]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)
	pre_embedding=torch.Tensor([-1.9076895,0.55813557,-0.26508763]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)
	pre_embedding=torch.Tensor([-1.0795037,-1.242666,1.1746824]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)
	pre_embedding=torch.Tensor([0.37897086,-1.7998953,-0.5839664]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)
	pre_embedding=torch.Tensor([0.36660576,-1.1392477,-0.75051]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)
	pre_embedding=torch.Tensor([0.06714192,-0.8505497,-0.8916967]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)
	pre_embedding=torch.Tensor([-0.25607798,-0.48358324,-1.0271046]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)
	pre_embedding=torch.Tensor([-0.9829124,-0.4305004,-1.0443646]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)
	pre_embedding=torch.Tensor([-0.26289868,0.32641464,1.1901977]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)
	pre_embedding=torch.Tensor([-0.9261849,0.006673306,1.2089044]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)
	embeddings=[]
	for i in range(11):
		embedding = (pre_embeddings[i]+pre_embeddings[i+1])/2
		embeddings.append(embedding)
	embedding = (pre_embeddings[11]+pre_embeddings[0])/2
	embeddings.append(embedding)
 
	pre_embeddings4q=[]
	pre_embedding4q=torch.Tensor([0.0024124333,0.0007107592,-0.0076082433,0.0034545814,0.0038565355,-0.0071173944,-0.0016199141,0.004400964,0.0018969257,-0.005828591,0.00055183773,-0.0016757401]).unsqueeze(0)
	pre_embeddings4q.append(pre_embedding4q)
	pre_embedding4q=torch.Tensor([0.0015623689,-0.0007958163,-0.007604964,0.0061809076,0.003562076,-0.0049662525,-0.0021639576,0.0049959137,0.0020056544,-0.007465657,0.0003324512,-0.0014514563]).unsqueeze(0)
	pre_embeddings4q.append(pre_embedding4q)
	pre_embedding4q=torch.Tensor([0.00077094877,0.0005144094,-0.009687866,0.0051149516,0.004010362,-0.0037882351,0.00046436046,0.00354262,-0.0015447251,-0.0062691765,0.00037945877,-0.0038112402]).unsqueeze(0)
	pre_embeddings4q.append(pre_embedding4q)
	pre_embedding4q=torch.Tensor([0.0024197754,0.0008162736,-0.0078103244,0.005074379,0.0033856076,-0.00551164,0.0009554983,0.003809845,0.00075760786,-0.0065974025,0.002136918,-0.0027757282]).unsqueeze(0)
	pre_embeddings4q.append(pre_embedding4q)
	pre_embedding4q=torch.Tensor([0.0021639476,0.0017928535,-0.0077921757,0.006481872,0.002438161,-0.0051513435,-0.0014609301,0.0040502306,0.0016231956,-0.0076003326,0.0015197434,-0.002371945]).unsqueeze(0)
	pre_embeddings4q.append(pre_embedding4q)
	pre_embedding4q=torch.Tensor([0.001569144,-0.00023291772,-0.0073467325,0.0056789955,0.0021916232,-0.004600441,0.00010959874,0.0038533537,0.00074748043,-0.005646593,0.001005145,-0.001181378]).unsqueeze(0)
	pre_embeddings4q.append(pre_embedding4q)
	pre_embedding4q=torch.Tensor([0.002209358,-0.000275427,-0.008185705,0.0044727316,0.0024819425,-0.0067694588,0.0004989442,0.0035088146,-5.8100093e-05,-0.0064638024,0.001214704,-0.003155149]).unsqueeze(0)
	pre_embeddings4q.append(pre_embedding4q)
	pre_embedding4q=torch.Tensor([0.0017795785,0.0016829093,-0.007126414,0.0048793266,0.004972908,-0.0053185434,-4.019006e-05,0.004891864,0.00017136289,-0.006844638,0.0009034239,-0.0013043723]).unsqueeze(0)
	pre_embeddings4q.append(pre_embedding4q)
	pre_embedding4q=torch.Tensor([0.0031121238,-0.00014476175,-0.0067942655,0.00527372,0.004677874,-0.005621027,-0.0019711899,0.003736692,-0.0009469007,-0.0066319117,0.0012307703,-0.00399597]).unsqueeze(0)
	pre_embeddings4q.append(pre_embedding4q)
	pre_embedding4q=torch.Tensor([0.001950722,0.0007393999,-0.009704115,0.006785209,0.0027901945,-0.0067291353,-0.001328749,0.0028258194,0.0018280726,-0.0059109693,0.0013215987,-0.0019313393]).unsqueeze(0)
	pre_embeddings4q.append(pre_embedding4q)
	pre_embedding4q=torch.Tensor([0.0003365836,0.00040529727,-0.008483228,0.0051165894,0.0011356312,-0.0056261793,-0.0003934605,0.0036848066,-6.738282e-05,-0.006597656,0.0004311402,3.1390933e-05]).unsqueeze(0)
	pre_embeddings4q.append(pre_embedding4q)
	pre_embedding4q=torch.Tensor([-0.00049414346,0.00028446922,-0.008214628,0.0057649575,0.0049274317,-0.0043454077,-0.00042496156,0.0034471892,0.0006012074,-0.0045809145,0.0025877326,-0.0016192016]).unsqueeze(0)
	pre_embeddings4q.append(pre_embedding4q)
	embeddings4q=[]
	for i in range(11):
		embedding4q = (pre_embeddings4q[i]+pre_embeddings4q[i+1])/2
		embeddings4q.append(embedding4q)
	embedding4q = (pre_embeddings4q[11]+pre_embeddings4q[0])/2
	embeddings4q.append(embedding4q)

	for i in range(task_num):
		if params["save_embedding"]:
			embed_csv_path = embed_dir + '/' + task_list[i] + ".csv"
			embed_file = open(embed_csv_path, "w")
			embed_writer = csv.writer(embed_file)
			embed4q_csv_path = embed4q_dir + '/' + task_list[i] + ".csv"
			embed4q_file = open(embed4q_csv_path, "w")
			embed4q_writer = csv.writer(embed4q_file)
		if params["save_velocity"]:
			velocity_csv_path = velocity_dir+ '/' + task_list[i] + ".csv"
			velocity_file = open(velocity_csv_path,'w')
			velocity_writer = csv.writer(velocity_file)
		embedding=embeddings[i]
		embedding4q=embeddings4q[i]
		ob=env.reset()
		with torch.no_grad():
			for t in range(1, max_ep_len+1):
				representation = pf_state.forward(torch.Tensor( ob ).to("cpu").unsqueeze(0))
				out=pf_action.explore(representation,embedding)
				act=out["action"]
				act = act.detach().cpu().numpy()
				next_ob, _, done, info = env.step(act)
				if params["save_velocity"]:
					x_velocity = info['x_velocity']
					velocity_writer.writerow([x_velocity])
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
			
			print(task_list[i], ":", dir)

		if params["save_embedding"]:
			embedding = embedding.squeeze(0)
			embedding = embedding.detach().cpu().numpy()
			embed_writer.writerow(embedding)
			embed_file.close()
			embedding4q = embedding4q.squeeze(0)
			embedding4q = embedding4q.detach().cpu().numpy()
			embed4q_writer.writerow(embedding4q)
			embed4q_file.close()
		if params["save_velocity"]:
			velocity_file.close()
			velocity_file = open(velocity_csv_path,'r')
			velocity_list = np.loadtxt(velocity_file)
			velocity_list = velocity_list[100:]
			average_v_writer.writerow([task_list[i], np.mean(velocity_list), np.std(velocity_list)])


	env.close()











######################## generate gif from saved images ########################

def save_gif(env_name):

	print("============================================================================================")

	gif_num = args.seed    
	experiment_id=str(args.id)

	# adjust following parameters to get desired duration, size (bytes) and smoothness of gif
	total_timesteps = 250
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
	max_ep_len = 1000          
	save_gif_images(env_name,  max_ep_len)
	# save_gif(env_name)



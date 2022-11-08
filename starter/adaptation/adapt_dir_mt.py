import sys
import csv


sys.path.append(".") 

import torch
import os
import time
import os.path as osp

import numpy as np

from torchrl.utils import get_args
from torchrl.utils import get_params
from torchrl.env import get_env
import torch.nn.functional as F

args = get_args()
params = get_params(args.config)

import torchrl.policies as policies
import torchrl.networks as networks
import gym




def experiment(args):

    device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")
    env=gym.make(params['env_name'])
    task_list=["direction_77_finetune_random"]
    target = 77
    task_num=len(task_list)
    input_shape = 24

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.backends.cudnn.deterministic=True
    

    experiment_name = os.path.split( os.path.splitext( args.config )[0] )[-1] if args.id is None \
        else args.id
    experiment_name_v2 = experiment_name + "_finetune"

    
    params['net']['base_type']=networks.MLPBase

    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

   
    
    embedding = F.normalize(torch.randn((1, input_shape)))
    # embedding4q = torch.Tensor([0.93127483,0.752826,-4.8544803]).unsqueeze(0)

    
    
    pf=policies.GuassianContPolicy(
        input_shape = env.observation_space.shape[0] + input_shape,
        output_shape = 2 * env.action_space.shape[0],
        **params['net'] 
    )
    
    model_dir = "log/"+experiment_name+"/"+ "Ant-Dir" +"/"+str(args.seed)+"/model/"
    # pf_state.load_state_dict(torch.load(model_dir+"model_pf_state_finish.pth", map_location='cpu'))
    # pf_action.load_state_dict(torch.load(model_dir+"model_pf_action_finish.pth", map_location='cpu'))
    
    # pf_state.load_state_dict(torch.load(model_dir+"model_pf_state_8060.pth", map_location='cpu'))
    # pf_action.load_state_dict(torch.load(model_dir+"model_pf_action_8060.pth", map_location='cpu'))
    
    # pf_state.load_state_dict(torch.load(model_dir+"model_pf_state_best.pth", map_location='cpu'))
    pf.load_state_dict(torch.load(model_dir+"model_pf_best.pth", map_location='cpu'))

    
    num_epoch = 10
    num_sample = 15
    num_best = 6
    ep = 2
    
    mean = torch.zeros(((num_sample-1) * num_best, 1, input_shape))
    std = torch.full(((num_sample-1) * num_best, 1, input_shape), 1)
    best_embeddings = embedding.repeat(num_best, 1, 1)
    zeros = torch.zeros_like(best_embeddings)
    
    # cmp_flag = 0
    embed_dir = "log/" + experiment_name_v2 + '/' + params['env_name'] + '/' + str(args.seed)+"/model/"
    if not os.path.exists(embed_dir):
        os.makedirs(embed_dir)
    log_dir =  "log/" + experiment_name_v2 + '/' + params['env_name'] + '/' + str(args.seed)   
    log_csv_path = log_dir + '/' + task_list[0] + "_log.csv"
    log_file = open(log_csv_path, "w")
    log_header = ['Epoch', 'Reward', 'Direction']
    log_writer = csv.DictWriter(log_file, log_header)
    log_writer.writeheader()
    
    with torch.no_grad():
        for i in range(num_best * num_sample):
            eval_reward = 0
            sum_vel = 0
            ob = env.reset()
            for t in range(200):
                # representation = pf_state.forward(torch.Tensor(ob).unsqueeze(0).to("cpu"))
                # out = pf_action.explore(representation, embedding)
                out = pf.explore(torch.cat((torch.Tensor( ob ).to("cpu").unsqueeze(0),embedding),dim=-1))
                action = out["action"]
                action = action.detach().cpu().numpy()
                next_ob, rewards, done, info = env.step(action)
                reward = rewards[0]
                eval_reward += reward
                if t >= 100:
                    sum_vel += info['x_velocity']
                ob = next_ob
                if done:
                    break
            x = info['x_position']
            y = info['y_position']
            dir = np.arctan(y/x)/ np.pi * 180
            if x < 0 and y > 0:
                dir += 180
            elif x < 0 and y < 0:
                dir += 180
            elif x > 0 and y < 0:
                dir += 360
            dir_error = -abs(dir - target)
            if dir_error < -180:
                dir_error = -360 - dir_error
            # if dir_error > -ep:
            #     dir_error = 0
    log_writer.writerows([{'Epoch': 0, 'Reward': eval_reward, 'Direction': dir}])
    
    
    
    
    
    
    
    
    for epoch in range(num_epoch):
        # print(best_embeddings)
        sample_embeddings = best_embeddings.repeat(num_sample, 1, 1)
        
        noises = torch.normal(mean, std)
        noises = torch.cat((zeros, noises), dim=0)
        sample_embeddings = F.normalize(sample_embeddings + noises, dim=-1)
        embedding_info = []
        with torch.no_grad():
            for i in range(num_best * num_sample):
                eval_reward = 0
                sum_vel = 0
                ob = env.reset()
                for t in range(200):
                    # representation = pf_state.forward(torch.Tensor(ob).unsqueeze(0).to("cpu"))
                    # out = pf_action.explore(representation, sample_embeddings[i])
                    out = pf.explore(torch.cat((torch.Tensor( ob ).to("cpu").unsqueeze(0),embedding),dim=-1))
                    action = out["action"]
                    action = action.detach().cpu().numpy()
                    next_ob, rewards, done, info = env.step(action)
                    reward = rewards[0]
                    eval_reward += reward
                    if t >= 100:
                        sum_vel += info['x_velocity']
                    ob = next_ob
                    if done:
                        break
                x = info['x_position']
                y = info['y_position']
                dir = np.arctan(y/x)/ np.pi * 180
                if x < 0 and y > 0:
                    dir += 180
                elif x < 0 and y < 0:
                    dir += 180
                elif x > 0 and y < 0:
                    dir += 360
                dir_error = -abs(dir - target)
                if dir_error < -180:
                    dir_error = -360 - dir_error
                if dir_error > -ep:
                    dir_error = 0
                embedding_info.append({'id': i, 'eval_reward': eval_reward, 'dir': dir,'dir_error': dir_error})
                
                
        # embedding_info = sorted(embedding_info, key = lambda x:(100 * x['dir_error'] + x['eval_reward']), reverse=True)
        # embedding_info = sorted(embedding_info, key = lambda x:(200 * x['dir_error'] + x['eval_reward']), reverse=True)
        embedding_info = sorted(embedding_info, key = lambda x:x['eval_reward'], reverse=True)
        

        
        
        # if cmp_flag:
        #     embedding_info = sorted(embedding_info, key = lambda x:(x['dir_error'], x['eval_reward']), reverse=True)
        # if not cmp_flag:
        #     cmp_flag = 1
        #     for idx in range(num_best):
        #         cmp_flag = cmp_flag and (embedding_info[idx]["eval_reward"] > 4000)
        
        print("----------------------------------------------------------------------------------")
        print(task_list[0], ":", "Epoch", epoch+1)
        # print("cmp_flag:", cmp_flag)
        sum_reward = 0
        sum_dir = 0
        for j in range(num_best):
            best_embeddings[j] = sample_embeddings[embedding_info[j]['id']]
            sum_reward += embedding_info[j]['eval_reward']
            sum_dir += embedding_info[j]['dir']
            print("Best", j+1, "th direction:", embedding_info[j]['dir'])
            print("Best", j+1, "th reward:", embedding_info[j]['eval_reward'])
            # print("Best", j+1, "th embedding:", best_embeddings[j].squeeze(0).detach().cpu().numpy())
        print("----------------------------------------------------------------------------------") 
        average_reward = sum_reward/ num_best
        average_dir = sum_dir/ num_best
        log_writer.writerows([{'Epoch': epoch+1, 'Reward': embedding_info[0]['eval_reward'], 'Direction': embedding_info[0]['dir']}])
        # log_writer.writerows([{'Epoch': epoch+1, 'Reward': average_reward, 'Direction': average_dir}])
        
    log_file.close()
    
    
    
    embed_csv_path = embed_dir + task_list[0] + "_finish.csv"
    embed_file = open(embed_csv_path, "w")
    embed_writer = csv.writer(embed_file)
    embedding = best_embeddings[0].squeeze(0).detach().cpu().numpy()
    embed_writer.writerow(embedding)
    embed_file.close()
    


if __name__ == "__main__":
    experiment(args)



    
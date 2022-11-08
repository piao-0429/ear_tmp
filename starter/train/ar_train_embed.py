import sys
import csv


sys.path.append(".") 

import torch
from torch.autograd import Variable
import os
import time
import os.path as osp

import numpy as np

from torchrl.utils import get_args
from torchrl.utils import get_params
from torchrl.env import get_env

from torchrl.utils import Logger

args = get_args()
params = get_params(args.config)

import torchrl.policies as policies
import torchrl.networks as networks
from torchrl.algo.off_policy.ar_sac_embed import ARSAC_Embed
from torchrl.collector.para.async_mt import AsyncMultiTaskParallelCollectorForActionRepresentation_Embed
from torchrl.replay_buffers.shared import AsyncSharedReplayBuffer
import gym




def experiment(args):

    device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")
    env=gym.make(params['env_name'])
    task_list=["forward_3.5_embed"]
    task_num=len(task_list)
    representation_shape= params['representation_shape']
    embedding_shape=params['embedding_shape']



    env.reset(seed = args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.backends.cudnn.deterministic=True
    
    buffer_param = params['replay_buffer']

    experiment_name = os.path.split( os.path.splitext( args.config )[0] )[-1] if args.id is None \
        else args.id
    experiment_name_v2 = experiment_name + "_embed"

    logger = Logger( experiment_name_v2 , params['env_name'], args.seed, params, args.log_dir )

    params['general_setting']['env'] = env
    params['general_setting']['logger'] = logger
    params['general_setting']['device'] = device

    params['p_state_net']['base_type']=networks.MLPBase
    params['p_action_net']['base_type']=networks.MLPBase
    params['q_net']['base_type']=networks.MLPBase

    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    pf_state = networks.Net(
        input_shape=env.observation_space.shape[0], 
        output_shape=representation_shape,
        **params['p_state_net']
    )
    
    embedding = torch.Tensor([0.67270404,2.8474584,-4.0545588])
    embedding = embedding.to(device).requires_grad_()
    
    pf_action=policies.ActionRepresentationGuassianContPolicy(
        input_shape = representation_shape + embedding_shape,
        output_shape = 2 * env.action_space.shape[0],
        **params['p_action_net'] 
    )
    qf1 = networks.FlattenNet( 
        input_shape = env.observation_space.shape[0] + env.action_space.shape[0] + task_num,
        output_shape = 1,
        **params['q_net'] )
    qf2 = networks.FlattenNet( 
        input_shape = env.observation_space.shape[0] + env.action_space.shape[0] + task_num,
        output_shape = 1,
        **params['q_net'] )
    
    model_dir = "log/"+experiment_name+"/"+params['env_name']+"/"+str(args.seed)+"/model/"
    pf_state.load_state_dict(torch.load(model_dir+"model_pf_state_finish.pth", map_location='cpu'))
    pf_action.load_state_dict(torch.load(model_dir+"model_pf_action_finish.pth", map_location='cpu'))

    example_ob = env.reset()
    example_dict = { 
        "obs": example_ob,
        "next_obs": example_ob,
        "acts": env.action_space.sample(),
        "rewards": [0],
        "terminals": [False],
        "task_idxs": [0],
        "task_inputs": np.zeros(task_num),
        "embeddings":np.zeros(embedding_shape)
    }
    
    replay_buffer = AsyncSharedReplayBuffer( int(buffer_param['size']),
            args.worker_nums
    )
    replay_buffer.build_by_example(example_dict)

    params['general_setting']['replay_buffer'] = replay_buffer

    epochs = params['general_setting']['pretrain_epochs'] + \
        params['general_setting']['num_epochs']

    params['general_setting']['collector'] = AsyncMultiTaskParallelCollectorForActionRepresentation_Embed(
        env=env, pf=[pf_state,pf_action], embedding = embedding, replay_buffer=replay_buffer,
        task_list=task_list,
        task_args={},
        device=device,
        reset_idx=True,
        epoch_frames=params['general_setting']['epoch_frames'],
        max_episode_frames=params['general_setting']['max_episode_frames'],
        eval_episodes = params['general_setting']['eval_episodes'],
        worker_nums=args.worker_nums, eval_worker_nums=args.eval_worker_nums,
        train_epochs = epochs, eval_epochs= params['general_setting']['num_epochs']
    )
    params['general_setting']['batch_size'] = int(params['general_setting']['batch_size'])
    params['general_setting']['save_dir'] = osp.join(logger.work_dir,"model")
    agent = ARSAC_Embed(
        pf_state = pf_state,
        pf_action = pf_action,
        embedding = embedding,
        qf1 = qf1,
        qf2 = qf2,
        task_nums=len(task_list),
        **params['sac'],
        **params['general_setting']
    )
    agent.train()
    embed_dir = "log/" + experiment_name_v2 + '/' + params['env_name'] + '/' + str(args.seed)+"/model/"
    embed_csv_path = embed_dir + '/' + task_list[0] + "_finish.csv"
    embed_file = open(embed_csv_path, "w")
    embed_writer = csv.writer(embed_file)
    embedding = agent.embedding.detach().cpu().numpy()
    embed_writer.writerow(embedding)
    embed_file.close()


if __name__ == "__main__":
    experiment(args)



    
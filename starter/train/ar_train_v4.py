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

from torchrl.utils import Logger

args = get_args()
params = get_params(args.config)

import torchrl.policies as policies
import torchrl.networks as networks
from torchrl.algo import ARSAC_v4
from torchrl.collector.para.async_mt import AsyncMultiTaskParallelCollectorForActionRepresentation
from torchrl.replay_buffers.shared import AsyncSharedReplayBuffer
import gym




def experiment(args):

    device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")
    env=gym.make(params['env_name'])
   
    # # task_list for Ant-Vel
    # task_list=["forward_1", "forward_2", "forward_3", "forward_4", "forward_5", "backward_1", "backward_2", "backward_3", "backward_4", "backward_5"]
    
    # task_list for Ant-Dir
    task_list = ["direction_0", "direction_30", "direction_60", "direction_90", "direction_120", "direction_150", "direction_180", "direction_210", "direction_240", "direction_270", "direction_300", "direction_330"]
    # task_list = ["direction_0", "direction_15", "direction_30", "direction_45", "direction_60", "direction_75", "direction_90", "direction_105", "direction_120", "direction_135", "direction_150", "direction_165"]
    
    
    task_num = len(task_list)
    representation_shape = params['representation_shape']
    embedding_shape = params['embedding_shape']
    embedding4q_shape = params['embedding4q_shape']



    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.backends.cudnn.deterministic=True
    
    buffer_param = params['replay_buffer']

    experiment_name = os.path.split( os.path.splitext( args.config )[0] )[-1] if args.id is None \
        else args.id
    logger = Logger( experiment_name , params['env_name'], args.seed, params, args.log_dir )

    params['general_setting']['env'] = env
    params['general_setting']['logger'] = logger
    params['general_setting']['device'] = device

    params['p_state_net']['base_type']=networks.MLPBase
    params['task_net']['base_type']=networks.MLPBase
    params['p_action_net']['base_type']=networks.MLPBase
    params['q_net']['base_type']=networks.MLPBase

    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    pf_state = networks.Net(
        input_shape=env.observation_space.shape[0], 
        output_shape=representation_shape,
        **params['p_state_net']
    )

    pf_task=networks.Net(
        input_shape=task_num, 
        output_shape=embedding_shape,
        **params['task_net']
    )

    pf_action=policies.ActionRepresentationGuassianContPolicy(
        input_shape = representation_shape + embedding_shape,
        output_shape = 2 * env.action_space.shape[0],
        **params['p_action_net'] 
    )
    
    qf_task=networks.Net(
        input_shape=task_num, 
        output_shape=embedding4q_shape,
        **params['task_net']
    )

    
    qf1 = networks.FlattenNet( 
        input_shape = env.observation_space.shape[0] + env.action_space.shape[0] + embedding4q_shape,
        output_shape = 1,
        **params['q_net'] 
    )
    qf2 = networks.FlattenNet( 
        input_shape = env.observation_space.shape[0] + env.action_space.shape[0] + embedding4q_shape,
        output_shape = 1,
        **params['q_net'] 
    )
    
    example_ob = env.reset()
    example_dict = { 
        "obs": example_ob,
        "next_obs": example_ob,
        "acts": env.action_space.sample(),
        "rewards": [0],
        "terminals": [False],
        "task_idxs": [0],
        "task_inputs": np.zeros(task_num),
    }
    
    replay_buffer = AsyncSharedReplayBuffer( int(buffer_param['size']),
            args.worker_nums
    )
    replay_buffer.build_by_example(example_dict)

    params['general_setting']['replay_buffer'] = replay_buffer

    epochs = params['general_setting']['pretrain_epochs'] + \
        params['general_setting']['num_epochs']

    params['general_setting']['collector'] = AsyncMultiTaskParallelCollectorForActionRepresentation(
        env=env, pf=[pf_state, pf_task, pf_action], replay_buffer=replay_buffer,
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

    agent = ARSAC_v4(
        pf_state = pf_state,
        pf_task = pf_task,
        pf_action = pf_action,
        qf_task=qf_task,
        qf1 = qf1,
        qf2 = qf2,
        task_nums = task_num,
        **params['sac'],
        **params['general_setting']
    )
    agent.train()
    

if __name__ == "__main__":
    experiment(args)



    
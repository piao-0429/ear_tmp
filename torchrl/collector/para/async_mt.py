import csv
import torch
import copy
import numpy as np

from .base import AsyncParallelCollector
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchrl.policies as policies

from torchrl.env.get_env import *
from torchrl.env.continuous_wrapper import *
from collections import OrderedDict


class AsyncSingleTaskParallelCollector(AsyncParallelCollector):
    def __init__(
            self,
            reset_idx=False,
            **kwargs):
        self.reset_idx = reset_idx
        super().__init__(**kwargs)

    @staticmethod
    def eval_worker_process(
            shared_pf, env_info, shared_que, start_barrier, epochs, reset_idx):

        pf = copy.deepcopy(shared_pf).to(env_info.device)

        # Rebuild Env
        env_info.env = env_info.env_cls(**env_info.env_args)

        # env_info.env.eval()
        # env_info.env._reward_scale = 1
        current_epoch = 0
        while True:
            start_barrier.wait()
            current_epoch += 1
            if current_epoch > epochs:
                break
            pf.load_state_dict(shared_pf.state_dict())

            eval_rews = []

            done = False
    
            for idx in range(env_info.eval_episodes):
                if reset_idx:
                    eval_ob = env_info.env.reset_with_index(idx)
                else:
                    eval_ob = env_info.env.reset()
                rew = 0
        
                while not done:
                    act = pf.eval_act( torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0))
                    act = act.detach().cpu().numpy()
                    eval_ob, r, done, info = env_info.env.step( act )
                    rew += r
                    if env_info.eval_render:
                        env_info.env.render()

                eval_rews.append(rew)
                done = False
        

            shared_que.put({
                'eval_rewards': eval_rews,
            })

    def start_worker(self):
        self.workers = []
        self.shared_que = self.manager.Queue(self.worker_nums)
        self.start_barrier = mp.Barrier(self.worker_nums)
    
        self.eval_workers = []
        self.eval_shared_que = self.manager.Queue(self.eval_worker_nums)
        self.eval_start_barrier = mp.Barrier(self.eval_worker_nums)

        self.env_info.env_cls  = self.env_cls
        self.env_info.env_args = self.env_args

        for i in range(self.worker_nums):
            self.env_info.task_rank = i
            p = mp.Process(
                target=self.__class__.train_worker_process,
                args=( self.__class__, self.shared_funcs,
                    self.env_info, self.replay_buffer, 
                    self.shared_que, self.start_barrier,
                    self.train_epochs))
            p.start()
            self.workers.append(p)

        for i in range(self.eval_worker_nums):
            eval_p = mp.Process(
                target=self.__class__.eval_worker_process,
                args=(self.shared_funcs["pf"],
                    self.env_info, self.eval_shared_que, self.eval_start_barrier,
                    self.eval_epochs, self.reset_idx))
            eval_p.start()
            self.eval_workers.append(eval_p)

    def eval_one_epoch(self):
        # self.eval_start_barrier.wait()
        eval_rews = []
        
        self.shared_funcs["pf"].load_state_dict(self.funcs["pf"].state_dict())
        for _ in range(self.eval_worker_nums):
            worker_rst = self.eval_shared_que.get()
            eval_rews += worker_rst["eval_rewards"]
           

        return {
            'eval_rewards':eval_rews,
        }


class AsyncMultiTaskParallelCollectorUniform(AsyncSingleTaskParallelCollector):

    def __init__(self, progress_alpha=0.1, **kwargs):
        super().__init__(**kwargs)
        self.tasks=self.task_list
        self.tasks_mapping = {}
        for idx, task_name in enumerate(self.tasks):
            self.tasks_mapping[task_name] = idx
        self.tasks_progress = [0 for _ in range(len(self.tasks))]
        self.progress_alpha = progress_alpha
  

    @classmethod
    def take_actions(cls, funcs, env_info, ob_info, replay_buffer):

        pf = funcs["pf"]
        ob = ob_info["ob"]
        task_idx = env_info.task_rank
        idx_flag = isinstance(pf, policies.MultiHeadGuassianContPolicy)
        embedding_flag = isinstance(pf, policies.ActionRepresentationGuassianContPolicy) or isinstance(pf, policies.EmbeddingGuassianContPolicyBase) or isinstance(pf, policies.EmbeddingDetContPolicyBase)

        

        pf.eval()
        task_input = torch.zeros(env_info.num_tasks)
        task_input[env_info.task_rank] = 1
        task_input = task_input.unsqueeze(0).to(env_info.device)
        with torch.no_grad():
            if idx_flag:
                idx_input = torch.Tensor([[task_idx]]).to(env_info.device).long()
                if embedding_flag:
        
                    out = pf.explore(torch.Tensor( ob ).to(env_info.device).unsqueeze(0), task_input,
                        [task_idx])
                else:
                    # print(torch.Tensor( ob ).to(env_info.device).unsqueeze(0))
                    # print(idx_input)
                    out = pf.explore(torch.Tensor( ob ).to(env_info.device).unsqueeze(0),
                        idx_input)
                act = out["action"]
                # act = act[0]
            else:
                if embedding_flag:
                  
                    out = pf.explore(torch.Tensor( ob ).to(env_info.device).unsqueeze(0), task_input)
                else:    
                    # print(torch.cat((torch.Tensor( ob ).to(env_info.device).unsqueeze(0)),task_input,dim=-1))
                    out = pf.explore(torch.cat((torch.Tensor( ob ).to(env_info.device).unsqueeze(0),task_input),dim=-1))
                act = out["action"]


        act = act.detach().cpu().numpy()
        if not env_info.continuous:
            act = act[0]
        
        if type(act) is not int:
            if np.isnan(act).any():
                print("NaN detected. BOOM")
                exit()

        next_ob, rewards, done, info = env_info.env.step(act)
        reward=rewards[task_idx]
        if env_info.train_render:
            env_info.env.render()
        env_info.current_step += 1

        sample_dict = {
            "obs": ob,
            "next_obs": next_ob,
            "acts": act,
            "task_idxs": [env_info.task_rank],
            "rewards": [reward],
            "terminals": [done]
        }
        if embedding_flag:
            sample_dict["task_inputs"] = task_input.cpu().numpy()

        if done or env_info.current_step >= env_info.max_episode_frames:
            next_ob = env_info.env.reset()
            env_info.finish_episode()
            env_info.start_episode() # reset current_step

        replay_buffer.add_sample( sample_dict, env_info.task_rank)

        return next_ob, done, reward, info

    @staticmethod
    def train_worker_process(cls, shared_funcs, env_info,
        replay_buffer, shared_que,
        start_barrier, epochs, start_epoch, task_name, shared_dict):

        replay_buffer.rebuild_from_tag()
        local_funcs = copy.deepcopy(shared_funcs)
        for key in local_funcs:
            local_funcs[key].to(env_info.device)
        
        c_ob = {
            "ob": env_info.env.reset()
        }
        train_rew = 0
        current_epoch = 0
        while True:
            start_barrier.wait()
            current_epoch += 1
            if current_epoch < start_epoch:
                shared_que.put({
                    'train_rewards': None,
                    'train_epoch_reward': None
                })
                continue
            if current_epoch > epochs:
                break

            for key in shared_funcs:
                local_funcs[key].load_state_dict(shared_funcs[key].state_dict())

            train_rews = []
            train_epoch_reward = 0    

            for t in range(env_info.epoch_frames):

                next_ob, done, reward, info = cls.take_actions(local_funcs, env_info, c_ob, replay_buffer )
                c_ob["ob"] = next_ob
                train_rew += reward
                train_epoch_reward += reward
                if done or t == env_info.epoch_frames-1:
                    train_rews.append(train_rew)
                    train_rew = 0

            # if norm_obs_flag:
            #     shared_dict[task_name] = {
            #         "obs_mean": env_info.env._obs_mean,
            #         "obs_var": env_info.env._obs_var
            #     }
                # print("Put", task_name)
            
            shared_que.put({
                'train_rewards':train_rews,
                'train_epoch_reward':train_epoch_reward
            })

    @staticmethod
    def eval_worker_process(shared_pf, 
        env_info, shared_que, start_barrier, epochs, start_epoch, task_name, shared_dict):

        pf = copy.deepcopy(shared_pf).to(env_info.device)
        idx_flag = isinstance(pf, policies.MultiHeadGuassianContPolicy)
        embedding_flag = isinstance(pf, policies.ActionRepresentationGuassianContPolicy) or isinstance(pf, policies.EmbeddingGuassianContPolicyBase) or isinstance(pf, policies.EmbeddingDetContPolicyBase)


        # # Rebuild Env
        # env_info.env = env_info.env_cls(**env_info.env_args)

        # norm_obs_flag = env_info.env_args["env_params"]["obs_norm"]

        # env_info.env.eval()
        # env_info.env._reward_scale = 1
        current_epoch = 0
        while True:
            start_barrier.wait()
            current_epoch += 1
            if current_epoch < start_epoch:
                shared_que.put({
                    'eval_rewards': None,
                    'task_name': task_name
                })
                continue
            if current_epoch > epochs:
                break
            pf.load_state_dict(shared_pf.state_dict())
            pf.eval()

            # print("Get", task_name)
            # if norm_obs_flag:
            #     env_info.env._obs_mean = shared_dict[task_name]["obs_mean"]
            #     env_info.env._obs_var = shared_dict[task_name]["obs_var"]
                # print(env_info.env._obs_mean)
                #  = {
                #     "obs_mean": env_info.env._obs_mean,
                #     "obs_var": env_info.env._obs_var
                # }

            eval_rews = []  

            done = False
          
            for idx in range(env_info.eval_episodes):

                eval_ob = env_info.env.reset()
                rew = 0

                task_idx = env_info.task_rank
                t = 0
                task_input = torch.zeros(env_info.num_tasks)
                task_input[env_info.task_rank] = 1
                task_input = task_input.unsqueeze(0).to(env_info.device)
                while not done:
                    t += 1
                    if idx_flag:
                        idx_input = torch.Tensor([[task_idx]]).to(env_info.device).long()
                        if embedding_flag:
                            act = pf.eval_act( torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0), task_input, [task_idx] )
                        else:
                            act = pf.eval_act( torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0), idx_input )
                    else:
                        if embedding_flag:
                            act = pf.eval_act( torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0), task_input)
                        else:
                            # print(torch.cat((torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0),task_input),dim=-1))
                            act = pf.eval_act( torch.cat((torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0),task_input),dim=-1))

                    eval_ob, rs, done, info = env_info.env.step( act )
                    r=rs[task_idx]
                    rew += r
                    if env_info.eval_render:
                        env_info.env.render()
                    done = done or (t >= env_info.max_episode_frames)

                eval_rews.append(rew)
                done = False
      
            shared_que.put({
                'eval_rewards': eval_rews,
             
                'task_name': task_name
            })

    def start_worker(self):
        self.workers = []
        self.shared_que = self.manager.Queue(self.worker_nums)
        self.start_barrier = mp.Barrier(self.worker_nums)
                
        self.eval_workers = []
        self.eval_shared_que = self.manager.Queue(self.eval_worker_nums)
        self.eval_start_barrier = mp.Barrier(self.eval_worker_nums)
        self.shared_dict = self.manager.dict()
        assert self.worker_nums == len(self.task_list)

        self.env_info.env = self.env
        self.env_info.num_tasks = len(self.task_list)
       
        single_mt_env_args = {
            "task_name": None,
            "task_rank": 0,
            "num_tasks": len(self.task_list),
            "max_obs_dim": np.prod(self.env.observation_space.shape),
        }
        
      
        tasks=self.task_list
        for i, task in enumerate(tasks):
          
            
            self.env_info.task_rank = i
            
            self.env_info.env_args = single_mt_env_args
            self.env_info.env_args["task_name"] = task

            
            start_epoch = 0
            self.env_info.env_args["task_rank"] = i
            p = mp.Process(
                target=self.__class__.train_worker_process,
                args=( self.__class__, self.shared_funcs,
                    self.env_info, self.replay_buffer, 
                    self.shared_que, self.start_barrier,
                    self.train_epochs, start_epoch, task, self.shared_dict))
            p.start()
            self.workers.append(p)
            # i += 1



        assert self.eval_worker_nums == len(self.task_list)
      
        self.env_info.env = self.env
        self.env_info.num_tasks = len(self.task_list)
       
        single_mt_env_args = {
            "task_name": None,
            "task_rank": 0,
            "num_tasks": len(self.task_list),
            "max_obs_dim": np.prod(self.env.observation_space.shape),
        }

        for i, task in enumerate(tasks):
           

            self.env_info.task_rank = i

            self.env_info.env_args = single_mt_env_args
            self.env_info.env_args["task_name"] = task

            start_epoch = 0
  

            self.env_info.env_args["task_rank"] = i
            eval_p = mp.Process(
                target=self.__class__.eval_worker_process,
                args=(self.shared_funcs["pf"],
                    self.env_info, self.eval_shared_que, self.eval_start_barrier,
                    self.eval_epochs, start_epoch, task, self.shared_dict))
            eval_p.start()
            self.eval_workers.append(eval_p)


    def eval_one_epoch(self):
        
        eval_rews = []
      
        self.shared_funcs["pf"].load_state_dict(self.funcs["pf"].state_dict())

        tasks_result = []

        active_task_counts = 0
        for _ in range(self.eval_worker_nums):
            worker_rst = self.eval_shared_que.get()
            if worker_rst["eval_rewards"] is not None:
                active_task_counts += 1
                eval_rews += worker_rst["eval_rewards"]
                tasks_result.append((worker_rst["task_name"], 
                np.mean(worker_rst["eval_rewards"])))

        tasks_result.sort()

        dic = OrderedDict()
        for task_name, eval_rewards in tasks_result:
            
            dic[task_name+"_eval_rewards"] = eval_rewards
            self.tasks_progress[self.tasks_mapping[task_name]] *= \
                (1 - self.progress_alpha)

        dic['eval_rewards']      = eval_rews
        

        return dic


    def train_one_epoch(self):
        train_rews = []
        train_epoch_reward = 0

        for key in self.shared_funcs:
            self.shared_funcs[key].load_state_dict(self.funcs[key].state_dict())
        
        active_worker_nums = 0
        for _ in range(self.worker_nums):
            worker_rst = self.shared_que.get()
            
            if worker_rst["train_rewards"] is not None:
                train_rews += worker_rst["train_rewards"]
                train_epoch_reward += worker_rst["train_epoch_reward"]
                active_worker_nums += 1
        self.active_worker_nums = active_worker_nums
     
        return {
            'train_rewards':train_rews,
            'train_epoch_reward':train_epoch_reward
        }

class AsyncMultiTaskParallelCollectorForActionRepresentation(AsyncSingleTaskParallelCollector):

    def __init__(self, progress_alpha=0.1, **kwargs):
        super().__init__(**kwargs)
        self.tasks=self.task_list
        self.tasks_mapping = {}
        for idx, task_name in enumerate(self.tasks):
            self.tasks_mapping[task_name] = idx
        self.tasks_progress = [0 for _ in range(len(self.tasks))]
        self.progress_alpha = progress_alpha
        self.pf_state=self.pf[0]
        self.pf_task=self.pf[1]
        self.pf_action=self.pf[2]
  

    @classmethod
    def take_actions(cls, funcs, env_info, ob_info, replay_buffer):

        pf_state = funcs["pf_state"]
        pf_task = funcs["pf_task"]
        pf_action = funcs["pf_action"]
        ob = ob_info["ob"]
        task_idx = env_info.task_rank
        

        pf_state.eval()
        pf_task.eval()
        pf_action.eval()

        with torch.no_grad():
        
            idx_input = torch.Tensor([[task_idx]]).to(env_info.device).long()
            task_input = torch.zeros(env_info.num_tasks)
            task_input[env_info.task_rank] = 1
            task_input = task_input.to(env_info.device).unsqueeze(0)
            ob = torch.Tensor( ob ).to(env_info.device).unsqueeze(0)
            embedding = pf_task.forward(task_input)
            representation = pf_state.forward(ob)

            out = pf_action.explore(representation, embedding)
            
            act = out["action"]
            # act = act[0]
            


        act = act.detach().cpu().numpy()
        if not env_info.continuous:
            act = act[0]
        
        if type(act) is not int:
            if np.isnan(act).any():
                print("NaN detected. BOOM")
                exit()

        next_ob, rewards, done, info = env_info.env.step(act)
        reward=rewards[task_idx]
        if env_info.train_render:
            env_info.env.render()
        env_info.current_step += 1

        sample_dict = {
            "obs": ob,
            "next_obs": next_ob,
            "acts": act,
            "task_idxs": [env_info.task_rank],
            "rewards": [reward],
            "terminals": [done],
            "task_inputs": task_input.cpu().numpy()
        }
       

        if done or env_info.current_step >= env_info.max_episode_frames:
            next_ob = env_info.env.reset()
            env_info.finish_episode()
            env_info.start_episode() 

        replay_buffer.add_sample( sample_dict, env_info.task_rank)

        return next_ob, done, reward, info

    @staticmethod
    def train_worker_process(cls, shared_funcs, env_info,
        replay_buffer, shared_que,
        start_barrier, epochs, start_epoch, task_name, shared_dict):

        replay_buffer.rebuild_from_tag()
        local_funcs = copy.deepcopy(shared_funcs)
        for key in local_funcs:
            local_funcs[key].to(env_info.device)
        
        c_ob = {
            "ob": env_info.env.reset()
        }
        train_rew = 0
        current_epoch = 0
        while True:
            start_barrier.wait()
            current_epoch += 1
            if current_epoch < start_epoch:
                shared_que.put({
                    'train_rewards': None,
                    'train_epoch_reward': None
                })
                continue
            if current_epoch > epochs:
                break

            for key in shared_funcs:
                local_funcs[key].load_state_dict(shared_funcs[key].state_dict())
            # print(list(shared_funcs["pf_state"].parameters()))
            train_rews = []
            train_epoch_reward = 0    

            for t in range(env_info.epoch_frames):

                next_ob, done, reward, info = cls.take_actions(local_funcs, env_info, c_ob, replay_buffer )
                c_ob["ob"] = next_ob
                train_rew += reward
                train_epoch_reward += reward
                train_rews.append(train_rew)
                train_rew = 0
            
            shared_que.put({
                'train_rewards':train_rews,
                'train_epoch_reward':train_epoch_reward
            })

    @staticmethod
    def eval_worker_process(shared_pf_state,shared_pf_task,shared_pf_action, 
        env_info, shared_que, start_barrier, epochs, start_epoch, task_name, shared_dict):

        pf_state = copy.deepcopy(shared_pf_state).to(env_info.device)
        pf_task = copy.deepcopy(shared_pf_task).to(env_info.device)
        pf_action = copy.deepcopy(shared_pf_action).to(env_info.device)

        current_epoch = 0
        while True:
            start_barrier.wait()
            current_epoch += 1
            if current_epoch < start_epoch:
                shared_que.put({
                    'eval_rewards': None,
                    'task_name': task_name
                })
                continue
            if current_epoch > epochs:
                break
            pf_state.load_state_dict(shared_pf_state.state_dict())
            pf_task.load_state_dict(shared_pf_task.state_dict())
            pf_action.load_state_dict(shared_pf_action.state_dict())
            pf_state.eval()
            pf_task.eval()
            pf_action.eval()

            eval_rews = []  

            done = False
          
            for idx in range(env_info.eval_episodes):

                eval_ob = env_info.env.reset()
                rew = 0

                task_idx = env_info.task_rank
                t = 0
                while not done:
                    t += 1
                    task_input = torch.zeros(env_info.num_tasks)
                    task_input[env_info.task_rank] = 1
                    task_input = task_input.unsqueeze(0).to(env_info.device)
                    eval_ob =  torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0)
                    representation = pf_state.forward(eval_ob)
                    embedding = pf_task.forward(task_input)
                    act = pf_action.eval_act(representation, embedding)
                    eval_ob, rs, done, info = env_info.env.step( act )
                    r=rs[task_idx]
                    rew += r
                    if env_info.eval_render:
                        env_info.env.render()
                    done = done or (t >= env_info.max_episode_frames)

                eval_rews.append(rew)
                done = False
                # print(t)
            
            shared_que.put({
                'eval_rewards': eval_rews,
             
                'task_name': task_name
            })

    def start_worker(self):
        self.workers = []
        self.shared_que = self.manager.Queue(self.worker_nums)
        self.start_barrier = mp.Barrier(self.worker_nums)
                
        self.eval_workers = []
        self.eval_shared_que = self.manager.Queue(self.eval_worker_nums)
        self.eval_start_barrier = mp.Barrier(self.eval_worker_nums)


        self.shared_dict = self.manager.dict()

        print(self.worker_nums)
        print(len(self.task_list))
        assert self.worker_nums == len(self.task_list)

        self.env_info.env = self.env
       
        
        self.env_info.num_tasks = len(self.task_list)
       
        single_mt_env_args = {
            "task_name": None,
            "task_rank": 0,
            "num_tasks": len(self.task_list),
            "max_obs_dim": np.prod(self.env.observation_space.shape),
        }
        
      
        tasks=self.task_list
        for i, task in enumerate(tasks):
          
            
            self.env_info.task_rank = i
            
            self.env_info.env_args = single_mt_env_args
            self.env_info.env_args["task_name"] = task

            
            start_epoch = 0
            self.env_info.env_args["task_rank"] = i
            p = mp.Process(
                target=self.__class__.train_worker_process,
                args=( self.__class__, self.shared_funcs,
                    self.env_info, self.replay_buffer, 
                    self.shared_que, self.start_barrier,
                    self.train_epochs, start_epoch, task, self.shared_dict))
            p.start()
            self.workers.append(p)
            # i += 1



        assert self.eval_worker_nums == len(self.task_list)
      
        self.env_info.env = self.env
        self.env_info.num_tasks = len(self.task_list)
       
        single_mt_env_args = {
            "task_name": None,
            "task_rank": 0,
            "num_tasks": len(self.task_list),
            "max_obs_dim": np.prod(self.env.observation_space.shape),
        }

        for i, task in enumerate(tasks):
           

            self.env_info.task_rank = i

            self.env_info.env_args = single_mt_env_args
            self.env_info.env_args["task_name"] = task

            start_epoch = 0
  

            self.env_info.env_args["task_rank"] = i
            eval_p = mp.Process(
                target=self.__class__.eval_worker_process,
                args=(self.shared_funcs["pf_state"],self.shared_funcs["pf_task"],self.shared_funcs["pf_action"],
                    self.env_info, self.eval_shared_que, self.eval_start_barrier,
                    self.eval_epochs, start_epoch, task, self.shared_dict))
            eval_p.start()
            self.eval_workers.append(eval_p)


    def eval_one_epoch(self):
        
        eval_rews = []
      
        self.shared_funcs["pf_state"].load_state_dict(self.funcs["pf_state"].state_dict())
        self.shared_funcs["pf_task"].load_state_dict(self.funcs["pf_task"].state_dict())
        self.shared_funcs["pf_action"].load_state_dict(self.funcs["pf_action"].state_dict())
        tasks_result = []

        active_task_counts = 0
        for _ in range(self.eval_worker_nums):
            worker_rst = self.eval_shared_que.get()
            if worker_rst["eval_rewards"] is not None:
                active_task_counts += 1
                eval_rews += worker_rst["eval_rewards"]
                tasks_result.append((worker_rst["task_name"], 
                np.mean(worker_rst["eval_rewards"])))

        tasks_result.sort()

        dic = OrderedDict()
        for task_name, eval_rewards in tasks_result:
            
            dic[task_name+"_eval_rewards"] = eval_rewards
            self.tasks_progress[self.tasks_mapping[task_name]] *= \
                (1 - self.progress_alpha)

        dic['eval_rewards']      = eval_rews
        

        return dic


    def train_one_epoch(self):
        train_rews = []
        train_epoch_reward = 0

        for key in self.shared_funcs:
            self.shared_funcs[key].load_state_dict(self.funcs[key].state_dict())
        
        active_worker_nums = 0
        for _ in range(self.worker_nums):
            worker_rst = self.shared_que.get()
            
            if worker_rst["train_rewards"] is not None:
                
                train_rews += worker_rst["train_rewards"]
                train_epoch_reward += worker_rst["train_epoch_reward"]
                active_worker_nums += 1
        self.active_worker_nums = active_worker_nums
     
        return {
            'train_rewards':train_rews,
            'train_epoch_reward':train_epoch_reward
        }
    def to(self, device):
        for func in self.funcs:
            self.funcs[func].to(device)
    @property
    def funcs(self):
        return {
            "pf_state": self.pf[0],
            "pf_task": self.pf[1],
            "pf_action": self.pf[2]
        }
    



# class AsyncMultiTaskParallelCollectorForActionRepresentation_Embed(AsyncSingleTaskParallelCollector):

#     def __init__(self, embedding, progress_alpha=0.1, **kwargs):
#         self.embedding = embedding
#         self.share_embedding = copy.copy(self.embedding)
#         super().__init__(**kwargs)
#         self.tasks=self.task_list
#         self.tasks_mapping = {}
#         for idx, task_name in enumerate(self.tasks):
#             self.tasks_mapping[task_name] = idx
#         self.tasks_progress = [0 for _ in range(len(self.tasks))]
#         self.progress_alpha = progress_alpha
#         self.pf_state=self.pf[0]
#         self.pf_action=self.pf[1]

#     @classmethod
#     def take_actions(cls, funcs, embedding, env_info, ob_info, replay_buffer):

#         pf_state = funcs["pf_state"]
#         pf_action = funcs["pf_action"]
#         ob = ob_info["ob"]
#         task_idx = env_info.task_rank
        

#         pf_state.eval()
#         pf_action.eval()

#         with torch.no_grad():
        
#             idx_input = torch.Tensor([[task_idx]]).to(env_info.device).long()
#             task_input = torch.zeros(env_info.num_tasks)
#             task_input[env_info.task_rank] = 1
#             task_input = task_input.to(env_info.device).unsqueeze(0)
#             ob = torch.Tensor( ob ).to(env_info.device).unsqueeze(0)
#             representation = pf_state.forward(ob)
#             out = pf_action.explore(representation, embedding)
#             act = out["action"]
#             # act = act[0]
            


#         act = act.detach().cpu().numpy()
#         if not env_info.continuous:
#             act = act[0]
        
#         if type(act) is not int:
#             if np.isnan(act).any():
#                 print("NaN detected. BOOM")
#                 exit()

#         next_ob, rewards, done, info = env_info.env.step(act)
#         reward=rewards[task_idx]
#         if env_info.train_render:
#             env_info.env.render()
#         env_info.current_step += 1

#         sample_dict = {
#             "obs": ob,
#             "next_obs": next_ob,
#             "acts": act,
#             "task_idxs": [env_info.task_rank],
#             "rewards": [reward],
#             "terminals": [done],
#             "task_inputs": task_input.cpu().numpy(),
#             "embeddings": embedding.detach().cpu().numpy()
#         }
       

#         if done or env_info.current_step >= env_info.max_episode_frames:
#             next_ob = env_info.env.reset()
#             env_info.finish_episode()
#             env_info.start_episode() # reset current_step

#         replay_buffer.add_sample( sample_dict, env_info.task_rank)

#         return next_ob, done, reward, info

#     @staticmethod
#     def train_worker_process(cls, shared_funcs, share_embedding, env_info,
#         replay_buffer, shared_que,
#         start_barrier, epochs, start_epoch, task_name, shared_dict):

#         replay_buffer.rebuild_from_tag()
#         local_funcs = copy.deepcopy(shared_funcs)
#         local_embedding = copy.deepcopy(share_embedding)
    
        
#         for key in local_funcs:
#             local_funcs[key].to(env_info.device)
#         local_embedding.to(env_info.device)
#         c_ob = {
#             "ob": env_info.env.reset()
#         }
#         train_rew = 0
#         current_epoch = 0
#         while True:
#             start_barrier.wait()
#             current_epoch += 1
#             if current_epoch < start_epoch:
#                 shared_que.put({
#                     'train_rewards': None,
#                     'train_epoch_reward': None
#                 })
#                 continue
#             if current_epoch > epochs:
#                 break

#             for key in shared_funcs:
#                 local_funcs[key].load_state_dict(shared_funcs[key].state_dict())
#             local_embedding = share_embedding
#             local_embedding = local_embedding.unsqueeze(0).to("cpu")
#             train_rews = []
#             train_epoch_reward = 0    

#             for t in range(env_info.epoch_frames):

#                 next_ob, done, reward, info = cls.take_actions(local_funcs, local_embedding, env_info, c_ob, replay_buffer )
#                 c_ob["ob"] = next_ob
#                 train_rew += reward
#                 train_epoch_reward += reward
#                 train_rews.append(train_rew)
#                 train_rew = 0
#                 # if done:
#                 #     train_rews.append(train_rew)
#                 #     train_rew = 0
            
#             shared_que.put({
#                 'train_rewards':train_rews,
#                 'train_epoch_reward':train_epoch_reward
#             })

#     @staticmethod
#     def eval_worker_process(shared_pf_state,shared_pf_action, shared_embedding,
#         env_info, shared_que, start_barrier, epochs, start_epoch, task_name, shared_dict):

#         pf_state = copy.deepcopy(shared_pf_state).to(env_info.device)
#         pf_action = copy.deepcopy(shared_pf_action).to(env_info.device)
       
#         current_epoch = 0
#         while True:
#             start_barrier.wait()
#             current_epoch += 1
#             if current_epoch < start_epoch:
#                 shared_que.put({
#                     'eval_rewards': None,
#                     'task_name': task_name
#                 })
#                 continue
#             if current_epoch > epochs:
#                 break
#             pf_state.load_state_dict(shared_pf_state.state_dict())
#             pf_action.load_state_dict(shared_pf_action.state_dict())
#             pf_state.eval()
#             pf_action.eval()
#             embedding = copy.deepcopy(shared_embedding).to(env_info.device)
#             embedding = embedding.unsqueeze(0).to("cpu")
#             eval_rews = []  
  
#             done = False
          
#             for idx in range(env_info.eval_episodes):

#                 eval_ob = env_info.env.reset()
#                 rew = 0

#                 task_idx = env_info.task_rank
                
#                 while not done:
#                     eval_ob =  torch.Tensor(eval_ob).to(env_info.device).unsqueeze(0)
#                     representation = pf_state.forward(eval_ob)
#                     act = pf_action.eval_act(representation, embedding)
#                     eval_ob, rs, done, info = env_info.env.step( act )
#                     r=rs[task_idx]
#                     rew += r
#                     if env_info.eval_render:
#                         env_info.env.render()
#                 # print(info['x_velocity'])
#                 # print(info['reward_ctrl'])

#                 eval_rews.append(rew)
#                 done = False
      
#             shared_que.put({
#                 'eval_rewards': eval_rews,
#                 'embedding': shared_embedding.detach().cpu(),
#                 'task_name': task_name
#             })

#     def start_worker(self):
#         self.workers = []
#         self.shared_que = self.manager.Queue(self.worker_nums)
#         self.start_barrier = mp.Barrier(self.worker_nums)
                
#         self.eval_workers = []
#         self.eval_shared_que = self.manager.Queue(self.eval_worker_nums)
#         self.eval_start_barrier = mp.Barrier(self.eval_worker_nums)


#         self.shared_dict = self.manager.dict()

        

#         assert self.worker_nums == len(self.task_list)

#         self.env_info.env = self.env
       
        
#         self.env_info.num_tasks = len(self.task_list)
       
#         single_mt_env_args = {
#             "task_name": None,
#             "task_rank": 0,
#             "num_tasks": len(self.task_list),
#             "max_obs_dim": np.prod(self.env.observation_space.shape),
#         }
        
      
#         tasks=self.task_list
#         for i, task in enumerate(tasks):
          
            
#             self.env_info.task_rank = i
#             self.env_info.env_args = single_mt_env_args
#             self.env_info.env_args["task_name"] = task
            
            
#             start_epoch = 0
#             self.env_info.env_args["task_rank"] = i
#             p = mp.Process(
#                 target=self.__class__.train_worker_process,
#                 args=( self.__class__, self.shared_funcs, self.share_embedding,
#                     self.env_info, self.replay_buffer, 
#                     self.shared_que, self.start_barrier,
#                     self.train_epochs, start_epoch, task, self.shared_dict))
#             p.start()
#             self.workers.append(p)




#         assert self.eval_worker_nums == len(self.task_list)
      
#         self.env_info.env = self.env
#         self.env_info.num_tasks = len(self.task_list)
       
#         single_mt_env_args = {
#             "task_name": None,
#             "task_rank": 0,
#             "num_tasks": len(self.task_list),
#             "max_obs_dim": np.prod(self.env.observation_space.shape),
#         }

#         for i, task in enumerate(tasks):
           

#             self.env_info.task_rank = i

#             self.env_info.env_args = single_mt_env_args
#             self.env_info.env_args["task_name"] = task

#             start_epoch = 0
  

#             self.env_info.env_args["task_rank"] = i
#             eval_p = mp.Process(
#                 target=self.__class__.eval_worker_process,
#                 args=(self.shared_funcs["pf_state"],self.shared_funcs["pf_action"],self.share_embedding,
#                     self.env_info, self.eval_shared_que, self.eval_start_barrier,
#                     self.eval_epochs, start_epoch, task, self.shared_dict))
#             eval_p.start()
#             self.eval_workers.append(eval_p)


#     def eval_one_epoch(self):
        
#         eval_rews = []
      
#         self.shared_funcs["pf_state"].load_state_dict(self.funcs["pf_state"].state_dict())
#         self.shared_funcs["pf_action"].load_state_dict(self.funcs["pf_action"].state_dict())
#         tasks_result = []
        
#         embed_csv_path = "log/embedding_collection.csv"
#         embed_file = open(embed_csv_path, "a")
#         embed_writer = csv.writer(embed_file)
        
        
#         active_task_counts = 0
#         for _ in range(self.eval_worker_nums):
#             worker_rst = self.eval_shared_que.get()
#             embedding = worker_rst['embedding'].detach().cpu().numpy()
#             embed_writer.writerow(embedding)
            
#             # print(worker_rst['embedding'])
#             if worker_rst["eval_rewards"] is not None:
#                 active_task_counts += 1
#                 eval_rews += worker_rst["eval_rewards"]
#                 tasks_result.append((worker_rst["task_name"], 
#                 np.mean(worker_rst["eval_rewards"])))
#         embed_file.close()
#         tasks_result.sort()

#         dic = OrderedDict()
#         for task_name, eval_rewards in tasks_result:
            
#             dic[task_name+"_eval_rewards"] = eval_rewards
#             self.tasks_progress[self.tasks_mapping[task_name]] *= \
#                 (1 - self.progress_alpha)

#         dic['eval_rewards']      = eval_rews
        

#         return dic


#     def train_one_epoch(self):
#         train_rews = []
#         train_epoch_reward = 0

#         for key in self.shared_funcs:
#             self.shared_funcs[key].load_state_dict(self.funcs[key].state_dict())
#         # self.share_embedding = self.embedding
#         active_worker_nums = 0
#         for _ in range(self.worker_nums):
#             worker_rst = self.shared_que.get()
            
#             if worker_rst["train_rewards"] is not None:
                
#                 train_rews += worker_rst["train_rewards"]
#                 train_epoch_reward += worker_rst["train_epoch_reward"]
#                 active_worker_nums += 1
#         self.active_worker_nums = active_worker_nums
     
#         return {
#             'train_rewards':train_rews,
#             'train_epoch_reward':train_epoch_reward
#         }


#     def to(self, device):
#         for func in self.funcs:
#             self.funcs[func].to(device)
            
#     @property
#     def funcs(self):
#         return {
#             "pf_state": self.pf[0],
#             "pf_action": self.pf[1]
#         }

















# class AsyncMultiTaskParallelCollectorForActionRepresentation_Embed_withNorm(AsyncSingleTaskParallelCollector):

#     def __init__(self, param, progress_alpha=0.1, **kwargs):
#         self.param = param
#         self.share_param = copy.copy(self.param)
#         super().__init__(**kwargs)
#         self.tasks=self.task_list
#         self.tasks_mapping = {}
#         for idx, task_name in enumerate(self.tasks):
#             self.tasks_mapping[task_name] = idx
#         self.tasks_progress = [0 for _ in range(len(self.tasks))]
#         self.progress_alpha = progress_alpha
#         self.pf_state=self.pf[0]
#         self.pf_action=self.pf[1]

#     @classmethod
#     def take_actions(cls, funcs, param, env_info, ob_info, replay_buffer):

#         pf_state = funcs["pf_state"]
#         pf_action = funcs["pf_action"]
#         ob = ob_info["ob"]
#         task_idx = env_info.task_rank
        

#         pf_state.eval()
#         pf_action.eval()

#         with torch.no_grad():
        
#             idx_input = torch.Tensor([[task_idx]]).to(env_info.device).long()
#             task_input = torch.zeros(env_info.num_tasks)
#             task_input[env_info.task_rank] = 1
#             task_input = task_input.to(env_info.device).unsqueeze(0)
#             ob = torch.Tensor( ob ).to(env_info.device).unsqueeze(0)
#             representation = pf_state.forward(ob)
#             embedding_x = 5 * torch.sin(param[0]) * torch.cos(param[1]).unsqueeze(0)
#             embedding_y = 5 * torch.sin(param[0]) * torch.sin(param[1]).unsqueeze(0)
#             embedding_z = 5 * torch.cos(param[0]).unsqueeze(0)
#             embedding = torch.cat((embedding_x, embedding_y, embedding_z), dim=-1)
#             embedding = embedding.unsqueeze(0).to("cpu")
#             out = pf_action.explore(representation, embedding)
#             act = out["action"]
#             # act = act[0]
            


#         act = act.detach().cpu().numpy()
#         if not env_info.continuous:
#             act = act[0]
        
#         if type(act) is not int:
#             if np.isnan(act).any():
#                 print("NaN detected. BOOM")
#                 exit()

#         next_ob, rewards, done, info = env_info.env.step(act)
#         reward=rewards[task_idx]
#         if env_info.train_render:
#             env_info.env.render()
#         env_info.current_step += 1

#         sample_dict = {
#             "obs": ob,
#             "next_obs": next_ob,
#             "acts": act,
#             "task_idxs": [env_info.task_rank],
#             "rewards": [reward],
#             "terminals": [done],
#             "task_inputs": task_input.cpu().numpy(),
#             "params": param.detach().cpu().numpy()
#         }
       

#         if done or env_info.current_step >= env_info.max_episode_frames:
#             next_ob = env_info.env.reset()
#             env_info.finish_episode()
#             env_info.start_episode() # reset current_step

#         replay_buffer.add_sample( sample_dict, env_info.task_rank)

#         return next_ob, done, reward, info

#     @staticmethod
#     def train_worker_process(cls, shared_funcs, share_param, env_info,
#         replay_buffer, shared_que,
#         start_barrier, epochs, start_epoch, task_name, shared_dict):

#         replay_buffer.rebuild_from_tag()
#         local_funcs = copy.deepcopy(shared_funcs)
#         local_param = copy.deepcopy(share_param)
    
        
#         for key in local_funcs:
#             local_funcs[key].to(env_info.device)
#         local_param.to(env_info.device)
#         c_ob = {
#             "ob": env_info.env.reset()
#         }
#         train_rew = 0
#         current_epoch = 0
#         while True:
#             start_barrier.wait()
#             current_epoch += 1
#             if current_epoch < start_epoch:
#                 shared_que.put({
#                     'train_rewards': None,
#                     'train_epoch_reward': None
#                 })
#                 continue
#             if current_epoch > epochs:
#                 break

#             for key in shared_funcs:
#                 local_funcs[key].load_state_dict(shared_funcs[key].state_dict())
#             local_param = share_param
#             # local_embedding = local_embedding.unsqueeze(0).to("cpu")
#             train_rews = []
#             train_epoch_reward = 0    

#             for t in range(env_info.epoch_frames):

#                 next_ob, done, reward, info = cls.take_actions(local_funcs, local_param, env_info, c_ob, replay_buffer )
#                 c_ob["ob"] = next_ob
#                 train_rew += reward
#                 train_epoch_reward += reward
#                 train_rews.append(train_rew)
#                 train_rew = 0
#                 # if done:
#                 #     train_rews.append(train_rew)
#                 #     train_rew = 0
            
#             shared_que.put({
#                 'train_rewards':train_rews,
#                 'train_epoch_reward':train_epoch_reward
#             })

#     @staticmethod
#     def eval_worker_process(shared_pf_state,shared_pf_action, shared_param,
#         env_info, shared_que, start_barrier, epochs, start_epoch, task_name, shared_dict):

#         pf_state = copy.deepcopy(shared_pf_state).to(env_info.device)
#         pf_action = copy.deepcopy(shared_pf_action).to(env_info.device)
       
#         current_epoch = 0
#         while True:
#             start_barrier.wait()
#             current_epoch += 1
#             if current_epoch < start_epoch:
#                 shared_que.put({
#                     'eval_rewards': None,
#                     'task_name': task_name
#                 })
#                 continue
#             if current_epoch > epochs:
#                 break
#             pf_state.load_state_dict(shared_pf_state.state_dict())
#             pf_action.load_state_dict(shared_pf_action.state_dict())
#             pf_state.eval()
#             pf_action.eval()
#             param = copy.deepcopy(shared_param).to(env_info.device)
#             embedding_x = 5 * torch.sin(param[0]) * torch.cos(param[1]).unsqueeze(0)
#             embedding_y = 5 * torch.sin(param[0]) * torch.sin(param[1]).unsqueeze(0)
#             embedding_z = 5 * torch.cos(param[0]).unsqueeze(0)
#             # print(embedding_x)
#             # print(embedding_x.size())
#             embedding = torch.cat((embedding_x, embedding_y, embedding_z), dim=0)
            
            
#             embedding = embedding.unsqueeze(0).to("cpu")
#             eval_rews = []  
  
#             done = False
          
#             for idx in range(env_info.eval_episodes):

#                 eval_ob = env_info.env.reset()
#                 rew = 0

#                 task_idx = env_info.task_rank
                
#                 while not done:
#                     eval_ob =  torch.Tensor(eval_ob).to(env_info.device).unsqueeze(0)
#                     representation = pf_state.forward(eval_ob)
#                     act = pf_action.eval_act(representation, embedding)
#                     eval_ob, rs, done, info = env_info.env.step( act )
#                     r=rs[task_idx]
#                     rew += r
#                     if env_info.eval_render:
#                         env_info.env.render()
#                 # print(info['x_velocity'])
#                 # print(info['reward_ctrl'])

#                 eval_rews.append(rew)
#                 done = False
      
#             shared_que.put({
#                 'eval_rewards': eval_rews,
#                 'embedding': embedding.detach().cpu(),
#                 'task_name': task_name
#             })

#     def start_worker(self):
#         self.workers = []
#         self.shared_que = self.manager.Queue(self.worker_nums)
#         self.start_barrier = mp.Barrier(self.worker_nums)
                
#         self.eval_workers = []
#         self.eval_shared_que = self.manager.Queue(self.eval_worker_nums)
#         self.eval_start_barrier = mp.Barrier(self.eval_worker_nums)


#         self.shared_dict = self.manager.dict()

        

#         assert self.worker_nums == len(self.task_list)

#         self.env_info.env = self.env
       
        
#         self.env_info.num_tasks = len(self.task_list)
       
#         single_mt_env_args = {
#             "task_name": None,
#             "task_rank": 0,
#             "num_tasks": len(self.task_list),
#             "max_obs_dim": np.prod(self.env.observation_space.shape),
#         }
        
      
#         tasks=self.task_list
#         for i, task in enumerate(tasks):
          
            
#             self.env_info.task_rank = i
#             self.env_info.env_args = single_mt_env_args
#             self.env_info.env_args["task_name"] = task
            
            
#             start_epoch = 0
#             self.env_info.env_args["task_rank"] = i
#             p = mp.Process(
#                 target=self.__class__.train_worker_process,
#                 args=( self.__class__, self.shared_funcs, self.share_param,
#                     self.env_info, self.replay_buffer, 
#                     self.shared_que, self.start_barrier,
#                     self.train_epochs, start_epoch, task, self.shared_dict))
#             p.start()
#             self.workers.append(p)




#         assert self.eval_worker_nums == len(self.task_list)
      
#         self.env_info.env = self.env
#         self.env_info.num_tasks = len(self.task_list)
       
#         single_mt_env_args = {
#             "task_name": None,
#             "task_rank": 0,
#             "num_tasks": len(self.task_list),
#             "max_obs_dim": np.prod(self.env.observation_space.shape),
#         }

#         for i, task in enumerate(tasks):
           

#             self.env_info.task_rank = i

#             self.env_info.env_args = single_mt_env_args
#             self.env_info.env_args["task_name"] = task

#             start_epoch = 0
  

#             self.env_info.env_args["task_rank"] = i
#             eval_p = mp.Process(
#                 target=self.__class__.eval_worker_process,
#                 args=(self.shared_funcs["pf_state"],self.shared_funcs["pf_action"],self.share_param,
#                     self.env_info, self.eval_shared_que, self.eval_start_barrier,
#                     self.eval_epochs, start_epoch, task, self.shared_dict))
#             eval_p.start()
#             self.eval_workers.append(eval_p)


#     def eval_one_epoch(self):
        
#         eval_rews = []
      
#         self.shared_funcs["pf_state"].load_state_dict(self.funcs["pf_state"].state_dict())
#         self.shared_funcs["pf_action"].load_state_dict(self.funcs["pf_action"].state_dict())
#         tasks_result = []
        
#         embed_csv_path = "log/embedding_collection.csv"
#         embed_file = open(embed_csv_path, "a")
#         embed_writer = csv.writer(embed_file)
        
        
#         active_task_counts = 0
#         for _ in range(self.eval_worker_nums):
#             worker_rst = self.eval_shared_que.get()
#             embedding = worker_rst['embedding'].detach().cpu().numpy()
#             embed_writer.writerow(embedding)
            
#             print(worker_rst['embedding'])
#             if worker_rst["eval_rewards"] is not None:
#                 active_task_counts += 1
#                 eval_rews += worker_rst["eval_rewards"]
#                 tasks_result.append((worker_rst["task_name"], 
#                 np.mean(worker_rst["eval_rewards"])))
#         embed_file.close()
#         tasks_result.sort()

#         dic = OrderedDict()
#         for task_name, eval_rewards in tasks_result:
            
#             dic[task_name+"_eval_rewards"] = eval_rewards
#             self.tasks_progress[self.tasks_mapping[task_name]] *= \
#                 (1 - self.progress_alpha)

#         dic['eval_rewards']      = eval_rews
        

#         return dic


#     def train_one_epoch(self):
#         train_rews = []
#         train_epoch_reward = 0

#         for key in self.shared_funcs:
#             self.shared_funcs[key].load_state_dict(self.funcs[key].state_dict())
        
#         active_worker_nums = 0
#         for _ in range(self.worker_nums):
#             worker_rst = self.shared_que.get()
            
#             if worker_rst["train_rewards"] is not None:
                
#                 train_rews += worker_rst["train_rewards"]
#                 train_epoch_reward += worker_rst["train_epoch_reward"]
#                 active_worker_nums += 1
#         self.active_worker_nums = active_worker_nums
     
#         return {
#             'train_rewards':train_rews,
#             'train_epoch_reward':train_epoch_reward
#         }


#     def to(self, device):
#         for func in self.funcs:
#             self.funcs[func].to(device)
            
#     @property
#     def funcs(self):
#         return {
#             "pf_state": self.pf[0],
#             "pf_action": self.pf[1]
#         }
















# class AsyncMultiTaskParallelCollectorForActionRepresentation_Embed_withNorm_v4(AsyncSingleTaskParallelCollector):

#     def __init__(self, embedding, embedding4q, progress_alpha=0.1, **kwargs):
#         self.embedding = embedding
#         self.share_embedding = copy.copy(self.embedding)
#         self.embedding4q = embedding4q
#         self.share_embedding4q = copy.copy(self.embedding4q)
#         super().__init__(**kwargs)
#         self.tasks=self.task_list
#         self.tasks_mapping = {}
#         for idx, task_name in enumerate(self.tasks):
#             self.tasks_mapping[task_name] = idx
#         self.tasks_progress = [0 for _ in range(len(self.tasks))]
#         self.progress_alpha = progress_alpha
#         self.pf_state=self.pf[0]
#         self.pf_action=self.pf[1]

#     @classmethod
#     def take_actions(cls, funcs, embedding, embedding4q, env_info, ob_info, replay_buffer):

#         pf_state = funcs["pf_state"]
#         pf_action = funcs["pf_action"]
#         ob = ob_info["ob"]
#         task_idx = env_info.task_rank
        

#         pf_state.eval()
#         pf_action.eval()

#         with torch.no_grad():
        
#             idx_input = torch.Tensor([[task_idx]]).to(env_info.device).long()
#             task_input = torch.zeros(env_info.num_tasks)
#             task_input[env_info.task_rank] = 1
#             task_input = task_input.to(env_info.device).unsqueeze(0)
#             ob = torch.Tensor( ob ).to(env_info.device).unsqueeze(0)
#             representation = pf_state.forward(ob)
#             embedding = 5 * F.normalize(embedding.unsqueeze(0)).squeeze(0)
#             out = pf_action.explore(representation, embedding)
#             act = out["action"]
#             embedding4q = 5 * F.normalize(embedding4q.unsqueeze(0)).squeeze(0)
#             # act = act[0]
            


#         act = act.detach().cpu().numpy()
#         if not env_info.continuous:
#             act = act[0]
        
#         if type(act) is not int:
#             if np.isnan(act).any():
#                 print("NaN detected. BOOM")
#                 exit()

#         next_ob, rewards, done, info = env_info.env.step(act)
#         reward=rewards[task_idx]
#         if env_info.train_render:
#             env_info.env.render()
#         env_info.current_step += 1

#         sample_dict = {
#             "obs": ob,
#             "next_obs": next_ob,
#             "acts": act,
#             "task_idxs": [env_info.task_rank],
#             "rewards": [reward],
#             "terminals": [done],
#             "task_inputs": task_input.cpu().numpy(),
#             "embeddings": embedding.detach().cpu().numpy(),
#             "embeddings4q": embedding4q.detach().cpu().numpy()
#         }
       

#         if done or env_info.current_step >= env_info.max_episode_frames:
#             next_ob = env_info.env.reset()
#             env_info.finish_episode()
#             env_info.start_episode() # reset current_step

#         replay_buffer.add_sample( sample_dict, env_info.task_rank)

#         return next_ob, done, reward, info

#     @staticmethod
#     def train_worker_process(cls, shared_funcs, share_embedding, share_embedding4q, env_info,
#         replay_buffer, shared_que,
#         start_barrier, epochs, start_epoch, task_name, shared_dict):

#         replay_buffer.rebuild_from_tag()
#         local_funcs = copy.deepcopy(shared_funcs)
#         local_embedding = copy.deepcopy(share_embedding)
#         local_embedding4q = copy.deepcopy(share_embedding4q)
    
        
#         for key in local_funcs:
#             local_funcs[key].to(env_info.device)
#         local_embedding.to(env_info.device)
#         local_embedding4q.to(env_info.device)
#         c_ob = {
#             "ob": env_info.env.reset()
#         }
#         train_rew = 0
#         current_epoch = 0
#         while True:
#             start_barrier.wait()
#             current_epoch += 1
#             if current_epoch < start_epoch:
#                 shared_que.put({
#                     'train_rewards': None,
#                     'train_epoch_reward': None
#                 })
#                 continue
#             if current_epoch > epochs:
#                 break

#             for key in shared_funcs:
#                 local_funcs[key].load_state_dict(shared_funcs[key].state_dict())
#             local_embedding = share_embedding
#             local_embedding = local_embedding.unsqueeze(0).to("cpu")
#             local_embedding4q = share_embedding4q
#             local_embedding4q = local_embedding4q.unsqueeze(0).to("cpu")
#             train_rews = []
#             train_epoch_reward = 0    

#             for t in range(env_info.epoch_frames):

#                 next_ob, done, reward, info = cls.take_actions(local_funcs, local_embedding, local_embedding4q, env_info, c_ob, replay_buffer )
#                 c_ob["ob"] = next_ob
#                 train_rew += reward
#                 train_epoch_reward += reward
#                 train_rews.append(train_rew)
#                 train_rew = 0
#                 # if done:
#                 #     train_rews.append(train_rew)
#                 #     train_rew = 0
            
#             shared_que.put({
#                 'train_rewards':train_rews,
#                 'train_epoch_reward':train_epoch_reward
#             })

#     @staticmethod
#     def eval_worker_process(shared_pf_state,shared_pf_action, shared_embedding, shared_embedding4q,
#         env_info, shared_que, start_barrier, epochs, start_epoch, task_name, shared_dict):

#         pf_state = copy.deepcopy(shared_pf_state).to(env_info.device)
#         pf_action = copy.deepcopy(shared_pf_action).to(env_info.device)
       
#         current_epoch = 0
#         while True:
#             start_barrier.wait()
#             current_epoch += 1
#             if current_epoch < start_epoch:
#                 shared_que.put({
#                     'eval_rewards': None,
#                     'task_name': task_name
#                 })
#                 continue
#             if current_epoch > epochs:
#                 break
#             pf_state.load_state_dict(shared_pf_state.state_dict())
#             pf_action.load_state_dict(shared_pf_action.state_dict())
#             pf_state.eval()
#             pf_action.eval()
#             embedding = copy.deepcopy(shared_embedding).to(env_info.device)
#             embedding = embedding.unsqueeze(0).to("cpu")
#             embedding4q = copy.deepcopy(shared_embedding4q).to(env_info.device)
#             embedding4q = embedding4q.unsqueeze(0).to("cpu")
#             eval_rews = []  
  
#             done = False
          
#             for idx in range(env_info.eval_episodes):

#                 eval_ob = env_info.env.reset()
#                 rew = 0

#                 task_idx = env_info.task_rank
                
#                 while not done:
#                     eval_ob =  torch.Tensor(eval_ob).to(env_info.device).unsqueeze(0)
#                     representation = pf_state.forward(eval_ob)
#                     embedding = 5 * F.normalize(embedding)
#                     act = pf_action.eval_act(representation, embedding)
#                     eval_ob, rs, done, info = env_info.env.step( act )
#                     r=rs[task_idx]
#                     rew += r
#                     if env_info.eval_render:
#                         env_info.env.render()
#                 # print(info['x_velocity'])
#                 # print(info['reward_ctrl'])

#                 eval_rews.append(rew)
#                 done = False
      
#             shared_que.put({
#                 'eval_rewards': eval_rews,
#                 'embedding': 5 * F.normalize(shared_embedding.unsqueeze(0)).squeeze(0).detach().cpu(),
#                 'embedding4q': 5 * F.normalize(shared_embedding4q.unsqueeze(0)).squeeze(0).detach().cpu(),
#                 'task_name': task_name
#             })

#     def start_worker(self):
#         self.workers = []
#         self.shared_que = self.manager.Queue(self.worker_nums)
#         self.start_barrier = mp.Barrier(self.worker_nums)
                
#         self.eval_workers = []
#         self.eval_shared_que = self.manager.Queue(self.eval_worker_nums)
#         self.eval_start_barrier = mp.Barrier(self.eval_worker_nums)


#         self.shared_dict = self.manager.dict()

        

#         assert self.worker_nums == len(self.task_list)

#         self.env_info.env = self.env
       
        
#         self.env_info.num_tasks = len(self.task_list)
       
#         single_mt_env_args = {
#             "task_name": None,
#             "task_rank": 0,
#             "num_tasks": len(self.task_list),
#             "max_obs_dim": np.prod(self.env.observation_space.shape),
#         }
        
      
#         tasks=self.task_list
#         for i, task in enumerate(tasks):
          
            
#             self.env_info.task_rank = i
#             self.env_info.env_args = single_mt_env_args
#             self.env_info.env_args["task_name"] = task
            
            
#             start_epoch = 0
#             self.env_info.env_args["task_rank"] = i
#             p = mp.Process(
#                 target=self.__class__.train_worker_process,
#                 args=( self.__class__, self.shared_funcs, self.share_embedding, self.share_embedding4q,
#                     self.env_info, self.replay_buffer, 
#                     self.shared_que, self.start_barrier,
#                     self.train_epochs, start_epoch, task, self.shared_dict))
#             p.start()
#             self.workers.append(p)




#         assert self.eval_worker_nums == len(self.task_list)
      
#         self.env_info.env = self.env
#         self.env_info.num_tasks = len(self.task_list)
       
#         single_mt_env_args = {
#             "task_name": None,
#             "task_rank": 0,
#             "num_tasks": len(self.task_list),
#             "max_obs_dim": np.prod(self.env.observation_space.shape),
#         }

#         for i, task in enumerate(tasks):
           

#             self.env_info.task_rank = i

#             self.env_info.env_args = single_mt_env_args
#             self.env_info.env_args["task_name"] = task

#             start_epoch = 0
  

#             self.env_info.env_args["task_rank"] = i
#             eval_p = mp.Process(
#                 target=self.__class__.eval_worker_process,
#                 args=(self.shared_funcs["pf_state"],self.shared_funcs["pf_action"],self.share_embedding,self.share_embedding4q,
#                     self.env_info, self.eval_shared_que, self.eval_start_barrier,
#                     self.eval_epochs, start_epoch, task, self.shared_dict))
#             eval_p.start()
#             self.eval_workers.append(eval_p)


#     def eval_one_epoch(self):
        
#         eval_rews = []
      
#         self.shared_funcs["pf_state"].load_state_dict(self.funcs["pf_state"].state_dict())
#         self.shared_funcs["pf_action"].load_state_dict(self.funcs["pf_action"].state_dict())
#         tasks_result = []
        
#         embed_csv_path = "log/embedding_collection.csv"
#         embed_file = open(embed_csv_path, "a")
#         embed_writer = csv.writer(embed_file)
#         embed4q_csv_path = "log/embedding4q_collection.csv"
#         embed4q_file = open(embed4q_csv_path, "a")
#         embed4q_writer = csv.writer(embed4q_file)
        
        
#         active_task_counts = 0
#         for _ in range(self.eval_worker_nums):
#             worker_rst = self.eval_shared_que.get()
#             embedding = worker_rst['embedding'].detach().cpu().numpy()
#             embed_writer.writerow(embedding)
#             embedding4q = worker_rst['embedding4q'].detach().cpu().numpy()
#             embed4q_writer.writerow(embedding4q)
            
#             # print(worker_rst['embedding'])
#             if worker_rst["eval_rewards"] is not None:
#                 active_task_counts += 1
#                 eval_rews += worker_rst["eval_rewards"]
#                 tasks_result.append((worker_rst["task_name"], 
#                 np.mean(worker_rst["eval_rewards"])))
#         embed_file.close()
#         embed4q_file.close()
#         tasks_result.sort()

#         dic = OrderedDict()
#         for task_name, eval_rewards in tasks_result:
            
#             dic[task_name+"_eval_rewards"] = eval_rewards
#             self.tasks_progress[self.tasks_mapping[task_name]] *= \
#                 (1 - self.progress_alpha)

#         dic['eval_rewards']      = eval_rews
        

#         return dic


#     def train_one_epoch(self):
#         train_rews = []
#         train_epoch_reward = 0

#         for key in self.shared_funcs:
#             self.shared_funcs[key].load_state_dict(self.funcs[key].state_dict())
#         # self.share_embedding = self.embedding
#         active_worker_nums = 0
#         for _ in range(self.worker_nums):
#             worker_rst = self.shared_que.get()
            
#             if worker_rst["train_rewards"] is not None:
                
#                 train_rews += worker_rst["train_rewards"]
#                 train_epoch_reward += worker_rst["train_epoch_reward"]
#                 active_worker_nums += 1
#         self.active_worker_nums = active_worker_nums
     
#         return {
#             'train_rewards':train_rews,
#             'train_epoch_reward':train_epoch_reward
#         }


#     def to(self, device):
#         for func in self.funcs:
#             self.funcs[func].to(device)
            
#     @property
#     def funcs(self):
#         return {
#             "pf_state": self.pf[0],
#             "pf_action": self.pf[1]
#         }




















# class AsyncMultiTaskParallelCollectorForActionRepresentation_Net(AsyncSingleTaskParallelCollector):

#     def __init__(self, embedding, progress_alpha=0.1, **kwargs):
#         self.embedding = embedding
#         super().__init__(**kwargs)
#         self.tasks=self.task_list
#         self.tasks_mapping = {}
#         for idx, task_name in enumerate(self.tasks):
#             self.tasks_mapping[task_name] = idx
#         self.tasks_progress = [0 for _ in range(len(self.tasks))]
#         self.progress_alpha = progress_alpha
#         self.pf_state=self.pf[0]
#         self.pf_action=self.pf[1]
  

#     @classmethod
#     def take_actions(cls, funcs, embedding, env_info, ob_info, replay_buffer):

#         pf_state = funcs["pf_state"]
#         pf_action = funcs["pf_action"]
#         ob = ob_info["ob"]
#         task_idx = env_info.task_rank
        

#         pf_state.eval()
#         pf_action.eval()

#         with torch.no_grad():
        
#             idx_input = torch.Tensor([[task_idx]]).to(env_info.device).long()
#             task_input = torch.zeros(env_info.num_tasks)
#             task_input[env_info.task_rank] = 1
#             task_input = task_input.to(env_info.device).unsqueeze(0)
#             ob = torch.Tensor( ob ).to(env_info.device).unsqueeze(0)
#             representation = pf_state.forward(ob)
#             out = pf_action.explore(representation, embedding)
#             act = out["action"]
#             # act = act[0]
            


#         act = act.detach().cpu().numpy()
#         if not env_info.continuous:
#             act = act[0]
        
#         if type(act) is not int:
#             if np.isnan(act).any():
#                 print("NaN detected. BOOM")
#                 exit()

#         next_ob, rewards, done, info = env_info.env.step(act)
#         reward=rewards[task_idx]
#         if env_info.train_render:
#             env_info.env.render()
#         env_info.current_step += 1

#         sample_dict = {
#             "obs": ob,
#             "next_obs": next_ob,
#             "acts": act,
#             "task_idxs": [env_info.task_rank],
#             "rewards": [reward],
#             "terminals": [done],
#             "task_inputs": task_input.cpu().numpy(),
#             "embeddings": embedding.detach().cpu().numpy()
#         }
       

#         if done or env_info.current_step >= env_info.max_episode_frames:
#             next_ob = env_info.env.reset()
#             env_info.finish_episode()
#             env_info.start_episode() 

#         replay_buffer.add_sample( sample_dict, env_info.task_rank)

#         return next_ob, done, reward, info

#     @staticmethod
#     def train_worker_process(cls, shared_funcs, embedding, env_info,
#         replay_buffer, shared_que,
#         start_barrier, epochs, start_epoch, task_name, shared_dict):

#         replay_buffer.rebuild_from_tag()
#         local_funcs = copy.deepcopy(shared_funcs)
#         for key in local_funcs:
#             local_funcs[key].to(env_info.device)
#         embedding = embedding.unsqueeze(0).to(env_info.device)
#         c_ob = {
#             "ob": env_info.env.reset()
#         }
#         train_rew = 0
#         current_epoch = 0
#         while True:
#             start_barrier.wait()
#             current_epoch += 1
#             if current_epoch < start_epoch:
#                 shared_que.put({
#                     'train_rewards': None,
#                     'train_epoch_reward': None
#                 })
#                 continue
#             if current_epoch > epochs:
#                 break

#             for key in shared_funcs:
#                 local_funcs[key].load_state_dict(shared_funcs[key].state_dict())
#             train_rews = []
#             train_epoch_reward = 0    
           
#             for t in range(env_info.epoch_frames):

#                 next_ob, done, reward, info = cls.take_actions(local_funcs, embedding, env_info, c_ob, replay_buffer )
#                 c_ob["ob"] = next_ob
#                 train_rew += reward
#                 train_epoch_reward += reward
#                 train_rews.append(train_rew)
#                 train_rew = 0
            
#             shared_que.put({
#                 'train_rewards':train_rews,
#                 'train_epoch_reward':train_epoch_reward
#             })

#     @staticmethod
#     def eval_worker_process(shared_pf_state,shared_pf_action, embedding,
#         env_info, shared_que, start_barrier, epochs, start_epoch, task_name, shared_dict):

#         pf_state = copy.deepcopy(shared_pf_state).to(env_info.device)
#         pf_action = copy.deepcopy(shared_pf_action).to(env_info.device)
#         embedding = embedding.unsqueeze(0).to("cpu")
#         current_epoch = 0
#         while True:
#             start_barrier.wait()
#             current_epoch += 1
#             if current_epoch < start_epoch:
#                 shared_que.put({
#                     'eval_rewards': None,
#                     'task_name': task_name
#                 })
#                 continue
#             if current_epoch > epochs:
#                 break
#             pf_state.load_state_dict(shared_pf_state.state_dict())
#             pf_action.load_state_dict(shared_pf_action.state_dict())
#             pf_state.eval()
#             pf_action.eval()
           
#             eval_rews = []  

#             done = False
          
#             for idx in range(env_info.eval_episodes):

#                 eval_ob = env_info.env.reset()
#                 rew = 0

#                 task_idx = env_info.task_rank
                
#                 while not done:
#                     eval_ob =  torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0)
#                     representation = pf_state.forward(eval_ob)
#                     act = pf_action.eval_act(representation, embedding)
#                     eval_ob, rs, done, info = env_info.env.step( act )
#                     r=rs[task_idx]
#                     rew += r
#                     if env_info.eval_render:
#                         env_info.env.render()
                   

#                 eval_rews.append(rew)
#                 done = False
      
#             shared_que.put({
#                 'eval_rewards': eval_rews,
             
#                 'task_name': task_name
#             })

#     def start_worker(self):
#         self.workers = []
#         self.shared_que = self.manager.Queue(self.worker_nums)
#         self.start_barrier = mp.Barrier(self.worker_nums)
                
#         self.eval_workers = []
#         self.eval_shared_que = self.manager.Queue(self.eval_worker_nums)
#         self.eval_start_barrier = mp.Barrier(self.eval_worker_nums)


#         self.shared_dict = self.manager.dict()

        

#         assert self.worker_nums == len(self.task_list)

#         self.env_info.env = self.env
       
        
#         self.env_info.num_tasks = len(self.task_list)
       
#         single_mt_env_args = {
#             "task_name": None,
#             "task_rank": 0,
#             "num_tasks": len(self.task_list),
#             "max_obs_dim": np.prod(self.env.observation_space.shape),
#         }
        
      
#         tasks=self.task_list
#         for i, task in enumerate(tasks):
          
            
#             self.env_info.task_rank = i
            
#             self.env_info.env_args = single_mt_env_args
#             self.env_info.env_args["task_name"] = task

            
#             start_epoch = 0
#             self.env_info.env_args["task_rank"] = i
#             p = mp.Process(
#                 target=self.__class__.train_worker_process,
#                 args=( self.__class__, self.shared_funcs,self.embedding,
#                     self.env_info, self.replay_buffer, 
#                     self.shared_que, self.start_barrier,
#                     self.train_epochs, start_epoch, task, self.shared_dict))
#             p.start()
#             self.workers.append(p)
#             # i += 1



#         assert self.eval_worker_nums == len(self.task_list)
      
#         self.env_info.env = self.env
#         self.env_info.num_tasks = len(self.task_list)
       
#         single_mt_env_args = {
#             "task_name": None,
#             "task_rank": 0,
#             "num_tasks": len(self.task_list),
#             "max_obs_dim": np.prod(self.env.observation_space.shape),
#         }

#         for i, task in enumerate(tasks):
           

#             self.env_info.task_rank = i

#             self.env_info.env_args = single_mt_env_args
#             self.env_info.env_args["task_name"] = task

#             start_epoch = 0
  

#             self.env_info.env_args["task_rank"] = i
#             eval_p = mp.Process(
#                 target=self.__class__.eval_worker_process,
#                 args=(self.shared_funcs["pf_state"],self.shared_funcs["pf_action"],self.embedding,
#                     self.env_info, self.eval_shared_que, self.eval_start_barrier,
#                     self.eval_epochs, start_epoch, task, self.shared_dict))
#             eval_p.start()
#             self.eval_workers.append(eval_p)


#     def eval_one_epoch(self):
        
#         eval_rews = []
      
#         self.shared_funcs["pf_state"].load_state_dict(self.funcs["pf_state"].state_dict())
#         self.shared_funcs["pf_action"].load_state_dict(self.funcs["pf_action"].state_dict())
#         tasks_result = []

#         active_task_counts = 0
#         for _ in range(self.eval_worker_nums):
#             worker_rst = self.eval_shared_que.get()
#             if worker_rst["eval_rewards"] is not None:
#                 active_task_counts += 1
#                 eval_rews += worker_rst["eval_rewards"]
#                 tasks_result.append((worker_rst["task_name"], 
#                 np.mean(worker_rst["eval_rewards"])))

#         tasks_result.sort()

#         dic = OrderedDict()
#         for task_name, eval_rewards in tasks_result:
            
#             dic[task_name+"_eval_rewards"] = eval_rewards
#             self.tasks_progress[self.tasks_mapping[task_name]] *= \
#                 (1 - self.progress_alpha)

#         dic['eval_rewards']      = eval_rews
        

#         return dic


#     def train_one_epoch(self):
#         train_rews = []
#         train_epoch_reward = 0

#         for key in self.shared_funcs:
#             self.shared_funcs[key].load_state_dict(self.funcs[key].state_dict())
        
#         active_worker_nums = 0
#         for _ in range(self.worker_nums):
#             worker_rst = self.shared_que.get()
            
#             if worker_rst["train_rewards"] is not None:
                
#                 train_rews += worker_rst["train_rewards"]
#                 train_epoch_reward += worker_rst["train_epoch_reward"]
#                 active_worker_nums += 1
#         self.active_worker_nums = active_worker_nums
     
#         return {
#             'train_rewards':train_rews,
#             'train_epoch_reward':train_epoch_reward
#         }
#     def to(self, device):
#         for func in self.funcs:
#             self.funcs[func].to(device)
#     @property
#     def funcs(self):
#         return {
#             "pf_state": self.pf[0],
#             "pf_action": self.pf[1]
#         }



# class AsyncMultiTaskParallelCollectorForActionRepresentation_Embed_Net(AsyncSingleTaskParallelCollector):

#     def __init__(self, embedding, progress_alpha=0.1, **kwargs):
#         self.embedding = embedding
#         self.share_embedding = copy.copy(self.embedding)
#         super().__init__(**kwargs)
#         self.tasks=self.task_list
#         self.tasks_mapping = {}
#         for idx, task_name in enumerate(self.tasks):
#             self.tasks_mapping[task_name] = idx
#         self.tasks_progress = [0 for _ in range(len(self.tasks))]
#         self.progress_alpha = progress_alpha
#         self.pf_state=self.pf[0]
#         self.pf_action=self.pf[1]

#     @classmethod
#     def take_actions(cls, funcs, embedding, env_info, ob_info, replay_buffer):

#         pf_state = funcs["pf_state"]
#         pf_action = funcs["pf_action"]
#         ob = ob_info["ob"]
#         task_idx = env_info.task_rank
        

#         pf_state.eval()
#         pf_action.eval()

#         with torch.no_grad():
        
#             idx_input = torch.Tensor([[task_idx]]).to(env_info.device).long()
#             task_input = torch.zeros(env_info.num_tasks)
#             task_input[env_info.task_rank] = 1
#             task_input = task_input.to(env_info.device).unsqueeze(0)
#             ob = torch.Tensor( ob ).to(env_info.device).unsqueeze(0)
#             representation = pf_state.forward(ob)
#             out = pf_action.explore(representation, embedding)
#             act = out["action"]
#             # act = act[0]
            


#         act = act.detach().cpu().numpy()
#         if not env_info.continuous:
#             act = act[0]
        
#         if type(act) is not int:
#             if np.isnan(act).any():
#                 print("NaN detected. BOOM")
#                 exit()

#         next_ob, rewards, done, info = env_info.env.step(act)
#         reward=rewards[task_idx]
#         if env_info.train_render:
#             env_info.env.render()
#         env_info.current_step += 1

#         sample_dict = {
#             "obs": ob,
#             "next_obs": next_ob,
#             "acts": act,
#             "task_idxs": [env_info.task_rank],
#             "rewards": [reward],
#             "terminals": [done],
#             "task_inputs": task_input.cpu().numpy(),
#             "embeddings": embedding.detach().cpu().numpy()
#         }
       

#         if done or env_info.current_step >= env_info.max_episode_frames:
#             next_ob = env_info.env.reset()
#             env_info.finish_episode()
#             env_info.start_episode() # reset current_step

#         replay_buffer.add_sample( sample_dict, env_info.task_rank)

#         return next_ob, done, reward, info

#     @staticmethod
#     def train_worker_process(cls, shared_funcs, share_embedding, env_info,
#         replay_buffer, shared_que,
#         start_barrier, epochs, start_epoch, task_name, shared_dict):

#         replay_buffer.rebuild_from_tag()
#         local_funcs = copy.deepcopy(shared_funcs)
#         local_embedding = copy.deepcopy(share_embedding)
        
        
#         for key in local_funcs:
#             local_funcs[key].to(env_info.device)
#         local_embedding.to(env_info.device)
#         c_ob = {
#             "ob": env_info.env.reset()
#         }
#         train_rew = 0
#         current_epoch = 0
#         while True:
#             start_barrier.wait()
#             current_epoch += 1
#             if current_epoch < start_epoch:
#                 shared_que.put({
#                     'train_rewards': None,
#                     'train_epoch_reward': None
#                 })
#                 continue
#             if current_epoch > epochs:
#                 break

#             for key in shared_funcs:
#                 local_funcs[key].load_state_dict(shared_funcs[key].state_dict())
#             local_embedding = share_embedding
#             local_embedding = local_embedding.unsqueeze(0).to("cpu")
#             train_rews = []
#             train_epoch_reward = 0    

#             for t in range(env_info.epoch_frames):

#                 next_ob, done, reward, info = cls.take_actions(local_funcs, local_embedding, env_info, c_ob, replay_buffer )
#                 c_ob["ob"] = next_ob
#                 train_rew += reward
#                 train_epoch_reward += reward
#                 train_rews.append(train_rew)
#                 train_rew = 0
#                 # if done:
#                 #     train_rews.append(train_rew)
#                 #     train_rew = 0
            
#             shared_que.put({
#                 'train_rewards':train_rews,
#                 'train_epoch_reward':train_epoch_reward
#             })

#     @staticmethod
#     def eval_worker_process(shared_pf_state,shared_pf_action, shared_embedding,
#         env_info, shared_que, start_barrier, epochs, start_epoch, task_name, shared_dict):

#         pf_state = copy.deepcopy(shared_pf_state).to(env_info.device)
#         pf_action = copy.deepcopy(shared_pf_action).to(env_info.device)
       
#         current_epoch = 0
#         while True:
#             start_barrier.wait()
#             current_epoch += 1
#             if current_epoch < start_epoch:
#                 shared_que.put({
#                     'eval_rewards': None,
#                     'task_name': task_name
#                 })
#                 continue
#             if current_epoch > epochs:
#                 break
#             pf_state.load_state_dict(shared_pf_state.state_dict())
#             pf_action.load_state_dict(shared_pf_action.state_dict())
#             pf_state.eval()
#             pf_action.eval()
#             embedding = copy.deepcopy(shared_embedding).to(env_info.device)
#             embedding = embedding.unsqueeze(0).to("cpu")
#             eval_rews = []  
  
#             done = False
          
#             for idx in range(env_info.eval_episodes):

#                 eval_ob = env_info.env.reset()
#                 rew = 0

#                 task_idx = env_info.task_rank
                
#                 while not done:
#                     eval_ob =  torch.Tensor(eval_ob).to(env_info.device).unsqueeze(0)
#                     representation = pf_state.forward(eval_ob)
#                     act = pf_action.eval_act(representation, embedding)
#                     eval_ob, rs, done, info = env_info.env.step( act )
#                     r=rs[task_idx]
#                     rew += r
#                     if env_info.eval_render:
#                         env_info.env.render()
                   

#                 eval_rews.append(rew)
#                 done = False
      
#             shared_que.put({
#                 'eval_rewards': eval_rews,
#                 'embedding': shared_embedding.detach().cpu(),
#                 'task_name': task_name
#             })

#     def start_worker(self):
#         self.workers = []
#         self.shared_que = self.manager.Queue(self.worker_nums)
#         self.start_barrier = mp.Barrier(self.worker_nums)
                
#         self.eval_workers = []
#         self.eval_shared_que = self.manager.Queue(self.eval_worker_nums)
#         self.eval_start_barrier = mp.Barrier(self.eval_worker_nums)


#         self.shared_dict = self.manager.dict()

        

#         assert self.worker_nums == len(self.task_list)

#         self.env_info.env = self.env
       
        
#         self.env_info.num_tasks = len(self.task_list)
       
#         single_mt_env_args = {
#             "task_name": None,
#             "task_rank": 0,
#             "num_tasks": len(self.task_list),
#             "max_obs_dim": np.prod(self.env.observation_space.shape),
#         }
        
      
#         tasks=self.task_list
#         for i, task in enumerate(tasks):
          
            
#             self.env_info.task_rank = i
#             self.env_info.env_args = single_mt_env_args
#             self.env_info.env_args["task_name"] = task
            
            
#             start_epoch = 0
#             self.env_info.env_args["task_rank"] = i
#             p = mp.Process(
#                 target=self.__class__.train_worker_process,
#                 args=( self.__class__, self.shared_funcs, self.share_embedding,
#                     self.env_info, self.replay_buffer, 
#                     self.shared_que, self.start_barrier,
#                     self.train_epochs, start_epoch, task, self.shared_dict))
#             p.start()
#             self.workers.append(p)




#         assert self.eval_worker_nums == len(self.task_list)
      
#         self.env_info.env = self.env
#         self.env_info.num_tasks = len(self.task_list)
       
#         single_mt_env_args = {
#             "task_name": None,
#             "task_rank": 0,
#             "num_tasks": len(self.task_list),
#             "max_obs_dim": np.prod(self.env.observation_space.shape),
#         }

#         for i, task in enumerate(tasks):
           

#             self.env_info.task_rank = i

#             self.env_info.env_args = single_mt_env_args
#             self.env_info.env_args["task_name"] = task

#             start_epoch = 0
  

#             self.env_info.env_args["task_rank"] = i
#             eval_p = mp.Process(
#                 target=self.__class__.eval_worker_process,
#                 args=(self.shared_funcs["pf_state"],self.shared_funcs["pf_action"],self.share_embedding,
#                     self.env_info, self.eval_shared_que, self.eval_start_barrier,
#                     self.eval_epochs, start_epoch, task, self.shared_dict))
#             eval_p.start()
#             self.eval_workers.append(eval_p)


#     def eval_one_epoch(self):
        
#         eval_rews = []
      
#         self.shared_funcs["pf_state"].load_state_dict(self.funcs["pf_state"].state_dict())
#         self.shared_funcs["pf_action"].load_state_dict(self.funcs["pf_action"].state_dict())
#         tasks_result = []

#         active_task_counts = 0
#         for _ in range(self.eval_worker_nums):
#             worker_rst = self.eval_shared_que.get()
#             print(worker_rst['embedding'])
#             if worker_rst["eval_rewards"] is not None:
#                 active_task_counts += 1
#                 eval_rews += worker_rst["eval_rewards"]
#                 tasks_result.append((worker_rst["task_name"], 
#                 np.mean(worker_rst["eval_rewards"])))

#         tasks_result.sort()

#         dic = OrderedDict()
#         for task_name, eval_rewards in tasks_result:
            
#             dic[task_name+"_eval_rewards"] = eval_rewards
#             self.tasks_progress[self.tasks_mapping[task_name]] *= \
#                 (1 - self.progress_alpha)

#         dic['eval_rewards']      = eval_rews
        

#         return dic


#     def train_one_epoch(self):
#         train_rews = []
#         train_epoch_reward = 0

#         for key in self.shared_funcs:
#             self.shared_funcs[key].load_state_dict(self.funcs[key].state_dict())
#         # self.share_embedding = self.embedding
#         active_worker_nums = 0
#         for _ in range(self.worker_nums):
#             worker_rst = self.shared_que.get()
            
#             if worker_rst["train_rewards"] is not None:
                
#                 train_rews += worker_rst["train_rewards"]
#                 train_epoch_reward += worker_rst["train_epoch_reward"]
#                 active_worker_nums += 1
#         self.active_worker_nums = active_worker_nums
     
#         return {
#             'train_rewards':train_rews,
#             'train_epoch_reward':train_epoch_reward
#         }


#     def to(self, device):
#         for func in self.funcs:
#             self.funcs[func].to(device)
            
#     @property
#     def funcs(self):
#         return {
#             "pf_state": self.pf[0],
#             "pf_action": self.pf[1]
#         }
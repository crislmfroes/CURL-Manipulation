import torch
import numpy as np
import torch.nn as nn
import gym
import os
from collections import deque
import random
from torch.utils.data import Dataset, DataLoader
import time
from skimage.util.shape import view_as_windows
import copy
import cv2
import lz4.frame

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def make_dir(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


class ReplayBuffer:
    """Buffer to store environment transitions."""
    def __init__(self, action_shape, capacity, batch_size, device, image_size=84,transform=None, alpha=0.6, beta_start = 0.4, beta_frames=100000, config=None, n_envs=1):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size
        self.transform = transform
        self.n_envs = n_envs
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        #obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
        obs_dtype = 'O'

        #self.obses = np.empty(shape=(capacity, 1), dtype=obs_dtype)
        #self.next_obses = np.empty(shape=(capacity, 1), dtype=obs_dtype)
        #self.obses = deque([], maxlen=capacity)
        #self.next_obses = deque([], maxlen=capacity)
        self.obses = dict([(encoder_config['name'], np.empty((capacity, *encoder_config['obs_shape']), dtype=encoder_config['obs_dtype'])) for encoder_config in config['encoders']])
        self.next_obses = dict([(encoder_config['name'], np.empty((capacity, *encoder_config['obs_shape']), dtype=encoder_config['obs_dtype'])) for encoder_config in config['encoders']])
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        #self.priorities = np.empty((capacity, 1), dtype=np.float32)

        #self.alpha = alpha
        #self.beta_start = beta_start
        #self.beta_frames = beta_frames
        #self.frame = 1 #for beta calculation

        self.idx = 0
        self.last_save = 0
        self.full = False
        self.config = config

    def beta_by_frame(self, frame_idx):
        """
        Linearly increases beta from beta_start to 1 over time from 1 to beta_frames.
        
        3.4 ANNEALING THE BIAS (Paper: PER)
        We therefore exploit the flexibility of annealing the amount of importance-sampling
        correction over time, by defining a schedule on the exponent 
        that reaches 1 only at the end of learning. In practice, we linearly anneal from its initial value 0 to 1
        """
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = abs(prio) 
    

    def add(self, obs, action, reward, next_obs, done, previous_done):
        added_samples = 0
        for i in range(self.n_envs):
            if previous_done[i] == False:
                #print('obs', obs)
                #assert type(obs['robot_head_depth']) == np.ndarray
                #assert obs['robot_head_depth'].shape == (4,100,100)
                #assert obs['robot_arm_depth'].shape == (4,100,100)
                np.copyto(self.actions[self.idx], action[i])
                np.copyto(self.rewards[self.idx], reward[i])
                #np.copyto(self.next_obses[self.idx], next_obs)
                np.copyto(self.not_dones[self.idx], not done[i])

                #max_prio = self.priorities.max() if self.idx > 0 else 1.0 # gives max priority if buffer is not empty else 1

                #np.copyto(self.priorities[self.idx], max_prio)
                added_samples += 1
                for encoder_config in self.config['encoders']:
                    encoder_name = encoder_config['name']
                    #self.obses.append(dict([(k, obs[k][i]) for k in obs.keys()]))
                    #self.next_obses.append(dict([(k, next_obs[k][i]) for k in next_obs.keys()]))
                    #np.copyto(self.obses[self.idx], obs)
                    np.copyto(self.obses[encoder_name][self.idx], obs[encoder_name][i])
                    np.copyto(self.next_obses[encoder_name][self.idx], next_obs[encoder_name][i])

        self.idx = (self.idx + added_samples) % self.capacity
        self.full = self.full or self.idx == 0

    def sample_proprio(self):
        
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )
        
        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses = torch.as_tensor(obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            next_obses, device=self.device
        ).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return obses, actions, rewards, next_obses, not_dones

    def sample_cpc(self):

        start = time.time()

        N = self.idx if not self.full else self.capacity
        #if N == self.capacity:
        #    prios = self.priorities
        #else:
        #    prios = self.priorities[:self.idx]
        #print(prios)
        # calc P = p^a/sum(p^a)
        #probs  = prios ** self.alpha
        #P = probs/probs.sum()
        #P = np.ones(shape=(N, 1), dtype=np.float32)/N
        #gets the indices depending on the probability p
        #idxs = np.random.choice(N, self.batch_size, p=P.flatten())

        idxs = np.random.choice(N, self.batch_size)
        #print(idxs)

        #idxs = np.random.randint(
        #    0, self.capacity if self.full else self.idx, size=self.batch_size
        #)
      
        #obses = copy.deepcopy([dict(self.obses[i]) for i in idxs])
        #next_obses = copy.deepcopy([dict(self.next_obses[i]) for i in idxs])
        #pos = copy.deepcopy(obses)
        obses = dict([(encoder_config['name'], random_crop(self.obses[encoder_config['name']][idxs].copy(), self.image_size)) if encoder_config['name'] in self.config['image_fields'] else (encoder_config['name'], self.obses[encoder_config['name']][idxs]) for encoder_config in self.config['encoders']])
        next_obses = dict([(encoder_config['name'], random_crop(self.next_obses[encoder_config['name']][idxs].copy(), self.image_size)) if encoder_config['name'] in self.config['image_fields'] else (encoder_config['name'], self.next_obses[encoder_config['name']][idxs]) for encoder_config in self.config['encoders']])
        pos = dict([(k, obses[k].copy()) for k in obses.keys()])
        
        #print(idxs)
        '''for idx in idxs:
            #print(obses[i]['robot_head_depth'])
            #print(self.obses)
            obs = copy.deepcopy(self.obses[idx])
            next_obs = copy.deepcopy(self.next_obses[idx])
            posi = copy.deepcopy(self.obses[idx])

            for field in self.config['image_fields']:
                obs[field] = random_crop(obs[field].copy(), self.image_size)
                
                #print(idx, type(next_obs[field]))
                next_obs[field] = random_crop(next_obs[field].copy(), self.image_size)

                posi[field] = random_crop(posi[field].copy(), self.image_size)
            
            obses.append(obs)
            next_obses.append(next_obs)
            pos.append(posi)'''
            
        #obses = torch.as_tensor(obses.tolist(), device=self.device).float()
        #next_obses = torch.as_tensor(next_obses.tolist(), device=self.device).float()
        #pos = torch.as_tensor(pos.tolist(), device=self.device).float()

        '''obses = random_crop(obses, self.image_size)
        next_obses = random_crop(next_obses, self.image_size)
        pos = random_crop(pos, self.image_size)
    
        for i in range(obses.shape[0]):
            obses[i]['robot_head_depth'] = torch.as_tensor(obses[i]['robot_head_depth'], device=self.device).float()
            next_obses[i]['robot_head_depth'] = torch.as_tensor(
                next_obses[i]['robot_head_depth'], device=self.device
            ).float()'''
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        #beta = self.beta_by_frame(self.frame)
        #self.frame+=1

        #Compute importance-sampling weight
        #weights  = (N * P[idxs]) ** (-beta)
        # normalize weights
        #weights /= weights.max() 
        #weights  = np.array(weights, dtype=np.float32)

        tensor_obses = {}
        tensor_pos = {}
        tensor_next_obses = {}

        for encoder_config in self.config['encoders']:
            name = encoder_config['name']
            #for i in range(len(obses)):
            #    print(i, obses[i][name].shape)
            '''tensor_obses[name] = torch.as_tensor(np.array([obs[name] for obs in obses]), device=self.device)
            tensor_pos[name] = torch.as_tensor(np.array([obs[name] for obs in pos]), device=self.device)
            tensor_next_obses[name] = torch.as_tensor(np.array([obs[name] for obs in next_obses]), device=self.device)'''
            tensor_obses[name] = torch.as_tensor(obses[name], device=self.device)
            tensor_pos[name] = torch.as_tensor(pos[name], device=self.device)
            tensor_next_obses[name] = torch.as_tensor(next_obses[name], device=self.device)

        #weights = torch.as_tensor(weights, device=self.device)
        cpc_kwargs = dict(obs_anchor=tensor_obses, obs_pos=tensor_pos,
                          time_anchor=None, time_pos=None)
        #print('tensor obses: ', tensor_obses)
        #print('actions: ', actions)
        #exit()
        return tensor_obses, actions, rewards, tensor_next_obses, not_dones, cpc_kwargs#, weights, idxs

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end

class ReorderObs(gym.Wrapper):
    def __init__(self, env, obs_field, image_size):
        gym.Wrapper.__init__(self, env)
        self.image_size = image_size
        self.obs_field = obs_field
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.observation_space[obs_field] = gym.spaces.Box(shape=(1,image_size,image_size), low=0.0, high=1.0, dtype=np.float32)
        '''shp = env.observation_space['robot_head_depth'].shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=np.array(shp)[self._ordering],
            dtype=env.observation_space.dtype
        )'''

    def reset(self):
        obs = self.env.reset()
        return self._preprocess_obs(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._preprocess_obs(obs), reward, done, info

    def _preprocess_obs(self, obs):
        #print(cv2.resize(obs['robot_head_depth'], (100,100)).shape)
        obs[self.obs_field] = cv2.resize(obs[self.obs_field], (self.image_size,self.image_size)).reshape((1,self.image_size,self.image_size))
        return obs

    def render(self, mode):
        return self.env.render(mode)

class FrameStack(gym.Wrapper):
    def __init__(self, env, k, obs_field, image_size):
        gym.Wrapper.__init__(self, env)
        self.image_size = image_size
        self._k = k
        self._frames = deque([], maxlen=k)
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.observation_space[obs_field] = gym.spaces.Box(shape=(4, self.image_size, self.image_size), low=0.0, high=1.0, dtype=np.float32)
        '''shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )'''
        self._max_episode_steps = env._max_episode_steps
        self.obs_field = obs_field

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs[self.obs_field])
        return self._get_obs(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs[self.obs_field])
        return self._get_obs(obs), reward, done, info

    def _get_obs(self, obs):
        assert len(self._frames) == self._k
        obs[self.obs_field] = np.concatenate(self._frames, axis=0)
        return obs

    def render(self, mode):
        return self.env.render(mode)

class ActionRepeat(gym.Wrapper):
    def __init__(self, env, frame_skip):
        gym.Wrapper.__init__(self, env)
        self.frame_skip = frame_skip
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self._max_episode_steps = env._max_episode_steps // frame_skip

    def reset(self):
        return self.env.reset()

    def step(self, action):
        total_reward = 0
        for i in range(self.frame_skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

    def render(self, mode):
        return self.env.render(mode)


def random_crop(imgs, output_size):
    #print('imgs', imgs.shape)
    img_height = imgs.shape[2]
    img_width = imgs.shape[3]
    height = output_size
    width = output_size
    r = np.random.randint(0, (img_height-height+1))
    c = np.random.randint(0, (img_width-width+1))
    cropped_imgs = imgs[:,:,r:r+height,c:c+width]
    '''if cropped_imgs.shape[1:] != (84, 84):
        print('error!')
        print(imgs.shape)
        print(cropped_imgs.shape)
        exit()'''
    #print('cropped_imgs', cropped_imgs.shape)
    return cropped_imgs

def center_crop_image(obs, output_size, img_fields):
    cropped_obs = dict()
    for field in obs.keys():
        if field in img_fields:
            image = obs[field]
            img_height = image.shape[2]
            img_width = image.shape[3]
            height = output_size
            width = output_size
            r = (img_height-height)//2
            c = (img_width-width)//2
            cropped_obs[field] = image[:,:,r:r+height,c:c+width]
            #print(obs['robot_head_depth'].shape)
        else:
            cropped_obs[field] = obs[field][:]
    return cropped_obs




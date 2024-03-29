import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

import utils
from encoder import make_encoder

LOG_FREQ = 10000


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Actor(nn.Module):
    """MLP actor network."""
    def __init__(
        self, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters, config=None
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type, encoder_feature_dim, num_layers,
            num_filters, output_logits=True, config=config
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(
        self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False
    ):
        obs = self.encoder(obs, detach=detach_encoder)

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        L.log_param('train_actor/fc1', self.trunk[0], step)
        L.log_param('train_actor/fc2', self.trunk[2], step)
        L.log_param('train_actor/fc3', self.trunk[4], step)


class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.Dropout(), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(
        self, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, num_layers, num_filters, config=None
    ):
        super().__init__()


        self.encoder = make_encoder(
            encoder_type, encoder_feature_dim, num_layers,
            num_filters, output_logits=True, config=config
        )

        self.Q1 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim
        )
        self.Q2 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        obs = self.encoder(obs, detach=detach_encoder)

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        self.encoder.log(L, step, log_freq)

        for k, v in self.outputs.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        for i in range(3):
            L.log_param('train_critic/q1_fc%d' % i, self.Q1.trunk[i * 2], step)
            L.log_param('train_critic/q2_fc%d' % i, self.Q2.trunk[i * 2], step)


class CURL(nn.Module):
    """
    CURL
    """

    def __init__(self, z_dim, batch_size, critic_encoder, critic_target_encoder, output_type="continuous", config=None):
        super(CURL, self).__init__()
        self.batch_size = batch_size

        self.encoder = critic_encoder

        self.encoder_target = critic_target_encoder 

        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        self.output_type = output_type

    def encode(self, x, detach=False, ema=False):
        """
        Encoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, value in r2
        """
        if ema:
            with torch.no_grad():
                z_out = self.encoder_target(x)
        else:
            z_out = self.encoder(x)

        if detach:
            z_out = z_out.detach()
        return z_out

    def compute_logits(self, z_a, z_pos):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits

class CurlSacAgent(object):
    """CURL representation learning with SAC."""
    def __init__(
        self,
        action_shape,
        device,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_type='multi_input',
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        num_layers=4,
        num_filters=32,
        cpc_update_freq=1,
        log_interval=100,
        detach_encoder=False,
        curl_latent_dim=128,
        image_size=84,
        config=None
    ):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.cpc_update_freq = cpc_update_freq
        self.log_interval = log_interval
        self.image_size = image_size
        self.curl_latent_dim = curl_latent_dim
        self.detach_encoder = detach_encoder
        self.encoder_type = encoder_type
        self.config = config

        self.actor = Actor(
            action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters, config=config
        ).to(device)

        self.critic = Critic(
            action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, config=config
        ).to(device)

        self.critic_target = Critic(
            action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, config=config
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic, and CURL and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)
        
        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        if self.encoder_type == 'multi_input':
            '''self.curls = {}
            self.encoder_optimizers = {}
            self.cpc_optimizers = {}'''
            '''for image_field_name in config['image_fields']:
                # create CURL encoder (the 128 batch size is probably unnecessary)
                self.curls[image_field_name] = CURL(encoder_feature_dim,
                            self.curl_latent_dim, self.critic.encoder.encoders[image_field_name],self.critic_target.encoder.encoders[image_field_name], output_type='continuous', config=config).to(self.device)

                # optimizer for critic encoder for reconstruction loss
                self.encoder_optimizers[image_field_name] = torch.optim.Adam(
                    self.critic.encoder.encoders[image_field_name].parameters(), lr=encoder_lr
                )

                self.cpc_optimizers[image_field_name] = torch.optim.Adam(
                    self.curls[image_field_name].parameters(), lr=encoder_lr
                )'''
            self.CURL = CURL(encoder_feature_dim,
                            self.curl_latent_dim, self.critic.encoder,self.critic_target.encoder, output_type='continuous', config=config).to(self.device)
            self.encoder_optimizer = torch.optim.Adam(
                    self.critic.encoder.parameters(), lr=encoder_lr
                )
            self.cpc_optimizer = torch.optim.Adam(
                    self.CURL.parameters(), lr=encoder_lr
                )
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.encoder_type == 'multi_input':
            self.CURL.train(training)
            '''for image_field_name in self.config['image_fields']:
                self.curls[image_field_name].train(training)
'''
    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        #print(obs['robot_head_depth'].shape)
        #assert obs['robot_head_depth'].shape == (4, 84, 84)
        new_obs = utils.center_crop_image(obs, self.image_size, self.config['image_fields'])
        #new_obs = obs
        #print(new_obs)
        #print(new_obs['robot_head_depth'].shape)
        for encoder_config in self.config['encoders']:
            name = encoder_config['name']
            #print(type(new_obs[name]))
            new_obs[name] = torch.as_tensor(np.array(new_obs[name]), device=self.device)
            #print(type(new_obs[name]))
        #print(new_obs)
        with torch.no_grad():
            '''img_obs = torch.FloatTensor(obs['robot_head_depth']).to(self.device)
            img_obs = img_obs.unsqueeze(0)'''
            mu, _, _, _ = self.actor(
                new_obs, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy()

    def sample_action(self, obs):
        new_obs = utils.center_crop_image(obs, self.image_size, self.config['image_fields'])
        #new_obs = obs
        for encoder_config in self.config['encoders']:
            name = encoder_config['name']
            #print(type(new_obs[name]))
            new_obs[name] = torch.as_tensor(np.array(new_obs[name]), device=self.device)
            #print(type(new_obs[name]))
        with torch.no_grad():
            '''img_obs = torch.FloatTensor(obs['robot_head_depth']).to(self.device)
            img_obs = img_obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(img_obs, compute_log_pi=False)'''
            mu, pi, _, _ = self.actor(new_obs, compute_log_pi=False)
            return pi.cpu().data.numpy()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step, replay_buffer, should_log=True):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(
            obs, action, detach_encoder=self.detach_encoder)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        '''td_error1 = target_Q.detach() - current_Q1#,reduction="none"
        td_error2 = target_Q.detach() - current_Q2
        critic1_loss = 0.5* (td_error1.pow(2)*weights).mean()
        critic2_loss = 0.5* (td_error2.pow(2)*weights).mean()
        critic_loss = critic1_loss + critic2_loss
        prios = abs(((td_error1 + td_error2)/2.0 + 1e-5).squeeze())'''
        if should_log:
            L.log('train_critic/loss', critic_loss, step)


        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        #replay_buffer.update_priorities(idxs, prios.data.cpu().numpy())
        self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, L, step, replay_buffer, should_log=True):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if should_log:
            L.log('train_actor/loss', actor_loss, step)
            L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * \
            (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        if should_log:                                    
            L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        if should_log:
            L.log('train_alpha/loss', alpha_loss, step)
            L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_cpc(self, obs_anchor, obs_pos, cpc_kwargs, L, step, should_log=True):
        z_a = self.CURL.encode(obs_anchor)
        z_pos = self.CURL.encode(obs_pos, ema=True)
            
        logits = self.CURL.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        loss = self.cross_entropy_loss(logits, labels)
            
        self.encoder_optimizer.zero_grad()
        self.cpc_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.cpc_optimizer.step()
        if should_log:
            L.log('train/curl_loss', loss, step)
        
        '''for image_field_name in self.config['image_fields']:

            z_a = self.curls[image_field_name].encode(obs_anchor[image_field_name])
            z_pos = self.curls[image_field_name].encode(obs_pos[image_field_name], ema=True)
            
            logits = self.curls[image_field_name].compute_logits(z_a, z_pos)
            labels = torch.arange(logits.shape[0]).long().to(self.device)
            loss = self.cross_entropy_loss(logits, labels)
            
            self.encoder_optimizers[image_field_name].zero_grad()
            self.cpc_optimizers[image_field_name].zero_grad()
            loss.backward()

            self.encoder_optimizers[image_field_name].step()
            self.cpc_optimizers[image_field_name].step()
            if step % self.log_interval == 0:
                L.log('train/curl_{}_loss'.format(image_field_name), loss, step)'''


    def update(self, replay_buffer, L, step, should_log=True):
        if self.encoder_type == 'multi_input':
            obs, action, reward, next_obs, not_done, cpc_kwargs = replay_buffer.sample_cpc()
            #print(action)
            '''#print(obs[0]['robot_head_depth'].shape)
            depth_arr = np.array([obsi['robot_head_depth'] for obsi in obs])
            depth_obs = torch.tensor(depth_arr).cuda()
            next_depth_arr = np.array([obsi['robot_head_depth'] for obsi in next_obs])
            next_depth_obs = torch.tensor(next_depth_arr).cuda()
            #next_depth_obs = torch.tensor([obsi['robot_head_depth'] for obsi in next_obs]).cuda()
            weights = torch.tensor(weights).cuda()
            #print(depth_obs.shape)'''
        else:
            exit()
            obs, action, reward, next_obs, not_done = replay_buffer.sample_proprio()
    
        if should_log:
            L.log('train/batch_reward', reward.mean(), step)

        critic_updates = 20

        for _ in range(critic_updates):
            self.update_critic(obs, action, reward, next_obs, not_done, L, step, replay_buffer, should_log=should_log)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step, replay_buffer, should_log=should_log)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )
        
        if step % self.cpc_update_freq == 0 and self.encoder_type == 'multi_input':
            obs_anchor, obs_pos = cpc_kwargs["obs_anchor"], cpc_kwargs["obs_pos"]
            '''depth_obs_anchor = torch.tensor([obsi['robot_head_depth'] for obsi in obs_anchor]).cuda()
            depth_obs_pos = torch.tensor([obsi['robot_head_depth'] for obsi in obs_pos]).cuda()'''
            '''depth_anchor_arr = np.array([obsi['robot_head_depth'] for obsi in obs_anchor])
            depth_obs_anchor = torch.tensor(depth_anchor_arr).cuda()

            depth_pos_arr = np.array([obsi['robot_head_depth'] for obsi in obs_pos])
            depth_obs_pos = torch.tensor(depth_pos_arr).cuda()'''
            self.update_cpc(obs_anchor, obs_pos,cpc_kwargs, L, step, should_log=should_log)

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )

    def save_curl(self, model_dir, step):
        '''for image_field_name in self.config['image_fields']:
            torch.save(
                self.curls[image_field_name].state_dict(), '%s/curl_%s_%s.pt' % (model_dir, image_field_name, step)
            )'''
        torch.save(
                self.CURL.state_dict(), '%s/curl_%s.pt' % (model_dir, step)
        )
    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )

    def load_curl(self, model_dir, step):
        self.CURL.load_state_dict(
            torch.load('%s/curl_%s.pt' % (model_dir, step))
        )
 
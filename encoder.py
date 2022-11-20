import torch
import torch.nn as nn


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


# for 84 x 84 inputs
OUT_DIM = {2: 39, 4: 35, 6: 31}
# for 64 x 64 inputs
OUT_DIM_64 = {2: 29, 4: 25, 6: 21}


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32,output_logits=False, image_size=84):
        super().__init__()

        #assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        #print(image_size)

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = OUT_DIM_64[num_layers] if image_size == 64 else OUT_DIM[num_layers] 
        #print(out_dim)
        #print(num_filters*out_dim*out_dim)
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()
        self.output_logits = output_logits

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        #obs = obs / 255.
        #obs = obs.reshape((-1, 12, 84, 84))
        #print(obs.shape)
        self.outputs['obs'] = obs
        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            #print('conv', i, conv.shape)
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
            self.outputs['tanh'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters,*args):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass

class MultiInputEncoder(nn.Module):
    def __init__(self, feature_dim, config, output_logits=False, device='cuda'):
        super().__init__()
        encoder_dict = dict()
        for encoder_config in config['encoders']:
            encoder_dict[encoder_config['name']] = make_encoder(**encoder_config).to(device)
        self.encoders = nn.ModuleDict(encoder_dict)
        self.feature_dim = feature_dim
        self.fc = nn.Linear(sum([encoder_config['feature_dim'] for encoder_config in config['encoders']]), self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)
        self.outputs = dict()
        self.output_logits = output_logits
    
    def forward(self, obs, detach=False):
        outputs = []
        for key in self.encoders.keys():
            output = self.encoders[key](obs[key])
            #print(output.shape)
            outputs.append(output)
        h = torch.cat(outputs, dim=1)

        if detach:
            h = h.detach()
        #print(h.device)
        #print(self.fc)
        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
            self.outputs['tanh'] = out

        return out

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass



_AVAILABLE_ENCODERS = {'pixel': PixelEncoder, 'identity': IdentityEncoder}


def make_encoder(
    encoder_type=None, obs_shape=None, feature_dim=None, num_layers=None, num_filters=None, output_logits=False, config={}
, image_size=84, **kwargs):
    assert encoder_type in _AVAILABLE_ENCODERS or encoder_type == 'multi_input'
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters, output_logits, image_size
    ) if encoder_type != 'multi_input' else MultiInputEncoder(feature_dim, config, output_logits)

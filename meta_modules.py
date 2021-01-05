import torch
from torch import nn
from collections import OrderedDict
from modules import *
import numpy as np
    
class MetaSDF(nn.Module):
    def __init__(self, hypo_module, loss, init_lr=1e-1, num_meta_steps=3, first_order=False, lr_type='per_parameter'):
        super().__init__()

        self.hypo_module = hypo_module
        self.loss = loss
        self.num_meta_steps = num_meta_steps
        
        self.first_order = first_order

        self.lr_type = lr_type
        if self.lr_type == 'static':
            self.register_buffer('lr', torch.Tensor([init_lr]))
        elif self.lr_type == 'global':
            self.lr = nn.Parameter(torch.Tensor([init_lr]))
        elif self.lr_type == 'per_step':
            self.lr = nn.ParameterList([nn.Parameter(torch.Tensor([init_lr]))
                                        for _ in range(num_meta_steps)])
        elif self.lr_type == 'per_parameter':
            self.lr = nn.ModuleList([])
            hypo_parameters = hypo_module.parameters()
            for param in hypo_parameters:
                self.lr.append(nn.ParameterList([nn.Parameter(torch.ones(param.size()) * init_lr)
                                                 for _ in range(num_meta_steps)]))
        elif self.lr_type == 'simple_per_parameter':
            self.lr = nn.ParameterList([nn.Parameter(torch.Tensor([init_lr])) for _ in hypo_module.parameters()])
        
        self.sigma = nn.Parameter(torch.ones(2))
        self.sigma_outer = nn.Parameter(torch.ones(2))

        num_outputs = list(self.hypo_module.parameters())[-1].shape[0]


    def generate_params(self, context_x, context_y, num_meta_steps=None, **kwargs):
        meta_batch_size = context_x.shape[0]
        num_meta_steps = num_meta_steps if num_meta_steps != None else self.num_meta_steps
        
        with torch.enable_grad():
            adapted_parameters = OrderedDict()
            for name, param in self.hypo_module.meta_named_parameters():
                adapted_parameters[name] = param[None, ...].repeat((meta_batch_size,) + (1,) * len(param.shape))
            for j in range(num_meta_steps):
                context_x.requires_grad_()
                predictions = self.hypo_module(context_x, params=adapted_parameters)
                
                loss = self.loss(predictions, context_y, sigma=self.sigma)

                grads = torch.autograd.grad(loss, adapted_parameters.values(), allow_unused=False, create_graph=(True if (not self.first_order or j == num_meta_steps-1) else False))

                for i, ((name, param), grad) in enumerate(zip(adapted_parameters.items(), grads)):                    
                    if self.lr_type in ['static', 'global']:
                        lr = self.lr
                    elif self.lr_type in ['per_step']:
                        lr = self.lr[j]
                    elif self.lr_type in ['per_parameter']:
                        lr = self.lr[i][j] if num_meta_steps <= self.num_meta_steps else 1e-2
                    elif self.lr_type in ['simple_per_parameter']:
                        lr = self.lr[i]
                    else:
                        raise NotImplementedError
                    adapted_parameters[name] = param - lr * grad
                    # TODO: Add proximal regularization from iMAML
                    # Add meta-regularization
                
        return adapted_parameters
    
    def forward_with_params(self, query_x, fast_params, **kwargs):
        output = self.hypo_module(query_x, params=fast_params)
        return output

    def forward(self, meta_batch, **kwargs):
        context_x, context_y = meta_batch['context']
        query_x = meta_batch['query'][0]

        context_x = context_x.cuda()
        context_y = context_y.cuda()
        query_x = query_x.cuda()

        fast_params = self.generate_params(context_x, context_y)
        return self.forward_with_params(query_x, fast_params), fast_params

    
def hyperfan_out_init_H(m, out_features_main_net):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            var_H = 2/(out_features_main_net * m.in_features)
            std_H = np.sqrt(var_H)

            with torch.no_grad():
                m.weight.normal_(0., std_H)

        if hasattr(m, 'bias'):
            m.bias.data.fill_(0.0)


def hyperfan_out_init_G(m, in_features_main_net, out_features_main_net):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            var_G = max(2 * (1 - (in_features_main_net/out_features_main_net))/(m.out_features), 0)
            std_G = np.sqrt(var_G)

            with torch.no_grad():
                m.weight.normal_(0., std_G)
        if hasattr(m, 'bias'):
            m.bias.data.fill_(0.0)


class HyperNetwork(nn.Module):
    def __init__(self, hyper_in_features, hyper_hidden_layers, hyper_hidden_features, hypo_module, per_param=False):
        super().__init__()

        hypo_parameters = hypo_module.meta_named_parameters()

        self.names = []
        self.nets = nn.ModuleList()
        self.param_shapes = []
        for name, param in hypo_parameters:
            self.names.append(name)
            self.param_shapes.append(param.size())

            hn = FCBlock(hyper_hidden_features, hyper_hidden_layers, hyper_in_features,
                         int(torch.prod(torch.tensor(param.size()))), outermost_linear=True)
            with torch.no_grad():
                hn.net[-1].weight *= 1e-1
            self.nets.append(hn)

    def forward(self, z):
        params = OrderedDict()
        for name, net, param_shape in zip(self.names, self.nets, self.param_shapes):
            batch_param_shape = (-1,) + param_shape
            params[name] = net(z).reshape(batch_param_shape)

        return params


class FCLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.net(input)    


class FCBlock(nn.Module):
    def __init__(self,
                 hidden_ch,
                 num_hidden_layers,
                 in_features,
                 out_features,
                 outermost_linear=False):
        super().__init__()

        self.net = []
        self.net.append(FCLayer(in_features=in_features, out_features=hidden_ch))

        for i in range(num_hidden_layers):
            self.net.append(FCLayer(in_features=hidden_ch, out_features=hidden_ch))

        if outermost_linear:
            self.net.append(nn.Linear(in_features=hidden_ch, out_features=out_features))
        else:
            self.net.append(FCLayer(in_features=hidden_ch, out_features=out_features))

        self.net = nn.Sequential(*self.net)
        self.net.apply(init_weights_normal)

    def forward(self, input):
        return self.net(input)


class SDFHyperNetwork(nn.Module):
    '''
    Framework for swapping in different types of encoders and modules to use with
    hypernetworks.
    See Hypernetworks_MNIST for examples.
    '''

    def __init__(self, encoder, hypernetwork, hypo_module):
        super().__init__()
        self.encoder = encoder
        self.hypo_module = hypo_module
        self.hypernetwork = hypernetwork

    def forward(self, index, coords):
        z = self.encoder(index)
        batch_size = z.shape[0]
        z = z.reshape(batch_size, -1)
        params = self.hypernetwork(z)
        out = self.hypo_module.forward(coords, params)
        return out, z
        
    def freeze_hypernetwork(self):
        # Freeze hypernetwork for latent code optimization
        for param in self.hypernetwork.parameters():
            param.requires_grad = False

    def unfreeze_hypernetwork(self):
        # Unfreeze hypernetwork for training
        for param in self.hypernetwork.parameters():
            param.requires_grad = True

            
class AutoDecoder(nn.Module):
    '''
    Autodecoder module; takes an idx as input and returns a latent code, z
    '''

    def __init__(self, num_instances, latent_dim):
        super().__init__()
        self.latent_codes = nn.Embedding(num_instances, latent_dim)        
        torch.nn.init.normal_(self.latent_codes.weight.data, 0.0, 1e-3)

    def forward(self, idx, **kwargs):
        z = self.latent_codes(idx)
        return z
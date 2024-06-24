import os
from importlib import import_module
import torch
import torch.nn as nn
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        #import pdb; pdb.set_trace()
        print('Making model...') 
        self.args = args
        self.scale = args['scale']
        self.device = torch.device('cpu')
                
        module = import_module('model.' + args['model'].lower())
        self.model = module.make_model(args).to(self.device)
        self.load()
        
    def forward(self, x):                                    
        return self.model(x)

    def get_model(self):  
        return self.model
       
    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

 
    def load(self):
        #import pdb; pdb.set_trace()
        kwargs = {'map_location': lambda storage, loc: storage}
            
        model_epoch =  self.args['model_to_load']

        print(f'Loading model {model_epoch}')
        self.get_model().load_state_dict(
            torch.load(
                model_epoch,
                **kwargs
            ),
            strict=True
        )
        
    

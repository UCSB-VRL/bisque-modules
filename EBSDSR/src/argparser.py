import argparse
import torch

class Argparser:
    """
    The actual argparser

    """ 
    def __init__(self):
        self.args = self.prepare_arg_parser().parse_args()

    def prepare_arg_parser(self):
        """
        Add all args to the argparser     

        """
        
        arg_parser = argparse.ArgumentParser()
                   
        # Train, Val, Test DataSet specifications
         
        arg_parser.add_argument('--input_map', type=str, 
                                default='input_data/Ti64_LR.npy',
                                help=' input_lr_ebsd_map')
        arg_parser.add_argument('--n_colors', type=int, default=4,
                                help='number of channels to use')
        arg_parser.add_argument('--scale', type=int, default=4,
                                help='super resolution scale')
        

        # Models specificaitons
 
        arg_parser.add_argument('--model', default='han',
                                help='name of super-resolution model')        
        arg_parser.add_argument('--act', type=str, default='relu',
                                help='activation function')
        arg_parser.add_argument('--n_resblocks', type=int, default=20,
                                help='number of residual blocks')
        arg_parser.add_argument('--n_resgroups', type=int, default=10,
                                help='number of residual groups')
        arg_parser.add_argument('--reduction', type=int, default=16,
                                help='number of feature maps reduction')
        arg_parser.add_argument('--n_feats', type=int, default=128,
                                help='number of feature maps')
        arg_parser.add_argument('--res_scale', type=float, default=1,
                                help='residual scaling')
        # Parameters saving specificaitons

        arg_parser.add_argument('--save', type=str, default='output_data',
                                help='file name to save trained model')

        arg_parser.add_argument('--model_to_load', type=str, default='han_rotdist_symm_ti64',
                                help='file name to load')
                        
        #Training Parameters

        arg_parser.add_argument('--leak_value', type=float, default=0.2,
                                help='leak value in leaky relu')                                                  
        return arg_parser 
    

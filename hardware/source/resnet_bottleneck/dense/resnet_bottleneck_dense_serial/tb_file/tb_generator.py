import torch 
import torch.nn.functional as F
import numpy as np
import os
import math

import argparse


def is_power_of_two(n):
    return (math.ceil(math.log2(n)) == math.floor(math.log2(n)))

# This class generate the ground truth input/output and weights using pytorch for a residual block 
class test_bench_generator: 
    def __init__(self, out_dir, PI_FACTOR=[64, 16, 32, 32, 16, 64]):
        self.PI_FACTOR = PI_FACTOR
        self.out_dir = out_dir
        self.batch = 1
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

    def get_parameter(self, H, W, IC_0, OC_0, OC_3, OC_1):
        if not is_power_of_two(IC_0):
            raise Exception("IC_0 is not power of two")
        if not is_power_of_two(OC_0):
            raise Exception("OC_0 is not power of two")
        if not is_power_of_two(OC_3):
            raise Exception("OC_3 is not power of two")
        if not is_power_of_two(OC_1):
            raise Exception("OC_1 is not power of two")

        self.H = H
        self.W = W
        self.IC_0 = IC_0
        self.OC_0 = OC_0
        self.IC_3 = OC_0
        self.OC_3 = OC_3
        self.IC_1 = OC_3
        self.OC_1 = OC_1
        self.max_c = max(IC_0, OC_0, OC_3, OC_1)
        # use positive input and positive spatial mask weight to simulate the effect of spatial pruning
        self.input_feature = torch.randint(0, 127, size=(self.batch, self.IC_0, self.H, self.W)) 
        np.savetxt(os.path.join(self.out_dir, self.out_dir + '_dense_input.txt'), self.transform_feature(self.input_feature), fmt = '%d')

    #change the layeout of 3x3 weights
    def transform_w_3x3(self, w, PO=16, PI=16):
        w = w.clone().numpy().squeeze()
        w = np.transpose(w, (2, 3, 0, 1))
        w = np.ndarray.flatten(w)
        return w


    #change the layeout of 1x1 weights
    def transform_w_1x1(self, w, PI=64, PO=64):
        w = w.clone().numpy().squeeze()
        w = np.ndarray.flatten(w)
        return w


    #change the layeout of feature
    def transform_feature(self, feature):
        feature = feature.clone().numpy().squeeze()
        feature = np.transpose(feature, (1,2,0))
        feature = np.ndarray.flatten(feature)
        return feature

    #8-bit weights 
    def generate_weight(self): 
        self.w0 = torch.randint(-128, 127, size=(self.OC_0, self.IC_0, 1, 1))
        self.w3 = torch.randint(-128, 127, size=(self.OC_3, self.IC_3, 3, 3))
        self.w1 = torch.randint(-128, 127, size=(self.OC_1, self.IC_1, 1, 1))

        self.w0_dense, self.w3_dense, self.w1_dense  = self.w0.clone(), self.w3.clone(), self.w1.clone()
        
        np.savetxt(os.path.join(self.out_dir, self.out_dir + '_w0_dense_layout.txt'), self.transform_w_1x1(self.w0_dense, PI=self.PI_FACTOR[0], PO=self.PI_FACTOR[1]), fmt = '%d')
        np.savetxt(os.path.join(self.out_dir, self.out_dir + '_w3_dense_layout.txt'), self.transform_w_3x3(self.w3_dense, PI=self.PI_FACTOR[2], PO=self.PI_FACTOR[3]), fmt = '%d')
        np.savetxt(os.path.join(self.out_dir, self.out_dir + '_w1_dense_layout.txt'), self.transform_w_1x1(self.w1_dense, PI=self.PI_FACTOR[4], PO=self.PI_FACTOR[5]), fmt = '%d')

    #This is a simple quantize function for functional verfication, can change to other quantize method
    def quantize(self, feature): 
        feature = feature >> 10
        feature = torch.clamp(feature, min=-128, max=127)
        return feature

    def add_quantize(self, feature, identity): 
        feature = feature >> 10
        feature += identity
        feature = torch.clamp(feature, min=-128, max=127)
        return feature

    def dense_conv(self):
        # use torch conv function to produce ground truth
        identity = self.input_feature
        out0 = F.conv2d(self.input_feature, self.w0_dense) 
        out0 = self.quantize(out0)
        out0 = F.relu(out0)
        out3 = F.conv2d(out0, self.w3_dense, padding=1, stride=1)
        out3 = self.quantize(out3)
        out3 = F.relu(out3)
        out1 = F.conv2d(out3, self.w1_dense) 
        out1 = self.add_quantize(out1, identity)
        out1 = F.relu(out1)          
        
        np.savetxt(os.path.join(self.out_dir, self.out_dir + '_dense_out_0.txt'), self.transform_feature(out0), fmt = '%d')
        np.savetxt(os.path.join(self.out_dir, self.out_dir + '_dense_out_3.txt'), self.transform_feature(out3), fmt = '%d')
        np.savetxt(os.path.join(self.out_dir, self.out_dir + '_dense_out_1.txt'), self.transform_feature(out1), fmt = '%d')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Layer parameters')
    parser.add_argument('--Height', type=int, default=8, help='Height of input')
    parser.add_argument('--Width', type=int, default=8, help='Width of input')
    parser.add_argument('--IC_0', type=int, default=128, help='Number of input channel of first conv 1x1')
    parser.add_argument('--OC_0', type=int, default=128, help='Number of output channel of first conv 1x1')
    parser.add_argument('--OC_3', type=int, default=128, help='Number of output channel of conv 3x3')
    parser.add_argument('--OC_1', type=int, default=128, help='Number of output channel of last conv 1x1')
    args = parser.parse_args()

    output_dir = "TXT_FILES"

    tb_128 = test_bench_generator(output_dir, PI_FACTOR=[64, 16, 32, 32, 16, 64])
    tb_128.get_parameter(H=args.Height, W=args.Width, IC_0=args.IC_0, OC_0=args.OC_0, OC_3=args.OC_3, OC_1=args.OC_1)
    tb_128.generate_weight()
    tb_128.dense_conv()
    print("Generated output at:" + output_dir)

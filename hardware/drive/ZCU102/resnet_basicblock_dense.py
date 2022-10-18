import pynq
from pynq import Overlay 
from pynq import allocate
from pynq import Xlnk
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
import gc
import json
import os
import pickle

__all__ = ['resnet_basicblock_dense']


class resnet_basicblock_dense_drive:
    def __init__(self, bitfile):
        self.overlay = Overlay(bitfile)
        self.accel = self.overlay.top_0
        self.overlay.download()
        self.MAX_H = 256
        self.MAX_W = 256
        self.MAX_IC = 2048
        self.MAX_C = 512
        self.MAX_OC = 2048
        self.P_factor = [64, 16, 16, 64]
        self.CPRUNE_F = 64
        self.total_time = 0
        self.max_batch_size = 1
        self.in_buffer_size = 128 * 128 * 128 * self.max_batch_size
        self.out_buffer_size = 128 * 128 * 128 * self.max_batch_size
        self.w_buffer_size = self.MAX_IC * self.MAX_C + self.MAX_C * self.MAX_C * 9 + self.MAX_OC * self.MAX_C 
        self.w_buffer_size = int(2 ** math.ceil(math.log2(self.w_buffer_size)))
        self.mask_bits = 64 * 8
        if "serial" in bitfile:
            self.impl = "serial"
            self.P_factor = [32, 32, 32, 32]
        else:
            self.impl = "parallel"
            self.P_factor = [16, 16, 16, 16]
        
        self.in_feature_buffer = allocate(shape=(self.in_buffer_size), dtype=np.int8)
        self.out_feature_buffer = allocate(shape=(self.out_buffer_size), dtype=np.int8)
        self.w_buffer = allocate(shape=(self.w_buffer_size), dtype=np.int8)
        
        self.accel.register_map.fin_1 = self.in_feature_buffer.physical_address
        self.accel.register_map.fout_1 = self.out_feature_buffer.physical_address
        self.accel.register_map.fres_1 = self.in_feature_buffer.physical_address
        self.accel.register_map.weight_1 = self.w_buffer.physical_address
    

    def pack_weight(self, IC_0, OC_0, OC_3, OC_1, skip_0, skip_3, skip_1, first_layer=0):
        start = 0
        end = 0

        if first_layer == 1 and self.impl == "serial":
            size = 4 * 64 * 9
            end += size
            self.w_buffer[start: end] = np.random.randint(-128, 127, [size], dtype=np.int8)
            self.w_buffer.flush()
            start = end            
            
        if skip_0 == 0:
            size = IC_0 * OC_0 * 9
            end += size
            self.w_buffer[start: end] = np.random.randint(-128, 127, [size], dtype=np.int8)
            self.w_buffer.flush()
            start = end
            
        if skip_3 == 0:
            size = OC_0 * OC_3 * 9
            end += size
            self.w_buffer[start: end] = np.random.randint(-128, 127, [size], dtype=np.int8)
            self.w_buffer.flush()
            start = end
            
        if skip_1 == 0:
            size = OC_3 * OC_1
            end += size
            self.w_buffer[start: end] = np.random.randint(-128, 127, [size], dtype=np.int8)
            self.w_buffer.flush()
            start = end

    def pack_flags(self, *flags):
        flags_pack = int(0)
        for flag in flags:
            flags_pack = flags_pack << 1
            flags_pack = flags_pack + flag
        return flags_pack
    
    def ceil_channel(self, C, factor):
        return int(math.ceil(C / factor) * factor)

    def run(self, 
            H, 
            W, 
            IC_0, 
            OC_0, 
            OC_3, 
            OC_1, 
            stride2=0, 
            res=0, 
            skip_0=0, 
            skip_3=0, 
            skip_1=0, 
            relu_0=1, 
            relu_3=1, 
            relu_1=1, 
            batch=1, 
            first_layer=0):
        
        if hasattr(self, 'batch'):
            batch = self.batch
        if hasattr(self, 'num_test'):
            rep = self.num_test
        else:
            rep = 1
        


        if first_layer != 1:
            IC_0 = self.ceil_channel(IC_0, self.P_factor[0])
        OC_0 = self.ceil_channel(OC_0, self.P_factor[1])
        OC_3 = self.ceil_channel(OC_3, self.P_factor[2])
        OC_1 = self.ceil_channel(OC_1, self.P_factor[3])
        

        self.pack_weight(IC_0, OC_0, OC_3, OC_1, skip_0, skip_3, skip_1, first_layer=first_layer)
        flags = self.pack_flags(relu_1, relu_3, relu_0, first_layer, res, stride2, skip_1, skip_3, skip_0)
        
        
        self.out_feature_buffer[:] = np.zeros([self.out_buffer_size], dtype=np.int8)
        self.out_feature_buffer.flush()   
        
        info = "H:{} W:{} IC_0:{} OC_0:{} OC_3:{} OC_1:{} s2:{} r:{}  s0:{} s3:{} s1:{} flags:{}".format(
            H, W, IC_0, OC_0, OC_3, OC_1, stride2, res, skip_0, skip_3, skip_1, flags)
        
        self.accel.register_map.H = H
        self.accel.register_map.W = W
        self.accel.register_map.IC_0 = IC_0
        self.accel.register_map.OC_0 = OC_0
        self.accel.register_map.OC_3 = OC_3
        self.accel.register_map.OC_1 = OC_1
        self.accel.register_map.flags = flags
        self.accel.register_map.batch = batch
        
        total_time = 0
        for i in range(rep):
            idle = 0
            begin = time.time()
            self.accel.register_map.CTRL.AP_START = 1
            while idle == 0:
                idle = self.accel.register_map.CTRL.AP_IDLE  
            end = time.time()
            total_time += end - begin
        t = (total_time / rep)*1000
        self.total_time += t


class resnet_basicblock_dense(resnet_basicblock_dense_drive):
    def __init__(self, bitfile):
        super(resnet_basicblock_dense, self).__init__(bitfile)

    def resnet34(self, H, W, batch=1, num_of_test=100 ):
        self.num_test = num_of_test
        self.batch = 1
        self.log = []
        self.H = H
        self.W = W
        self.total_time = 0
        self.expansion = 1
        self.inplanes = 64
        self.base_width = 64
        layers = [3, 4, 6, 3]
        print("-------------------start-------------------")
        print("Model: Dense ResNet34 ")

        self.first_layer(self.H, self.W)
        self.H, self.W = self.H // 4, self.W // 4

        self.make_layer(64, layers[0], stride=1)
        self.make_layer(128, layers[1], stride=2)
        self.make_layer(256, layers[2], stride=2)
        self.last_block(layers[3])
        self.fc()

        print("Averaged latency:{} in {} times of test".format(self.total_time, self.num_test))
        print("-------------------end-------------------\n")
        
    def resnet18(self, H, W, num_of_test=100):
        self.num_test = num_of_test
        self.batch = 1
        self.H = H
        self.W = W

        self.total_time = 0
        self.expansion = 1
        self.inplanes = 64
        self.base_width = 64
        layers = [2, 2, 2, 2]
        print("-------------------start-------------------")
        print("Model: Dense ResNet18 ")
        self.first_layer(self.H, self.W)
        self.H, self.W = self.H // 4, self.W // 4

        self.make_layer(64, layers[0], stride=1)
        self.make_layer(128, layers[1], stride=2)
        self.make_layer(256, layers[2], stride=2)
        self.last_block(layers[3])
        self.fc()

        print("Averaged latency:{} in {} times of test".format(self.total_time, self.num_test))
        print("-------------------end-------------------\n")

    def make_layer(self, planes, blocks, stride):
        layer_count = 0 
        if stride != 1 or self.inplanes != planes * self.expansion:
            use_mask = 1 if (layer_count > 0 and self.share_mask) else 0
            self.basicblock(self.inplanes, planes, stride, self.base_width)
            layer_count += 1
        self.inplanes = planes * self.expansion

        for _ in range(1, blocks):
            self.basicblock(self.inplanes, planes, stride=1, base_width=self.base_width)
            layer_count += 1
    
    def basicblock(self, inplanes, planes, stride, base_width=64):
        width = int(planes * (base_width / 64.0))        

        IC = inplanes
        OC = planes * self.expansion
        if stride == 1:
            self.run(
                self.H, 
                self.W, 
                IC, 
                width, 
                OC, 
                OC, 
                stride2=0, 
                res=1,
                skip_1=1)
        else:
            # TODO: software dowsampling of feature
            self.run(
                self.H // 2, 
                self.W //2, 
                IC, 
                IC, 
                IC, 
                OC, 
                stride2=0, 
                res=0, 
                skip_0=1, 
                skip_3=1)
            self.run(
                self.H, 
                self.W, 
                IC, 
                width, 
                OC, 
                OC, 
                stride2=1, 
                res=1,
                skip_1=1)
            self.H, self.W = self.H // 2, self.W // 2

    
    def run_512(self, stride2=0):  # The last stage of ResNet exceed the maximum on-chip buffer size, so we split it into multiple kernel calls 
        if stride2 == 0:    
            self.run(self.H, self.W, 512, 256, 256, 256, stride2=0, skip_0=0, skip_3=1, skip_1=1, res=0)
            self.run(self.H, self.W, 512, 256, 256, 256, stride2=0, skip_0=0, skip_3=1, skip_1=1, res=0)
            self.run(self.H, self.W, 512, 512, 256, 256, stride2=0, skip_0=1, skip_3=0, skip_1=1, res=1)
            self.run(self.H, self.W, 512, 512, 256, 256, stride2=0, skip_0=1, skip_3=0, skip_1=1, res=1)
            
        if (stride2 == 1):

            self.run(self.H//2, self.W//2, 256, 256, 256, 256, skip_0=1, skip_3=1, skip_1=0, res=0)
            self.run(self.H//2, self.W//2, 256, 256, 256, 256, skip_0=1, skip_3=1, skip_1=0, res=0)
            self.run(self.H, self.W, 256, 256, 256, 256, skip_0=0, skip_3=1, skip_1=1, stride2=1, res=0)
            self.run(self.H, self.W, 256, 256, 256, 256, skip_0=0, skip_3=1, skip_1=1, stride2=1, res=0)
            self.H, self.W = self.H // 2, self.W// 2
            self.run(self.H, self.W, 512, 512, 256, 256, skip_0=1, skip_3=0, skip_1=1, stride2=0, res=1)
            self.run(self.H, self.W, 512, 512, 256, 256, skip_0=1, skip_3=0, skip_1=1, stride2=0, res=1)
              
            
    def last_block(self, n=3):
        self.run_512(stride2=1)
        for i in range(n - 1):
            self.run_512(stride2=0)

    def fc(self, num_class=1000, max_c3=512):
        for i in range(math.ceil(num_class / max_c3)):
            self.run(1, 1, 512, 512, max_c3, max_c3, skip_0=1, skip_3=1)


    def first_layer(self, H, W):
        if self.impl == "serial":
            self.run(H, W, 64, 64, 64, 64, stride2=1, first_layer=1, skip_0=1, skip_3=1, skip_1=1)
        else:
            self.run(H, W, 16, 64, 64, 64, stride2=1, first_layer=1, skip_0=0, skip_1=1)



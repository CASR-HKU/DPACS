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

__all__ = ['resnet_basicblock_sparse']

class resnet_basicblock_sparse_drive:
    def __init__(self, bitfile):
        self.overlay = Overlay(bitfile)
        self.accel = self.overlay.top_0
        self.overlay.download()
        self.MAX_H = 256
        self.MAX_W = 256
        self.MAX_IC = 2048
        self.MAX_C = 512
        self.MAX_OC = 2048
        
        self.CPRUNE_F = 32
        self.total_time = 0
        self.max_batch_size = 1
        self.in_buffer_size = 256 * 256 * 64 * self.max_batch_size
        self.out_buffer_size = 128 * 128 * 128 * self.max_batch_size
        self.w_buffer_size = self.MAX_IC * self.MAX_C + self.MAX_C * self.MAX_C * 9 + self.MAX_OC * self.MAX_C + self.MAX_C * 16 + self.MAX_H * self.MAX_W * self.max_batch_size / 32
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
        self.cmask_out_buffer = allocate(shape=(16), dtype=np.int32)
        self.smask_out_buffer = allocate(shape=(64*64), dtype=np.uint8)
        
        self.accel.register_map.fin_1 = self.in_feature_buffer.physical_address
        self.accel.register_map.fout_1 = self.out_feature_buffer.physical_address
        self.accel.register_map.fres_1 = self.in_feature_buffer.physical_address
        self.accel.register_map.weight_1 = self.w_buffer.physical_address
        self.accel.register_map.cmask_out_1 = self.cmask_out_buffer.physical_address
        self.accel.register_map.smask_out_1 = self.smask_out_buffer.physical_address

        
    def pack_mask(self, mask):
        mask_flattened = mask.flatten().astype(int)
        byte_size = math.ceil(mask_flattened.shape[0] / self.mask_bits) * (self.mask_bits // 8)
        mask_buffer = np.zeros(byte_size, dtype=np.int8)
        
        index = 0
        for i in range(math.ceil(mask_flattened.shape[0] / 8)):
            pack = 0
            for j in range(8):
                pack += mask_flattened[index] << j
                index += 1
                if (index == mask_flattened.shape[0]): break
            mask_buffer[i] = pack
        return mask_buffer

    def generate_stat_input(self, mask, H, W, IC):
        mask = mask.flatten().astype(int)
        count = 0
        index = 0
        for flag in mask:
            if flag == 1:
                self.in_feature_buffer[index * IC: (index + 1) * IC] = np.random.randint(0, 10, [IC], dtype=np.int8)
                self.in_feature_buffer.flush()
                count += 1
            else:
                self.in_feature_buffer[index * IC: (index + 1) * IC] = np.zeros([IC], dtype=np.int8)
                self.in_feature_buffer.flush()
            index += 1       
        self.in_feature_buffer.flush()      

    def pack_weight(self, IC_0, OC_0, OC_3, OC_1, next_c, mask, skip_0, skip_3, skip_1, first_layer, use_mask, use_cprune, enable_pool):
        start = 0
        end = 0
            
        if use_mask == 0:
            size = IC_0
            end += size
            self.w_buffer[start: end] = np.ones(size, dtype=np.uint8)
            self.w_buffer.flush()
            start = end
        
        if enable_pool == 1:
            size = int(OC_1 * next_c / self.CPRUNE_F)
            end += size
            self.w_buffer[start: end] = np.ones(size, dtype=np.uint8)
            self.w_buffer.flush()
            start = end           
        
        if first_layer == 1 and self.impl == "serial": 
            size = int(IC_0 * 4 * 9)
            end += size
            self.w_buffer[start: end] = np.ones(size, dtype=np.uint8)
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
            
        if use_mask == 1:
            mask_packed = self.pack_mask(mask)
            end += mask_packed.shape[0]
            self.w_buffer[start: end] = mask_packed
            self.w_buffer.flush()            
    
    def pack_flags(self, *flags):
        flags_pack = int(0)
        for flag in flags:
            flags_pack = flags_pack << 1
            flags_pack = flags_pack + flag
        return flags_pack
    
    def pack_cprune_list(self, cprune_list):
        pack = int(0)
        assert len(cprune_list) == 16
        cprune_list.reverse()
        for flag in cprune_list:
            pack = pack << 1
            pack = pack + flag 
        return pack
    
    def ceil_channel(self, C, factor):
        return int(math.ceil(C / factor) * factor)

    def generate_cprune(self, use_cprune, C, cprune_list=None, cp_ratio=None):
        if use_cprune == 0 and cprune_list == None:
            cprune_list = [1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1]
        elif use_cprune == 1 and cprune_list == None:
            c_grouped = C // self.CPRUNE_F
            nz_c = int(c_grouped * cp_ratio)
            cprune_list = np.zeros(16, dtype=np.int8)
            nz_c = np.random.choice(range(c_grouped), nz_c, replace=False)
            cprune_list[nz_c] = 1
            cprune_list = cprune_list.tolist()
        c_prune = self.pack_cprune_list(cprune_list)
        return c_prune

    def run(self, 
            H, 
            W, 
            IC_0, 
            OC_0, 
            OC_3, 
            OC_1, 
            next_c, 
            nz_ratio=1, 
            cp_ratio=1, 
            mask=None, 
            stride2=0, 
            res=0, 
            skip_0=0,
            skip_3=0, 
            skip_1=0, 
            relu_0=1,
            relu_3=1, 
            relu_1=1, 
            use_mask=0, 
            batch=1, 
            use_cprune=0, 
            enable_pool=0, 
            first_layer=0, 
            return_mask=1):
        
#         res=0
        
        if hasattr(self, 'batch'):
            batch = self.batch

        if hasattr(self, 'num_test'):
            rep = self.num_test
        else:
            rep = 1
            
        if mask is None:
            mask = np.random.choice([0, 1], size=[batch * H * W], p=[1 - nz_ratio, nz_ratio])  
        
        if first_layer != 1:
            IC_0 = self.ceil_channel(IC_0, self.P_factor[0])
        OC_0 = self.ceil_channel(OC_0, self.P_factor[1])
        OC_3 = self.ceil_channel(OC_3, self.P_factor[2])
        OC_1 = self.ceil_channel(OC_1, self.P_factor[3])
        
        if use_mask == 0 and first_layer == 0:
            self.generate_stat_input(mask, H, W, IC_0)

        c_prune_in = self.generate_cprune(use_cprune, OC_0, cp_ratio=cp_ratio)
        if OC_0 == OC_3:
            c_prune_out = c_prune_in
        else:
            c_prune_out = self.generate_cprune(use_cprune, OC_3, cp_ratio=cp_ratio)

        self.pack_weight(IC_0, OC_0, OC_3, OC_1, next_c, mask, skip_0, skip_3, skip_1, first_layer, use_mask, use_cprune, enable_pool)
        flags = self.pack_flags(relu_1, relu_3, relu_0, return_mask, first_layer, use_cprune, use_mask, res, stride2, enable_pool, skip_1, skip_3, skip_0)
        
        self.out_feature_buffer[:] = np.zeros([self.out_buffer_size], dtype=np.int8)
        self.out_feature_buffer.flush()   
        
        nz_count = np.count_nonzero(mask)
        sparsity = nz_count / mask.size
        info = "H:{} W:{} IC_0:{} OC_0:{} OC_3:{} OC_1:{} next_c:{} s2:{} r:{} sp:{} cp:{} s0:{} s3:{} s1:{} um:{} uc:{} pool:{} flags:{}".format(
            H, W, IC_0, OC_0, OC_3, OC_1, next_c, stride2, res, sparsity, cp_ratio, skip_0, skip_3, skip_1, use_mask, use_cprune, enable_pool, flags)


        self.accel.register_map.H = H
        self.accel.register_map.W = W
        self.accel.register_map.IC_0 = IC_0
        self.accel.register_map.OC_0 = OC_0
        self.accel.register_map.OC_3 = OC_3
        self.accel.register_map.OC_1 = OC_1
        self.accel.register_map.flags = flags
        self.accel.register_map.next_c = next_c
        self.accel.register_map.cmask_ic = c_prune_in
        self.accel.register_map.cmask_oc = c_prune_out
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





class resnet_basicblock_sparse(resnet_basicblock_sparse_drive):
    def __init__(self, bitfile):
        super(resnet_basicblock_sparse, self).__init__(bitfile)

    
    def resnet34(self, H, W, share_mask, s_ratio=1, c_ratio=1, num_of_test=100):
        self.num_test = num_of_test
        self.batch = 1
        self.H = H
        self.W = W
        self.share_mask = share_mask
        self.nz_ratio = s_ratio
        self.cp_ratio = c_ratio
        self.total_time = 0
        self.expansion = 1
        self.inplanes = 64
        self.base_width = 64
        layers = [3, 4, 6, 3]
        
        print("-------------------start-------------------")
        print("Model: ResNet34 with spatial ratio {} and channel ratio {}".format(s_ratio, c_ratio))

        self.first_layer(self.H, self.W) 

        self.H, self.W = self.H // 4, self.W // 4
        self.mask = np.random.choice([0, 1], size=[self.H * self.W], p=[1 - self.nz_ratio, self.nz_ratio])

        
        self.make_layer(64, layers[0], stride=1)
        self.make_layer(128, layers[1], stride=2)
        self.make_layer(256, layers[2], stride=2, use_cprune=1, enable_pool=1)
        self.last_block(layers[3])
        self.fc()
        print("Averaged latency:{} in {} times of test".format(self.total_time, self.num_test))
        print("-------------------end-------------------\n")
      

        
    def resnet18(self, H, W, share_mask, s_ratio=1, c_ratio=1, num_of_test=100):
        self.num_test = num_of_test
        self.batch = 1
        self.H = H
        self.W = W
        self.share_mask = share_mask
        self.nz_ratio = s_ratio
        self.cp_ratio = c_ratio
        self.total_time = 0
        self.expansion = 1
        self.inplanes = 64
        self.base_width = 64
        layers = [2, 2, 2, 2]
        
        print("-------------------start-------------------")
        print("Model: ResNet18 with spatial ratio {} and channel ratio {}".format(s_ratio, c_ratio))

        self.first_layer(self.H, self.W) 
        self.H, self.W = self.H // 4, self.W // 4
        self.mask = np.random.choice([0, 1], size=[self.H * self.W], p=[1 - self.nz_ratio, self.nz_ratio])

        
        self.make_layer(64, layers[0], stride=1)
        self.make_layer(128, layers[1], stride=2)
        self.make_layer(256, layers[2], stride=2, use_cprune=1, enable_pool=1)
        self.last_block(layers[3])
        self.fc()
        print("Averaged latency:{} in {} times of test".format(self.total_time, self.num_test))
        print("-------------------end-------------------\n")

 
    def make_layer(self, planes, blocks, stride, use_cprune=0, enable_pool=0):
        layer_count = 0 
        if stride != 1 or self.inplanes != planes * self.expansion:
            use_mask = 1 if (layer_count > 0 and self.share_mask) else 0
            self.basicblock(self.inplanes, planes, stride, self.base_width, use_mask=use_mask, use_cprune=use_cprune, enable_pool=enable_pool)
            layer_count += 1
        self.inplanes = planes * self.expansion

        for _ in range(1, blocks):
            use_mask = 1 if (layer_count > 0 and self.share_mask) else 0
            self.basicblock(self.inplanes, planes, stride=1, base_width=self.base_width, use_mask=use_mask, use_cprune=use_cprune, enable_pool=enable_pool)
            layer_count += 1
    
    def basicblock(self, inplanes, planes, stride, base_width=64, use_mask=0, use_cprune=0, enable_pool=0):
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
                next_c=width, 
                nz_ratio=self.nz_ratio, 
                cp_ratio=self.cp_ratio, 
                stride2=0, 
                res=1, 
                use_mask=use_mask, 
                use_cprune=use_cprune, 
                enable_pool=enable_pool,
                skip_1=1)
        else:
            mask_stride2 = np.zeros([self.batch, self.H, self.W], dtype=np.uint8)
            mask_stride2[:, ::2, ::2] = 1
            self.run(
                self.H, 
                self.W, 
                IC, 
                IC, 
                IC, 
                OC, 
                next_c=width, 
                mask=mask_stride2, 
                stride2=0, 
                res=0, 
                skip_0=1, 
                skip_3=1, 
                use_mask=1)

            self.run(
                self.H, 
                self.W, 
                IC, 
                width, 
                OC, 
                OC, 
                next_c=OC, 
                nz_ratio=self.nz_ratio, 
                cp_ratio=self.cp_ratio, 
                stride2=1, 
                res=1, 
                use_mask=use_mask, 
                use_cprune=use_cprune, 
                enable_pool=enable_pool,
                skip_1=1)
            
            self.H, self.W = self.H // 2, self.W // 2

    
    def run_512(self, stride2=0): # The last stage of ResNet exceed the maximum on-chip buffer size, so we split it into multiple kernel calls 
        if stride2 == 0:
            if self.share_mask:
                use_mask = 1
            else:
                use_mask = 0
            self.run(self.H, self.W, 512, 256, 256, 256, 256, stride2=0, skip_0=0, skip_3=1, skip_1=1, cp_ratio=self.cp_ratio, enable_pool=0, use_mask=use_mask, nz_ratio=self.nz_ratio, res=0)
            self.run(self.H, self.W, 512, 256, 256, 256, 256, stride2=0, skip_0=0, skip_3=1, skip_1=1, cp_ratio=self.cp_ratio, enable_pool=0, use_mask=use_mask, nz_ratio=self.nz_ratio, res=0)
            self.run(self.H, self.W, 512, 512, 256, 256, 256, stride2=0, skip_0=1, skip_3=0, skip_1=1, cp_ratio=self.cp_ratio, enable_pool=0, use_mask=use_mask, nz_ratio=self.nz_ratio, res=1)
            self.run(self.H, self.W, 512, 512, 256, 256, 256, stride2=0, skip_0=1, skip_3=0, skip_1=1, cp_ratio=self.cp_ratio, enable_pool=0, use_mask=use_mask, nz_ratio=self.nz_ratio, res=1)
        else:
            # TODO: solftware downsample of feature
            self.run(self.H//2, self.W//2, 256, 256, 256, 256, 256, skip_0=1, skip_1=1, cp_ratio=self.cp_ratio, enable_pool=0, use_mask=1, nz_ratio=self.nz_ratio, res=0)
            self.run(self.H//2, self.W//2, 256, 256, 256, 256, 256, skip_0=1, skip_1=1, cp_ratio=self.cp_ratio, enable_pool=0, use_mask=1, nz_ratio=self.nz_ratio, res=0)

            self.run(self.H, self.W, 256, 256, 256, 256, 256, skip_0=0, skip_3=1, skip_1=1, stride2=1, cp_ratio=self.cp_ratio, enable_pool=0, use_mask=0, nz_ratio=self.nz_ratio, res=0)
            self.run(self.H, self.W, 256, 256, 256, 256, 256, skip_0=0, skip_3=1, skip_1=1, stride2=1, cp_ratio=self.cp_ratio, enable_pool=0, use_mask=1, nz_ratio=self.nz_ratio, res=0)
            self.H, self.W = self.H // 2, self.W // 2 
            self.run(self.H, self.W, 512, 512, 256, 256, 256, skip_0=1, skip_3=0, skip_1=1, stride2=0, cp_ratio=self.cp_ratio, enable_pool=1, use_mask=1, nz_ratio=self.nz_ratio, res=1)
            self.run(self.H, self.W, 512, 512, 256, 256, 256, skip_0=1, skip_3=0, skip_1=1, stride2=0, cp_ratio=self.cp_ratio, enable_pool=1, use_mask=1, nz_ratio=self.nz_ratio, res=1)
                       
            
    def last_block(self, n=3):
        self.run_512(stride2=1)
        for i in range(n-1):
            self.run_512(stride2=0)

            
    def fc(self, num_class=1000, max_c=512):
        for i in range(math.ceil(num_class / max_c)):
            self.run(1, 1, 512, 512, 512, max_c, max_c, use_mask=1, skip_0=1, skip_3=1)

            
    def first_layer(self, H, W):
        if self.impl is 'parallel':
            self.run(H, W, 16, 16, 64, 64, 64, stride2=1, skip_0=1, skip_1=1, first_layer=1)
        else:
            self.run(H, W, 64, 64, 64, 64, 64, stride2=1, skip_0=1, skip_3=1, skip_1=1, first_layer=1)

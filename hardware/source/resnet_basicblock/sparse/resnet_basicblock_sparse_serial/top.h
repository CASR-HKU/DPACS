#define PRAGMA_SUB(x) _Pragma (#x)
#define DO_PRAGMA(x) PRAGMA_SUB(x)
// #include "/tools/Xilinx/Vitis_HLS/2020.2/include/gmp.h"
// #include "/tools/Xilinx/Vitis_HLS/2021.1/include/gmp.h"
#include "hls_stream.h"
#include "ap_int.h"
#include <iostream>

using namespace std;
#include "para.h"
#include "mem.h"
#include "mask.h"
#include "linebuffer.h"
#include "conv.h"
#include "conv_pack.h"
#include "c_prune.h"

void top(
    ap_int<FW * PI_0>  *fin,
    ap_int<FW * PO_1>  *fout,
    ap_int<FW * PO_1>  *fres,
    ap_int<W_FACTOR * WW>  *weight,
    ap_int<PW>  *cmask_out,
    ap_uint<MW> *out_mask,
    int  H,
    int  W,
    int  IC_0,
    int  OC_0,
    int  OC_3,
    int  OC_1,
    int  next_c,
    ap_uint<16> flags,
    ap_uint<16> cmask_ic,
    ap_uint<16> cmask_oc,
    int  batch
);

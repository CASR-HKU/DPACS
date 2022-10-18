#define PRAGMA_SUB(x) _Pragma (#x)
#define DO_PRAGMA(x) PRAGMA_SUB(x)
#include "hls_stream.h"
#include "ap_int.h"
#include <iostream>

using namespace std;
#include "para.h"
#include "mem.h"
#include "conv.h"
#include "conv_pack.h"


void top(
    ap_int<FW * PI_0>  *fin,
    ap_int<FW * PO_1>  *fout,
    ap_int<FW * PO_1>  *fres,
    ap_int<W_FACTOR * WW>  *weight,
    int  H,
    int  W,
    int  IC_0,
    int  OC_0,
    int  OC_3,
    int  OC_1,
    ap_uint<16> flags,
    int batch
);

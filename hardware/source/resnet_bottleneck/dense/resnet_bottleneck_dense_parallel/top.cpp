#include "top.h"

void wrapper(
    ap_int<FW * PI_0>  *fin,
    ap_int<FW * PO_1>  *fout,
    ap_int<FW * PO_1>  *fres,
    ap_int<W_FACTOR * WW>  *weight,
    int  Height,
    int  Width,
    int  IC_0,
    int  OC_0,
    int  OC_3,
    int  OC_1,
    ap_uint<16> flags,
    int batch
){
#pragma HLS DATAFLOW

    hls::stream<BundleT<PI_0, T_F> > fin_s;
#pragma HLS STREAM variable=fin_s depth=2
    hls::stream<BundleT<PI_0, T_F> > f_0_in;
#pragma HLS STREAM variable=f_0_in depth=2
    hls::stream<BundleT<PO_0, T_F> > f_0_out;
#pragma HLS STREAM variable=f_0_out depth=2

    hls::stream<BundleT<PI_3, T_F> > f_3_in;
#pragma HLS STREAM variable=f_3_in depth=2
    hls::stream<BundleT<PO_3, T_F> > f_3_out;
#pragma HLS STREAM variable=f_3_out depth=2

    hls::stream<BundleT<PI_1, T_F> > f_1_in;
#pragma HLS STREAM variable=f_1_in depth=2
    hls::stream<BundleT<PO_1, T_F> > f_1_out;
#pragma HLS STREAM variable=f_1_out depth=2
    hls::stream<BundleT<PO_1, T_F> > fres_s;
#pragma HLS STREAM variable=fres_s depth=2

    hls::stream<ap_int<WW * PI_0> > weight_s;
#pragma HLS STREAM variable=weight_s depth=2
    hls::stream<ap_int<WW * PI_0> > w_0_s;
#pragma HLS STREAM variable=w_0_s depth=2
    hls::stream<ap_int<WW * PI_3> > w_3_s;
#pragma HLS STREAM variable=w_3_s depth=2
    hls::stream<ap_int<WW * PI_1> > w_1_s;
#pragma HLS STREAM variable=w_1_s depth=2

    bool skip_0      = flags[0];  // 1 - skip first conv 1x1, 0 - compute first conv 1x1
    bool skip_3      = flags[1];  // 1 - skip conv 3x3, 0 - compute conv 3x3
    bool skip_1      = flags[2];  // 1 - skip last conv 1x1, 0 - compute last conv 1x1
    bool stride2     = flags[3];  // 1 - stride 2, 0 - stride 1
    bool residual    = flags[4];  // 1 - enable identity add, 0 - no identity add
    bool first_layer = flags[5];  // 1 - execute first layer, 0 - execute other layer
    bool relu_0      = flags[6];  // 1 - enable relu after first conv 1x1, 0 - no relu after first conv 1x1
    bool relu_3      = flags[7];  // 1 - enable relu after conv 3x3, 0 - no relu after conv 3x3
    bool relu_1      = flags[8];  // 1 - enable relu after last conv 1x1, 0 - no relu after last conv 1x1


    if (first_layer) {
        skip_0 = 1;
        skip_3 = 0;
        skip_1 = 1;
    }

    T_H out_width = stride2 ? Width >> 1: Width;
    T_H out_height = stride2 ? Height >> 1: Height;

    input_unit<PI_0, FW>(fin, f_0_in, Height, Width, IC_0, batch, first_layer);
    
    M2S_residual<PO_1, FW>(fres, fres_s, out_height, out_width, OC_1, residual, batch);

    Load_Weight_Merge<W_FACTOR>(
        weight,
        weight_s,
        IC_0,
        OC_0,
        OC_3,
        OC_1,
        skip_0,
        skip_3,
        skip_1
    );

    route_weight<W_FACTOR>(
        weight_s,
	    w_0_s,
	    w_3_s,
	    w_1_s,
        IC_0,
        OC_0,
        OC_3,
        OC_1,
	    skip_0,
	    skip_3,
	    skip_1
    );

    conv1x1_dsp_wrap<PI_0, PO_0, 2048, 512>
    (
        f_0_in,
        f_0_out,
        w_0_s,
        Height,
        Width,
        IC_0,
        OC_0,
        relu_0,
        skip_0,
        batch,
        first_layer
    );

    adjust_stream_same<PO_0, PI_3>
    (
        f_0_out,
        f_3_in,
        Height,
        Width,
        OC_0,
        batch
    );

    conv3x3_dsp_wrap<PI_3, PO_3, PO_1, 512, 256>
    (
        f_3_in,
        f_3_out,
        w_3_s,
        Height,
        Width,
        OC_0,
        OC_3,
        OC_1,
        stride2,
        relu_3, 
        residual,
        skip_3,
        batch,
        first_layer
    );


    adjust_stream_same<PO_3, PI_1>
    (
        f_3_out,
        f_1_in,
        out_height,
        out_width,
        OC_3,
        batch
    );


    conv1x1_dsp_residual<PI_1, PO_1, 512, 2048>
    (
        f_1_in,
        f_1_out,
        fres_s,
        w_1_s,
        out_height,
        out_width,
        OC_3,
        OC_1,
        relu_1, 
        residual,
        skip_1,
        batch
    );

    S2M_F<PO_1, FW>(f_1_out, fout, batch * out_height * out_width * OC_1 / PO_1);
}

/*
    This is the top function of the accelerator.
    args:
        fin: input feature
        fout: output feature
        fres: identity feature
        weight: weight
        H: height of input feature
        W: width of input feature
        IC_0: input channel of first conv 3x3 layer
        OC_0: output channel of first conv 3x3 layer
        OC_3: output channel of second conv 3x3 layer
        OC_1: output channel of conv 1x1 layer
        flags: control flags
        batch: batch size
*/

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
)
{

DO_PRAGMA(HLS INTERFACE m_axi port=fin bundle=gmem0 depth=65536)  
DO_PRAGMA(HLS INTERFACE m_axi port=fout bundle=gmem0 depth=65536) 
DO_PRAGMA(HLS INTERFACE m_axi port=fres bundle=gmem1 depth=65536)    
DO_PRAGMA(HLS INTERFACE m_axi port=weight   bundle=gmem2 depth=65536)

#pragma HLS INTERFACE s_axilite port=fin bundle=control 
#pragma HLS INTERFACE s_axilite port=fout bundle=control 
#pragma HLS INTERFACE s_axilite port=fres bundle=control 
#pragma HLS INTERFACE s_axilite port=weight bundle=control 
#pragma HLS INTERFACE s_axilite port=H bundle=control 
#pragma HLS INTERFACE s_axilite port=W bundle=control 
#pragma HLS INTERFACE s_axilite port=IC_0 bundle=control 
#pragma HLS INTERFACE s_axilite port=OC_0 bundle=control 
#pragma HLS INTERFACE s_axilite port=OC_3 bundle=control 
#pragma HLS INTERFACE s_axilite port=OC_1 bundle=control 
#pragma HLS INTERFACE s_axilite port=flags bundle=control 
#pragma HLS INTERFACE s_axilite port=batch bundle=control 
#pragma HLS INTERFACE s_axilite port=return bundle=control

    wrapper(
        fin, 
        fout, 
        fres, 
        weight, 
        H, 
        W,  
        IC_0, 
        OC_0, 
        OC_3, 
        OC_1, 
        flags,
        batch
    );
}


#include "top.h"


void wrapper(
    ap_int<FW * PI_0>  *fin,
    ap_int<FW * PO_1>  *fout,
    ap_int<FW * PO_1>  *fres,
    ap_int<W_FACTOR * WW>  *weight,
    ap_int<PW>  *cmask_out,
    ap_uint<MW> *smask_out,
    int  Height,
    int  Width,
    int  IC_0,
    int  OC_0,
    int  OC_3,
    int  OC_1,
    int  next_c,
    ap_uint<16> flags,
    ap_uint<16> cmask_ic,
    ap_uint<16> cmask_oc
){
#pragma HLS DATAFLOW

    hls::stream<BundleT<PI_0, T_F> > fin_s;
#pragma HLS STREAM variable=fin_s depth=16
    hls::stream<BundleT<PI_0, T_F> > f_first;
#pragma HLS STREAM variable=f_first depth=16
    hls::stream<BundleT<PI_0, T_F> > f_0_in;
#pragma HLS STREAM variable=f_0_in depth=16
    hls::stream<BundleT<PO_0, T_F> > f_0_out;
#pragma HLS STREAM variable=f_0_out depth=16

    hls::stream<BundleT<PI_3, T_F> > f_3_in;
#pragma HLS STREAM variable=f_3_in depth=16
    hls::stream<BundleT<PO_3, T_F> > f_3_out;
#pragma HLS STREAM variable=f_3_out depth=16

    hls::stream<BundleT<PI_1, T_F> > f_1_in;
#pragma HLS STREAM variable=f_1_in depth=16
    hls::stream<BundleT<PO_1, T_F> > f_1_out;
#pragma HLS STREAM variable=f_1_out depth=16
    hls::stream<BundleT<PO_1, T_F> > f_pool;
#pragma HLS STREAM variable=f_pool depth=16
    hls::stream<BundleT<PO_1, T_F> > fres_s;
#pragma HLS STREAM variable=fres_s depth=16

	hls::stream<T_K > key_first;
#pragma HLS STREAM variable=key_first depth=4
	hls::stream<T_K > in_key0;
#pragma HLS STREAM variable=in_key0 depth=4
	hls::stream<T_K > out_key0;
#pragma HLS STREAM variable=out_key0 depth=4
	hls::stream<T_K > in_key3;
#pragma HLS STREAM variable=in_key3 depth=4
	hls::stream<T_K > out_key3;
#pragma HLS STREAM variable=out_key3 depth=4
	hls::stream<T_K > in_key1;
#pragma HLS STREAM variable=in_key1 depth=4
	hls::stream<T_K > out_key1;
#pragma HLS STREAM variable=out_key1 depth=4
	hls::stream<T_K > key_pool;
#pragma HLS STREAM variable=key_pool depth=4
	hls::stream<T_K > key_mask;
#pragma HLS STREAM variable=key_mask depth=4

    hls::stream<ap_int<W_FACTOR * WW> > weight_s;
#pragma HLS STREAM variable=weight_s depth=4
    hls::stream<ap_int<P4 * WW> > w_first_s;
#pragma HLS STREAM variable=w_first_s depth=2
    hls::stream<ap_int<PI_0 * WW> > w_mask_s;
#pragma HLS STREAM variable=w_mask_s depth=2
    hls::stream<ap_int<PI_0 * WW> > w_0_s;
#pragma HLS STREAM variable=w_0_s depth=3
    hls::stream<ap_int<PI_3 * WW> > w_3_s;
#pragma HLS STREAM variable=w_3_s depth=5
    hls::stream<ap_int<PI_1 * WW> > w_1_s;
#pragma HLS STREAM variable=w_1_s depth=7
    hls::stream<ap_int<PO_1 * WW> > fc_s;
#pragma HLS STREAM variable=fc_s depth=9

    hls::stream<T_MASK > mask_s;
#pragma HLS STREAM variable=mask_s depth=2

    
    bool skip_0      = flags[0];  // 1 - bypass first conv 3x3, 0 - compute first conv 3x3
    bool skip_3      = flags[1];  // 1 - bypass second conv 3x3, 0 - compute second conv 3x3 
    bool skip_1      = flags[2];  // 1 - bypass conv 1x1 for downsample branch, 0 - compute conv 1x1 for downsample branch
    bool enable_pool = flags[3];  // 1 - predict channel pruning mask for next block, 0 - bypass channel mask prediction
    bool stride2     = flags[4];  // 1 - stride size equal to 2, 0 - stride size is 1
    bool residual    = flags[5];  // 1 - use identity add in residual block, 0 - bypass identity add
    bool use_mask    = flags[6];  // 1 - use pre-computed spatial mask, 0 - use spatial mask unit to compute spatial mask
    bool use_cprune  = flags[7];  // 1 - do channel pruning, 0 - do full channel inference
    bool first_layer = flags[8];  // 1 - execute the first conv layer, 0 - execute rest of the layer
    bool return_mask = flags[9];  // 1 - output spatial mask, 0 - do not output spatial mask
    bool relu_0      = flags[10]; // 1 - use relu activation on first conv 3x3 layer, 0 - no activation on first conv 3x3 
    bool relu_3      = flags[11]; // 1 - use relu activation on conv 3x3 layer, 0 - no activation on conv 3x3 
    bool relu_1      = flags[12]; // 1 - use relu activation on last conv 1x1 layer, 0 - no activation on last conv 1x1 



    if (first_layer) {
        skip_0 = 0;
        skip_3 = 1;
        skip_1 = 1;
        IC_0 = PI_0;
        OC_0 = 64;
        OC_3 = 64;
        OC_1 = 64;
        stride2 = 1;
    }

    Load_Weight_Wrap<W_FACTOR>(
        weight,
        weight_s,
        IC_0,
        OC_0,
        OC_3,
        OC_1,
        next_c,
        Height,
        Width,
        skip_0,
        skip_3,
        skip_1,
        use_mask,
        enable_pool,
        cmask_ic,
        cmask_oc,
        use_cprune,
        first_layer       
    );


    if(use_cprune){
        ap_uint<4> nz_channel_0 = 0;
        for (ap_uint<4> i = 0; i < 8; i++){
        #pragma HLS UNROLL
            nz_channel_0 += cmask_ic[i];
        }
        T_C pruned_OC_0 = nz_channel_0 * CPRUNE_FACTOR;

        OC_0 = pruned_OC_0;
    }



    route_weight<W_FACTOR>(
        weight_s,
	    w_mask_s,
	    w_0_s,
	    w_3_s,
	    w_1_s,
        fc_s,
        mask_s,
        IC_0,
        OC_0,
        OC_3,
        OC_1,
        next_c,
	    Height,
	    Width,
	    skip_0,
	    skip_3,
	    skip_1,
	    use_mask,
        enable_pool,
        first_layer
    );

    input_unit<PI_0, FW>(
        mask_s,
        w_mask_s,
        fin,
        in_key0,
        f_0_in,
	    Height,
	    Width,
	    IC_0,     
        use_mask,
        first_layer   
    );


    conv3x3_dsp_wrap<PI_3, PO_3, 512, 256>
    (
        f_0_in,
        f_0_out,
        in_key0,
        out_key0,
        w_0_s,
        Height,
        Width,
        IC_0,
        OC_0,
        stride2,
        relu_0, 
        skip_0,
        use_cprune,
        first_layer
    );

    adjust_stream_same<PO_0, PI_3>
    (
        f_0_out,
        f_3_in,
        out_key0,
        in_key3,
        OC_0
    );


    T_H out_width = stride2 ? Width >> 1: Width;
    T_H out_height = stride2 ? Height >> 1: Height;


    conv3x3_dsp_res_wrap<PI_3, PO_3, 512, 256>
    (
        f_3_in,
        f_3_out,
        in_key3,
        out_key3,
        fres,
        w_3_s,
        out_height,
        out_width,
        OC_0,
        OC_3,
        relu_3,
        residual,
        skip_3,
        use_cprune
    );


    adjust_stream_same<PO_3, PI_1>
    (
        f_3_out,
        f_1_in,
        out_key3,
        in_key1,
        OC_3
    );

    conv1x1_dsp_wrap<PI_1, PO_1, 512, 512>
    (
        f_1_in,
        f_1_out,
        in_key1,
        out_key1,
        w_1_s,
        OC_3,
        OC_1,
        relu_1,
        skip_1,
        use_cprune
    );



    S2M_key<PO_1, FW>
    (
        f_1_out, 
        f_pool, 
        fout, 
        smask_out,
        out_key1, 
        key_pool, 
        out_height, 
        out_width, 
        OC_1,
        enable_pool, 
        first_layer,
        return_mask
    );
    
    max_pool<PO_1, 2048, 512>
    (
        f_pool, 
        key_pool, 
        fc_s, 
        cmask_out, 
        enable_pool,
        OC_1, 
        next_c
    );


}

/*
    This is the top function of the accelerator.
    args:
        fin: input feature
        fout: output feature
        fres: identity feature
        weight: weight
        cmask_out: output channel mask
        H: height of input feature
        W: width of input feature
        IC_0: input channel of first conv 1x1 layer
        OC_0: output channel of first conv 1x1 layer
        OC_3: output channel of conv 3x3 layer
        OC_1: output channel of last conv 1x1 layer
        next_c: next block's channel size for pruning
        flags: control flags
        cmask_ic: channel mask for this block
        cmask_oc: channel mask for this block
        batch: batch size (not implementated yet)
*/
void top(
    ap_int<FW * PI_0>  *fin,
    ap_int<FW * PO_1>  *fout,
    ap_int<FW * PO_1>  *fres,
    ap_int<W_FACTOR * WW>  *weight,
    ap_int<PW>  *cmask_out,
    ap_uint<MW> *smask_out,
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
)
{

#pragma HLS INTERFACE m_axi port=fin bundle=gmem0 depth=65536 
#pragma HLS INTERFACE m_axi port=fout bundle=gmem0 depth=65536 
#pragma HLS INTERFACE m_axi port=fres bundle=gmem1 depth=65536    
#pragma HLS INTERFACE m_axi port=weight bundle=gmem2 depth=65536 
#pragma HLS INTERFACE m_axi port=smask_out bundle=gmem2 depth=65536 
#pragma HLS INTERFACE m_axi port=cmask_out bundle=gmem1 depth=65536 

#pragma HLS INTERFACE s_axilite port=fin bundle=control 
#pragma HLS INTERFACE s_axilite port=fout bundle=control 
#pragma HLS INTERFACE s_axilite port=fres bundle=control 
#pragma HLS INTERFACE s_axilite port=weight bundle=control 
#pragma HLS INTERFACE s_axilite port=smask_out bundle=control 
#pragma HLS INTERFACE s_axilite port=H bundle=control 
#pragma HLS INTERFACE s_axilite port=W bundle=control 
#pragma HLS INTERFACE s_axilite port=IC_0 bundle=control 
#pragma HLS INTERFACE s_axilite port=OC_0 bundle=control 
#pragma HLS INTERFACE s_axilite port=OC_3 bundle=control 
#pragma HLS INTERFACE s_axilite port=OC_1 bundle=control 
#pragma HLS INTERFACE s_axilite port=next_c bundle=control 
#pragma HLS INTERFACE s_axilite port=flags bundle=control 
#pragma HLS INTERFACE s_axilite port=batch bundle=control 
#pragma HLS INTERFACE s_axilite port=cmask_out bundle=control 
#pragma HLS INTERFACE s_axilite port=cmask_ic bundle=control 
#pragma HLS INTERFACE s_axilite port=cmask_oc bundle=control 
#pragma HLS INTERFACE s_axilite port=return bundle=control

    wrapper(
        fin, 
        fout, 
        fres, 
        weight, 
        cmask_out,
        smask_out,
        H, 
        W,  
        IC_0, 
        OC_0, 
        OC_3, 
        OC_1, 
        next_c,
        flags,
        cmask_ic,
        cmask_oc
    );
}


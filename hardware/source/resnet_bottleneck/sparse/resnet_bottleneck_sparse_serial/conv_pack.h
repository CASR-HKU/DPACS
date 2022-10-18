template<int PI, int PO, int PF, int M_OC>
void first_layer_unit(
		hls::stream<BundleT<PI, T_F> > &fmap_in,
		hls::stream<BundleT<PO, T_F> > &fmap_out,
		// hls::stream<T_K > &in_key,
		hls::stream<T_K > &out_key,
		hls::stream<ap_int<P4 * WW> > &w,
		T_H  Height,
        T_H  Width,
		T_C  IC,
		T_C  OC,
		bool stride2,
		bool relu
){

#pragma HLS DATAFLOW	
	hls::stream<BundleT<9, ap_int<P4 * FW> > > fmap_win;
#pragma HLS STREAM variable=fmap_win depth=2
	hls::stream<BundleT<PF, T_P> > s_conv3;
#pragma HLS STREAM variable=s_conv3 depth=2
	hls::stream<BundleT<PF, T_F> > fmap_q;
#pragma HLS STREAM variable=fmap_q depth=2
	hls::stream<T_K > key_0;
#pragma HLS STREAM variable=key_0 depth=4
	hls::stream<T_K > key_1;
#pragma HLS STREAM variable=key_1 depth=4
	hls::stream<T_K > key_q;
#pragma HLS STREAM variable=key_q depth=4
	T_Q q_buffer[MAX_C];
// DO_PRAGMA(HLS ARRAY_PARTITION variable=q_buffer cyclic factor=PO3/2)


	T_H out_height, out_width;
	out_height = Height / 2;
	out_width = Width / 2;


	line_buffer_first_layer<PI>(
		fmap_in,
		fmap_win,
		// in_key,
		// key_0,
		Height,
		Width
	);


	first_layer<PI, PO, PF, M_OC>(
		fmap_win,
		s_conv3,
		// key_0,		
		key_1,
		w,
		out_height,
		out_width
		// IC,
		// OC
	);

	quantize_shift<PF>(s_conv3, fmap_q, key_1, key_q, q_buffer, OC, relu);
		
	
	adjust_stream_larger<PF, PO>(
		fmap_q,
		fmap_out,
		key_q,
		out_key,
		OC
	);

}



template<int PI, int PO, int PF, int M_OC>
void first_layer_wrap(
		hls::stream<BundleT<PI, T_F> > &fmap_in,
		hls::stream<BundleT<PO, T_F> > &fmap_out,
		hls::stream<T_K > &in_key,
		hls::stream<T_K > &out_key,
		hls::stream<ap_int<P4 * WW> > &w,
		T_H  Height,
        T_H  Width,
		T_C  IC,
		T_C  OC,
		bool stride2,
		bool relu,
		bool first_layer
){
	if(first_layer){
		first_layer_unit<PI, PO, PF, M_OC>(
			fmap_in,
			fmap_out,
			// in_key,
			out_key,
			w,
			Height,
			Width,
			IC,
			OC,
			stride2,
			relu
		);
	}
	else{
		adjust_stream_same<PI, PO>(
			fmap_in,
			fmap_out,
			in_key,
			out_key,
			OC
		);
	}
}



template<int PI, int PO3, int PO1, int M_IC, int M_OC>
void conv3x3_dsp(
		hls::stream<BundleT<PI, T_F> > &fmap_in,
		hls::stream<BundleT<PO3, T_F> > &fmap_out,
		hls::stream<T_K > &in_key,
		hls::stream<T_K > &out_key,
		ap_int<FW * PO1>  *fmap_res,
		hls::stream<BundleT<PO1, T_F> > &res_stream,
		hls::stream<ap_int<PI * WW> > &w,
		T_H  Height,
        T_H  Width,
		T_C  IC,
		T_C  OC,
		T_C  OC_1,
		bool stride2,
		bool relu,
		bool residual,
		bool use_cprune
){
#pragma HLS Dataflow
	
	hls::stream<BundleT<PO3, T_P> > s_conv3;
#pragma HLS STREAM variable=s_conv3 depth=2

	hls::stream<BundleT<9, ap_int<PI * FW> > > fmap_win;
#pragma HLS STREAM variable=fmap_win depth=2 

	hls::stream<BundleT<PI, T_F> > fmap_serial;
#pragma HLS STREAM variable=fmap_serial depth=2 


	hls::stream<T_K > key_0;
#pragma HLS STREAM variable=key_0 depth=2

	hls::stream<T_K > key_1;
#pragma HLS STREAM variable=key_1 depth=2

	hls::stream<T_K > key_conv;
#pragma HLS STREAM variable=key_conv depth=4

	hls::stream<T_K > key_res_in;
DO_PRAGMA(HLS STREAM variable=key_res_in depth=MAX_C/PO1)


	hls::stream<T_OFFSET> offset_s;
#pragma HLS STREAM variable=key_conv depth=4


	T_Q q_buffer[MAX_C];
DO_PRAGMA(HLS ARRAY_PARTITION variable=q_buffer cyclic factor=PO3/2)
   
    T_H out_height = stride2 ? Height >> 1: Height;
    T_H out_width = stride2 ? Width >> 1: Width;

	// M2M(q, q_buffer, IC);

	conv_3x3_line_buffer_residual_stride_serial<PI, (64 * 128 / PI)>(
		fmap_in, 
		fmap_serial,
		offset_s,
		in_key, 
		key_0, 
		key_res_in, 
		Height, 
		Width, 
		IC, 
		stride2, 
		residual
	);


	Residual_read<PO1, FW>(fmap_res, key_res_in, res_stream, out_height, out_width, OC_1, residual);
	
	conv_3x3_double_serial<PI, PO3, M_IC, M_OC>(
		fmap_serial, 
		s_conv3,
		offset_s, 
		key_0,
		key_1,
		w, 
		IC, 
		OC, 
		use_cprune
	);

	quantize_shift<PO3>(s_conv3, fmap_out, key_1, out_key, q_buffer, OC, relu);
}


template<int PI, int PO3, int PO1, int M_IC, int M_OC>
void conv3x3_dsp_wrap(
		hls::stream<BundleT<PI, T_F> > &fmap_in,
		hls::stream<BundleT<PO3, T_F> > &fmap_out,
		hls::stream<T_K > &in_key,
		hls::stream<T_K > &out_key,
		ap_int<FW * PO1>  *fmap_res,
		hls::stream<BundleT<PO1, T_F> > &res_stream,
		hls::stream<ap_int<PI * WW> > &w,
		T_H  Height,
        T_H  Width,
		T_C  IC,
		T_C  OC,
		T_C  OC_1,
		bool stride2,
		bool relu,
		bool residual,
		bool skip,
		bool use_cprune
){
	if(skip){
		adjust_stream_same<PI, PO3>(
			fmap_in,
			fmap_out,
			in_key,
			out_key,
			IC
		);
	}
	else{
		conv3x3_dsp<PI, PO3, PO1, M_IC, M_OC>(
			fmap_in,
			fmap_out,
			in_key,
			out_key,
			fmap_res,
			res_stream,
			w,
			Height,
			Width,
			IC,
			OC,
			OC_1,
			stride2,
			relu,
			residual,
			use_cprune
		);
	}
}



template<int PI, int PO, int M_IC, int M_OC>
void conv1x1_dsp_res_unit(
		hls::stream<BundleT<PI, T_F> > &fmap_in,
		hls::stream<BundleT<PO, T_F> > &fmap_out,
		hls::stream<T_K > &in_key,
		hls::stream<T_K > &out_key,
		hls::stream<BundleT<PO, T_F> > &res_stream,
		hls::stream<ap_int<PI * WW> > &w,
		// T_Q  *q,
		int Width,
		T_C  IC,
		T_C  OC,
		bool relu,
		bool residual,
		bool use_cprune
){
#pragma HLS Dataflow

	hls::stream<BundleT<PO, T_P> > s_conv1;
#pragma HLS STREAM variable=s_conv1 depth=2

	hls::stream<T_K > key_s;
#pragma HLS STREAM variable=key_s depth=2

	static T_Q q_buffer[MAX_OC];
DO_PRAGMA( HLS ARRAY_PARTITION variable=q_buffer cyclic factor=PO/2)


	conv1x1_dsp_double<PI, PO, M_IC, M_OC>(fmap_in, s_conv1, in_key, key_s, w, IC, OC, use_cprune);
	quantize_shift_res<PO>(s_conv1, fmap_out, res_stream, key_s, out_key, q_buffer, OC, relu, residual);
}


template<int PI, int PO, int M_IC, int M_OC>
void conv1x1_dsp_residual(
		hls::stream<BundleT<PI, T_F> > &fmap_in,
		hls::stream<BundleT<PO, T_F> > &fmap_out,
		hls::stream<T_K > &in_key,
		hls::stream<T_K > &out_key,
		hls::stream<BundleT<PO, T_F> > &res_stream,
		hls::stream<ap_int<PI * WW> > &w,
		// T_Q  *q,
		int Width,
		T_C  IC,
		T_C  OC,
		bool relu,
		bool residual,
		bool skip,
		bool use_cprune
){
	if(skip){
		adjust_stream_larger<PI, PO>(
			fmap_in,
			fmap_out,
			in_key,
			out_key,
			IC
		);
	}
	else{
		conv1x1_dsp_res_unit<PI, PO, M_IC, M_OC>(
			fmap_in,
			fmap_out,
			in_key,
			out_key,
			res_stream,
			w,
			Width,
			IC,
			OC,
			relu,
			residual,
			use_cprune
		);
	}
}


template<int PI, int PO, int M_IC, int M_OC>
void conv1x1_dsp_unit(
		hls::stream<BundleT<PI, T_F> > &fmap_in,
		hls::stream<BundleT<PO, T_F> > &fmap_out,
		hls::stream<T_K > &in_key,
		hls::stream<T_K > &out_key,
		hls::stream<ap_int<PI * WW> > &w,
		// T_Q  *q,
		T_C  IC,
		T_C  OC,
		bool relu,
		bool use_cprune
){
#pragma HLS Dataflow

	hls::stream<BundleT<PO, T_P> > s_conv1;
#pragma HLS STREAM variable=s_conv1 depth=2

	hls::stream<T_K > key_s;
#pragma HLS STREAM variable=key_s depth=2

	static T_Q q_buffer[MAX_OC];
DO_PRAGMA( HLS ARRAY_PARTITION variable=q_buffer cyclic factor=PO/2)

	// M2M(q, q_buffer, OC);
	conv1x1_dsp_double<PI, PO, M_IC, M_OC>(fmap_in, s_conv1, in_key, key_s, w, IC, OC, use_cprune);
	quantize_shift<PO>(s_conv1, fmap_out, key_s, out_key, q_buffer, OC, relu);
}

template<int PI, int PO, int M_IC, int M_OC>
void conv1x1_dsp_wrap(
		hls::stream<BundleT<PI, T_F> > &fmap_in,
		hls::stream<BundleT<PO, T_F> > &fmap_out,
		hls::stream<T_K > &in_key,
		hls::stream<T_K > &out_key,
		hls::stream<ap_int<PI * WW> > &w,
		// T_Q  *q,
		T_C  IC,
		T_C  OC,
		bool relu,
		bool skip,
		bool use_cprune
){


	if(skip){
		adjust_stream_smaller<PI, PO>(
			fmap_in,
			fmap_out,
			in_key,
			out_key,
			IC
		);
	}
	else{
		conv1x1_dsp_unit<PI, PO, M_IC, M_OC>(
			fmap_in,
			fmap_out,
			in_key,
			out_key,
			w,
			IC,
			OC,
			relu,
			use_cprune		
		);
	}

}
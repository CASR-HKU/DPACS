template<int PI, int PO3, int M_IC, int M_OC>
void conv3x3_dsp_res(
		hls::stream<BundleT<PI, T_F> > &fmap_in,
		hls::stream<BundleT<PO3, T_F> > &fmap_out,
		hls::stream<BundleT<PO3, T_F> > &fres,
		hls::stream<ap_int<PI * WW> > &w,
		T_H  Height,
        T_H  Width,
		T_C  IC,
		T_C  OC,
		bool relu,
		bool residual,
		T_BATCH batch
){
#pragma HLS Dataflow
	
	hls::stream<BundleT<PO3, T_P> > s_conv3;
#pragma HLS STREAM variable=s_conv3 depth=16

	hls::stream<BundleT<9, ap_int<PI * FW> > > fmap_win;
#pragma HLS STREAM variable=fmap_win depth=2 

	T_Q q_buffer[MAX_C];
DO_PRAGMA(HLS ARRAY_PARTITION variable=q_buffer cyclic factor=PO3/2)
   
	const bool stride2 = 0;

	conv_3x3_line_buffer<PI, 1024>(fmap_in, fmap_win, Height, Width, IC, batch);
	
	conv_3x3_double<PI, PO3, M_IC, M_OC>(fmap_win, s_conv3, w, Height, Width, IC, OC, stride2, batch);

	T_H out_width = stride2 ? Width >> 1: Width;
    T_H out_height = stride2 ? Height >> 1: Height;

	quantize_shift_res<PO3>(s_conv3, fres, fmap_out, q_buffer, batch * out_height * out_width, OC, relu, residual);
}


template<int PI, int PO3, int M_IC, int M_OC>
void conv3x3_dsp_res_wrap(
		hls::stream<BundleT<PI, T_F> > &fmap_in,
		hls::stream<BundleT<PO3, T_F> > &fmap_out,
		hls::stream<BundleT<PO3, T_F> > &fres,
		hls::stream<ap_int<PI * WW> > &w,
		T_H  Height,
        T_H  Width,
		T_C  IC,
		T_C  OC,
		bool relu,
		bool residual,
		bool skip,
		T_BATCH batch
){
	if(skip){
		adjust_stream_same<PI, PO3>(
			fmap_in,
			fmap_out,
			Height,
			Width,
			IC,
			batch
		);
	}
	else{
		conv3x3_dsp_res<PI, PO3,  M_IC, M_OC>(
			fmap_in,
			fmap_out,
			fres,
			w,
			Height,
			Width,
			IC,
			OC,
			relu,
			residual,
			batch
		);
	}
}



template<int PI, int PO3,  int M_IC, int M_OC>
void conv3x3_dsp(
		hls::stream<BundleT<PI, T_F> > &fmap_in,
		hls::stream<BundleT<PO3, T_F> > &fmap_out,
		hls::stream<ap_int<PI * WW> > &w,
		T_H  Height,
        T_H  Width,
		T_C  IC,
		T_C  OC,
		bool stride2,
		bool relu,
		T_BATCH batch,
		bool first_layer
){
#pragma HLS Dataflow
	
	hls::stream<BundleT<PO3, T_P> > s_conv3;
#pragma HLS STREAM variable=s_conv3 depth=16

	hls::stream<BundleT<9, ap_int<PI * FW> > > fmap_win;
#pragma HLS STREAM variable=fmap_win depth=2 

	T_Q q_buffer[MAX_C];
DO_PRAGMA(HLS ARRAY_PARTITION variable=q_buffer cyclic factor=PO3/2)
   

	conv_3x3_line_buffer_wrap<PI, 1024>(fmap_in, fmap_win, Height, Width, IC, stride2, batch, first_layer);


	conv_3x3_double<PI, PO3, M_IC, M_OC>(fmap_win, s_conv3, w, Height, Width, IC, OC, stride2, batch);

	T_H out_width = stride2 ? Width >> 1: Width;
    T_H out_height = stride2 ? Height >> 1: Height;

	quantize_shift<PO3>(s_conv3, fmap_out, q_buffer, batch * out_height * out_width, OC, relu);
}


template<int PI, int PO3,  int M_IC, int M_OC>
void conv3x3_dsp_wrap(
		hls::stream<BundleT<PI, T_F> > &fmap_in,
		hls::stream<BundleT<PO3, T_F> > &fmap_out,
		hls::stream<ap_int<PI * WW> > &w,
		T_H  Height,
        T_H  Width,
		T_C  IC,
		T_C  OC,
		bool stride2,
		bool relu,
		bool skip,
		T_BATCH batch,
		bool first_layer
){
	if(skip){
		adjust_stream_same<PI, PO3>(
			fmap_in,
			fmap_out,
			Height,
			Width,
			IC,
			batch
		);
	}
	else{
		conv3x3_dsp<PI, PO3, M_IC, M_OC>(
			fmap_in,
			fmap_out,
			w,
			Height,
			Width,
			IC,
			OC,
			stride2,
			relu,
			batch,
			first_layer
		);
	}
}


template<int PI, int PO, int M_IC, int M_OC>
void conv1x1_dsp_unit(
		hls::stream<BundleT<PI, T_F> > &fmap_in,
		hls::stream<BundleT<PO, T_F> > &fmap_out,
		hls::stream<ap_int<PI * WW> > &w,
		T_H  Height,
        T_H  Width,
		T_C  IC,
		T_C  OC,
		bool relu,
		T_BATCH batch
){
#pragma HLS Dataflow

	hls::stream<BundleT<PO, T_P> > s_conv1;
#pragma HLS STREAM variable=s_conv1 depth=2

	hls::stream<T_K > key_s;
#pragma HLS STREAM variable=key_s depth=2

	static T_Q q_buffer[MAX_OC];
DO_PRAGMA( HLS ARRAY_PARTITION variable=q_buffer cyclic factor=PO/2)

	conv1x1_dsp_double<PI, PO, M_IC, M_OC>(fmap_in, s_conv1, w, batch * Height * Width, IC, OC);
	quantize_shift<PO>(s_conv1, fmap_out, q_buffer, batch * Height * Width, OC, relu);
}

template<int PI, int PO, int M_IC, int M_OC>
void conv1x1_dsp_wrap(
		hls::stream<BundleT<PI, T_F> > &fmap_in,
		hls::stream<BundleT<PO, T_F> > &fmap_out,
		hls::stream<ap_int<PI * WW> > &w,
		T_H  Height,
        T_H  Width,
		T_C  IC,
		T_C  OC,
		bool relu,
		bool skip,
		T_BATCH batch
){

	if(skip){
		adjust_stream_same<PI, PO>(
			fmap_in,
			fmap_out,
			Height,
        	Width,
			IC,
			batch
		);
	}
	else{
		conv1x1_dsp_unit<PI, PO, M_IC, M_OC>(
			fmap_in,
			fmap_out,
			w,
			Height,
        	Width,
			IC,
			OC,
			relu,
			batch		
		);
	}

}
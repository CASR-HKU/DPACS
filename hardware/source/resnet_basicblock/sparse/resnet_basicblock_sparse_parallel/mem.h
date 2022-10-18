template<int PI, class T>
T ceil_div(T A){
	T d = A / PI;
	if (A > d * PI){
		return d + 1;
	}
	else{
		return d;
	}
}



template<typename T_IN>
void M2M(T_IN *m_in, T_IN *m_out ,int REP){
	for(int rep = 0; rep < REP; rep++){
#pragma HLS pipeline
		m_out[rep]= m_in[rep];
	}
}



template<int PO, int W, typename T_IN, typename T_OUT>
void S2M_F(hls::stream<T_IN> &s_in, T_OUT *mem, int REP){

	for(int rep = 0; rep < REP; rep++){
#pragma HLS pipeline II=1
		T_IN s_read = s_in.read();
		T_OUT pack;
		for(T_C po = 0; po < PO; po++ ){
			pack.range((po + 1) * W - 1, po * W) = s_read.data[po];
		}
		mem[rep] = pack;
	}
}


template<int PO, int W, typename T_IN, typename T_OUT>
void S2M_random(hls::stream<T_IN> &s_in, T_OUT *mem, int REP){

	for(int rep = 0; rep < REP; rep++){
#pragma HLS pipeline II=1
		T_IN s_read = s_in.read();
		T_OUT pack;
		for(int po = 0; po < PO; po++ ){
			pack.range((po + 1) * W - 1, po * W) = s_read.data[po];
		}
		int index = (8779 * rep) % REP;
		mem[index] = pack;
	}
}




template<int PI, int W, typename T_IN, typename T_OUT>
void M2S_residual(T_IN *mem, hls::stream<T_OUT > &s_out, int Height, int Width, int IC, bool residual){
	
	int index = 0;

	for(int h = 0; h < Height; h++){
		for(int w = 0; w < Width; w++){
			for(int ic = 0; ic < IC / PI; ic++){
#pragma HLS pipeline
				if (!residual) break;
				T_IN m_read = mem[index++];
				T_OUT pack;
#pragma HLS ARRAY_PARTITION variable=pack.data complete dim=0
				for(int pi = 0; pi < PI; pi++ ){
					pack.data[pi] = m_read.range((pi + 1) * W - 1, pi * W);
				}
				s_out.write(pack);
			}
		}
	}
}



template<typename T>
void duplicate_keys(
	hls::stream<T> &key_in, 
	hls::stream<T> &key_conv, 
	hls::stream<T> &key_res,
	bool residual
){
	
	T key;	

	if(residual){
		for(int rep = 0; rep < MAX_H * MAX_H; rep++){
#pragma HLS PIPELINE II=1
		key = key_in.read();
		key_res.write(key);
		key_conv.write(key);
		if(key.end == 1) break;
		}
	}
	else{
		for(int rep = 0; rep < MAX_H * MAX_H; rep++){
	#pragma HLS PIPELINE II=1
			key = key_in.read();
			key_conv.write(key);
			if(key.end == 1) break;
		}
	}


}


template<int PO, int W, typename T_IN, typename T_OUT>
void S2M_key(
	hls::stream<T_IN> &s_in, 
	hls::stream<T_IN> &s_out, 
	T_OUT *mem, 
	ap_uint<MW> *out_mask,
	hls::stream<T_K> &key_in, 
	hls::stream<T_K> &key_out,
	int Height,
	int Width, 
	int OC,
	bool enable_pool,
	bool first_layer,
	bool return_mask
){

	int index = 0;
	T_C OCPO = OC / PO;
	int count = 0;
	ap_uint<20> nz = 0;

	const int buffer_size = 64 * 64 / 16;
	static ap_uint<16> out_mask_buffer[buffer_size];
	ap_uint<8> odiv = 0, orem = 0;
	ap_uint<MW> mask_pack=0, mask_read=0;
	

	for(ap_uint<9> i = 0; i < buffer_size; i++){
#pragma HLS PIPELINE
		out_mask_buffer[i] = 0;
	}


	if(first_layer){
		ap_uint<32> out_index = 0;
		for(ap_uint<32> rep = 0; rep < MAX_H * MAX_H; rep++){
#pragma HLS pipeline II=1
			T_K key = key_in.read();
			if(key.end == 1) break;	
			for(ap_uint<8> oc = 0; oc < 64 / PO; oc++){
#pragma HLS UNROLL
				T_IN s_read = s_in.read();
				T_OUT pack;
				for(T_C po = 0; po < PO; po++ ){
					pack.range((po + 1) * W - 1, po * W) = s_read.data[po];
				}
				mem[out_index++] = pack;
			}
		}
	}
	else{
		for(ap_uint<32> rep = 0; rep < MAX_H * MAX_H; rep++){
			T_K key = key_in.read();
			
			if (enable_pool) key_out.write(key);
			if(key.end == 1) break;

			nz = (key.x + key.y * Width);
			index = nz * OCPO;

			if(return_mask){
				odiv = nz / 16;
				orem = nz % 16;
				mask_read = out_mask_buffer[odiv];
				mask_read[orem] = 1;
				out_mask_buffer[odiv] = mask_read;
			}

			
			for(T_C oc = 0; oc < OCPO; oc++){
#pragma HLS pipeline II=1
				T_IN s_read = s_in.read();
				if (enable_pool) s_out.write(s_read);
				
				T_OUT pack;
				for(T_C po = 0; po < PO; po++ ){
					pack.range((po + 1) * W - 1, po * W) = s_read.data[po];
				}
				mem[index++] = pack;
			}
		}

		if(return_mask){
			ap_uint<9> mi = 0;
			for(ap_uint<9> i = 0; i < 64 * 64 / MW; i++){
#pragma HLS PIPELINE
				mask_pack = 0;
				for(ap_uint<9> j = 0; j < MW / 16; j++){
					mask_pack.range((j + 1) * 16 - 1, j * 16) = out_mask_buffer[mi++];
				}
				out_mask[i] = mask_pack;
			}
		}
	}
}



template<int PO, int W, typename T_IN, typename T_OUT>
void adjust_stream_remove_key(
	hls::stream<T_IN> &s_in, 
	hls::stream<T_IN> &s_out, 
	hls::stream<T_K> &key_in, 
	int Height,
	int Width, 
	int OC,
	int batch
){

	int index = 0;
	T_C OCPO = OC / PO;
	int count = 0;

	for(T_BATCH b = 0; b < batch; b++){
		for(ap_uint<32> rep = 0; rep < MAX_H * MAX_H; rep++){
			T_K key = key_in.read();
			if(key.end == 1) break;
			for(T_C oc = 0; oc < OCPO; oc++){
#pragma HLS pipeline II=1
				T_IN s_read = s_in.read();
				s_out.write(s_read);
			}
		}
	}
}

template<int PO, int W, typename T_IN, typename T_OUT>
void S2M_first_layer(
	hls::stream<T_IN> &s_in, 
	T_OUT *mem, 
	hls::stream<T_K> &key_in, 
	int Height,
	int Width, 
	int OC,
	int batch
){
#pragma HLS DATAFLOW
	hls::stream<BundleT<PO, T_F> > f_s;
#pragma HLS STREAM variable=f_s depth=2

	adjust_stream_remove_key<PO, W, T_IN, T_OUT>(s_in, f_s, key_in, Height, Width, OC, batch);
	S2M_F<PO, W, T_IN, T_OUT>(f_s, mem, batch * Height * Width * OC / PO);
}



template<int PO, int W, typename T_IN, typename T_OUT>
void output_unit(
	hls::stream<T_IN> &s_in, 
	hls::stream<T_IN> &s_out, 
	T_OUT *mem, 
	hls::stream<T_K> &key_in, 
	hls::stream<T_K> &key_out,
	int Height,
	int Width, 
	int OC,
	int batch,
	bool enable_pool,
	bool first_layer
){


	S2M_key<PO, W, T_IN, T_OUT>(s_in, s_out, mem, key_in, key_out, Height, Width, OC, batch, enable_pool, first_layer);
	

}

template<int PI, int W, typename T_IN, typename T_OUT, typename T>
void Residual_read(
	T_IN *mem, 
	hls::stream<T> &key_in,
	hls::stream<T_OUT > &s_out, 
	int Height,
	int Width,
	int IC,
	bool residual
){
	
	int index = 0;
	int ICPI = IC / PI;

	if(residual){
		for(int rep = 0; rep < MAX_H * MAX_H; rep++){
			T key = key_in.read();
			if(key.end == 1) break;
			index = (key.x + key.y * Width) * ICPI;

			for(int ic = 0; ic < ICPI; ic++){
#pragma HLS pipeline
				T_IN m_read = mem[index++];
				T_OUT pack;
#pragma HLS ARRAY_PARTITION variable=pack.data complete dim=0
				for(int pi = 0; pi < PI; pi++ ){
					pack.data[pi] = m_read.range((pi + 1) * W - 1, pi * W);
				}
				s_out.write(pack);
			}
		}
	}
}




template<int PI, int PO>
void adjust_stream_larger(
	hls::stream<BundleT<PI, T_F> > &fmap_in,
	hls::stream<BundleT<PO, T_F> > &fmap_out,
	hls::stream<T_K > &in_key,
	hls::stream<T_K > &out_key,
    T_C  C,
	T_BATCH batch
){

	// assert PI < PO
	T_K key;
	T_C CPI = C / PI;
	
	T_C CPO = ceil_div<PO>(C);

	BundleT<PO, T_F> out_pack;
#pragma HLS ARRAY_PARTITION variable=out_pack.data complete dim=0

	BundleT<PI, T_F> read_pack;
#pragma HLS ARRAY_PARTITION variable=read_pack.data complete dim=0


	int read_count = 0;
	int count = 0;


	for(int b = 0; b < batch; b++){
		for(int rep = 0; rep < MAX_H * MAX_H; rep++){
			key = in_key.read();
			out_key.write(key);
			if(key.end == 1) break;
			for(T_C i = 0; i < CPO; i++){
#pragma HLS PIPELINE
				for(T_C j = 0; j < PO / PI; j++){
					read_pack = fmap_in.read();
					for(T_C pi = 0; pi < PI; pi++){
						out_pack.data[j * PI + pi] = read_pack.data[pi];
					}
				}
				fmap_out.write(out_pack);
			}
		}
	}
}

template<int PI, int PO>
void adjust_stream_larger(
	hls::stream<BundleT<PI, T_F> > &fmap_in,
	hls::stream<BundleT<PO, T_F> > &fmap_out,
	hls::stream<T_K > &in_key,
	hls::stream<T_K > &out_key,
    T_C  C,
	T_BATCH batch,
	bool first_layer
){

	T_K key;
	T_C CPI = C / PI;
	
	T_C CPO = ceil_div<PO>(C);

	BundleT<PO, T_F> out_pack;
#pragma HLS ARRAY_PARTITION variable=out_pack.data complete dim=0

	BundleT<PI, T_F> read_pack;
#pragma HLS ARRAY_PARTITION variable=read_pack.data complete dim=0


	int read_count = 0;
	int count = 0;

	for(int b = 0; b < batch; b++){
		for(int rep = 0; rep < MAX_H * MAX_H; rep++){
			key = in_key.read();
			out_key.write(key);
			if(key.end == 1) break;
			for(T_C i = 0; i < CPO; i++){
#pragma HLS PIPELINE
				if (!first_layer){
					for(T_C j = 0; j < PO / PI; j++){
						read_pack = fmap_in.read();
						for(T_C pi = 0; pi < PI; pi++){
							out_pack.data[j * PI + pi] = read_pack.data[pi];
						}
					}
					fmap_out.write(out_pack);
				}
				else{
					read_pack = fmap_in.read();
					for(T_C pi = 0; pi < PI; pi++){
						out_pack.data[pi] = read_pack.data[pi];
					}
					fmap_out.write(out_pack);
				}
			}
		}
	}
}



template<int PI, int PO>
void adjust_stream_first_layer(
	hls::stream<BundleT<PI, T_F> > &fmap_in,
	hls::stream<BundleT<PO, T_F> > &fmap_out,
	hls::stream<T_K > &in_key,
	hls::stream<T_K > &out_key,
    T_C  C,
	T_BATCH batch
){

	T_K key;
	T_C CPI = ceil_div<PI>(C);

	BundleT<PO, T_F> out_pack;
#pragma HLS ARRAY_PARTITION variable=out_pack.data complete dim=0

	BundleT<PI, T_F> read_pack;
#pragma HLS ARRAY_PARTITION variable=read_pack.data complete dim=0

	int out_count = 0;

	for(T_BATCH b = 0; b < batch; b++){
		for(int rep = 0; rep < MAX_H * MAX_H; rep++){
#pragma HLS PIPELINE II=1
			key = in_key.read();
			out_key.write(key);
			if(key.end == 1) break;
			read_pack = fmap_in.read();
			for(T_C po = 0; po < PO; po++){
				out_pack.data[po] = read_pack.data[po];
			}
			fmap_out.write(out_pack);
		}
	}
}


template<int PI, int PO>
void adjust_stream_smaller(
	hls::stream<BundleT<PI, T_F> > &fmap_in,
	hls::stream<BundleT<PO, T_F> > &fmap_out,
	hls::stream<T_K > &in_key,
	hls::stream<T_K > &out_key,
    T_C  C,
	T_BATCH batch
){

	T_K key;
	T_C CPI = ceil_div<PI>(C);


	BundleT<PO, T_F> out_pack;
#pragma HLS ARRAY_PARTITION variable=out_pack.data complete dim=0

	BundleT<PI, T_F> read_pack;
#pragma HLS ARRAY_PARTITION variable=read_pack.data complete dim=0

	int out_count = 0;

	for(T_BATCH b = 0; b < batch; b++){
		for(int rep = 0; rep < MAX_H * MAX_H; rep++){
			key = in_key.read();
			out_key.write(key);
			if(key.end == 1) break;
			for(T_C i = 0; i < CPI; i++){
#pragma HLS PIPELINE 
				read_pack = fmap_in.read();
				for(T_C j = 0; j < PI / PO; j++){
					for(T_C po = 0; po < PO; po++){
						out_pack.data[po] = read_pack.data[j * PO + po];
					}
					fmap_out.write(out_pack);
				}
			}
		}
	}
}


template<int PI, int PO>
void adjust_stream_same(
	hls::stream<BundleT<PI, T_F> > &fmap_in,
	hls::stream<BundleT<PO, T_F> > &fmap_out,
	hls::stream<T_K > &in_key,
	hls::stream<T_K > &out_key,
    T_C  C
){

	T_K key;
	T_C CPI = ceil_div<PI>(C);

	BundleT<PO, T_F> out_pack;
#pragma HLS ARRAY_PARTITION variable=out_pack.data complete dim=0

	BundleT<PI, T_F> read_pack;
#pragma HLS ARRAY_PARTITION variable=read_pack.data complete dim=0
	int count = 0 ;


	for(int rep = 0; rep < MAX_H * MAX_H; rep++){
		key = in_key.read();
		out_key.write(key);
		if(key.end == 1) break;
		
		for(T_C i = 0; i < CPI; i++){
#pragma HLS PIPELINE II=1
			read_pack = fmap_in.read();
			fmap_out.write(read_pack);
		}
	}
	
}


template<int PI>
void Load_Weight_Merge(
	ap_int<PI * WW> *weight, 
	hls::stream<ap_int<PI * WW> > &weight_s,
    T_C  IC_0,
    T_C  OC_0,
    T_C  OC_3,
    T_C  OC_1,
	T_C  next_c,
	T_H  Height,
	T_H  Width,
	bool skip_0,
	bool skip_3,
	bool skip_1,
	bool use_mask,
	bool enable_pool,
	bool first_layer
){


	ap_uint<16> w_mask_count = 0; 
	ap_uint<16> mask_count = 0; 
	ap_uint<16> w0_count = 0; 
	ap_uint<16> w3_count = 0; 
	ap_uint<16> w1_count = 0; 
	ap_uint<16> fc_count = 0; 


	if(first_layer){
		w_mask_count = 0;
		mask_count = 0;
	}
	else if (!use_mask){
		w_mask_count = IC_0 / PI;
	}
	else{
		mask_count =  ceil_div<MW>(Height * Width);
	}

	if (!skip_0){
		w0_count = 9 * IC_0 * OC_0 / PI;
	}
	if (!skip_3){
		w3_count = OC_0 * OC_3 * 9 / PI;
	}
	if (!skip_1){
		w1_count = OC_3 * OC_1 / PI;
	}
	if (enable_pool){
		fc_count = OC_1 * next_c / CPRUNE_FACTOR / PI;
	}

	ap_uint<32> total_count = mask_count + w_mask_count + w0_count + w3_count + w1_count + fc_count;

	for(ap_uint<32> rep = 0; rep < total_count; rep++){
#pragma HLS PIPELINE II=1
		ap_int<PI * WW> w_read = weight[rep];
		weight_s.write(w_read);
	}
}

template<typename T>
void read_weight_cprune(
	T *weight,
	hls::stream<T> &weight_s,
	ap_uint<32> start_index
){
#pragma HLS INLINE OFF

	for(int i = 0; i < CPRUNE_FACTOR; i++){
#pragma HLS PIPELINE II=1
		T w_read = weight[start_index++];
		weight_s.write(w_read);
	}
}


template<int PI>
void Load_Weight_C_PRUNE(
	ap_int<PI * WW> *weight, 
	hls::stream<ap_int<PI * WW> > &weight_s,
    T_C  IC_0,
    T_C  OC_0,
    T_C  OC_3,
    T_C  OC_1,
	T_C  next_c,
	T_H  Height,
	T_H  Width,
	bool skip_0,
	bool skip_3,
	bool skip_1,
	bool use_mask,
	bool enable_pool,
	ap_uint<16> cmask_ic,
	ap_uint<16> cmask_oc
){

	ap_uint<16> w_mask_count = 0; 
	ap_uint<16> mask_count = 0; 
	ap_uint<16> fc_count = 0;

	if (!use_mask){
		w_mask_count = IC_0 / PI;
	}
	else{
		mask_count = ceil_div<MW>(Height * Width);
	}
	if(enable_pool){
		fc_count = OC_1 * next_c / CPRUNE_FACTOR / PI;
	}


	ap_uint<32> total_static = w_mask_count + fc_count;
	ap_uint<32> read_index = 0;


	for(ap_uint<32> rep = 0; rep < total_static; rep++){
#pragma HLS PIPELINE II=1
		ap_int<PI * WW> w_read = weight[read_index++];
		weight_s.write(w_read);
	}


	if(!skip_0){
		for(ap_uint<4> k = 0; k < 9; k++){
			for(T_C oc = 0; oc < OC_0 / CPRUNE_FACTOR; oc++){
				bool oc_flag = cmask_ic[oc];
				for(T_C ic = 0; ic < IC_0 / CPRUNE_FACTOR; ic++){
					if(oc_flag){
						read_weight_cprune<ap_int<CPRUNE_FACTOR * WW> >(weight, weight_s, read_index);
					}
					read_index += CPRUNE_FACTOR;
				}
			}
		}
	}


	if(!skip_3){
		for(ap_uint<4> k = 0; k < 9; k++){
			for(T_C oc = 0; oc < OC_3 / CPRUNE_FACTOR; oc++){
				for(T_C ic = 0; ic < OC_0 / CPRUNE_FACTOR; ic++){
					bool ic_flag = cmask_ic[ic];
					if(ic_flag){
						read_weight_cprune<ap_int<CPRUNE_FACTOR * WW> >(weight, weight_s, read_index);
					}
					read_index += CPRUNE_FACTOR;
				}
			}
		}
	}


	if(!skip_1){
		for(T_C oc = 0; oc < OC_1 / CPRUNE_FACTOR; oc++){
			for(T_C ic = 0; ic < OC_3 / CPRUNE_FACTOR; ic++){
				bool ic_flag = cmask_ic[ic];
				if(ic_flag){
					read_weight_cprune<ap_int<CPRUNE_FACTOR * WW> >(weight, weight_s, read_index);
				}
				read_index += CPRUNE_FACTOR;
			}
		}
	}



	for(ap_uint<32> rep = 0; rep < mask_count; rep++){
#pragma HLS PIPELINE II=1
		ap_int<PI * WW> w_read = weight[read_index++];
		weight_s.write(w_read);
	}	

}



template<int PI>
void Load_Weight_Wrap(
	ap_int<PI * WW> *weight, 
	hls::stream<ap_int<PI * WW> > &weight_s,
    T_C  IC_0,
    T_C  OC_0,
    T_C  OC_3,
    T_C  OC_1,
	T_C  next_c,
	T_H  Height,
	T_H  Width,
	bool skip_0,
	bool skip_3,
	bool skip_1,
	bool use_mask,
	bool enable_pool,
	ap_uint<16> cmask_ic,
	ap_uint<16> cmask_oc,
	bool use_c_prune,
	bool first_layer
){
	if (use_c_prune){
		Load_Weight_C_PRUNE<PI>(
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
			cmask_oc
		);
	}
	else{
		Load_Weight_Merge<PI>(
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
			first_layer
		);		
	}
}



template<int PI>
void route_weight(
	hls::stream<ap_int<PI * WW> > &weight_s,
	hls::stream<ap_int<PI_0 * WW> > &w_mask_s,
	hls::stream<ap_int<PI_0 * WW> > &w_0_s,
	hls::stream<ap_int<PI_3 * WW> > &w_3_s,
	hls::stream<ap_int<PI_1 * WW> > &w_1_s,
	hls::stream<ap_int<PO_1 * WW> > &fc_s,
	hls::stream<T_MASK > &mask_s,
    T_C  IC_0,
    T_C  OC_0,
    T_C  OC_3,
    T_C  OC_1,
	T_C  next_c,
	T_H  Height,
	T_H  Width,
	bool skip_0,
	bool skip_3,
	bool skip_1,
	bool use_mask,
	bool enable_pool,
	bool first_layer
){

	ap_uint<16> w_mask_count = 0; 
	ap_uint<16> mask_count = 0; 
	ap_uint<16> w0_count = 0; 
	ap_uint<16> w3_count = 0; 
	ap_uint<16> w1_count = 0; 
	ap_uint<16> fc_count = 0; 
	int global_count = 0;

	if(first_layer){
		w_mask_count = 0;
		mask_count = 0;
	}
	else if (!use_mask){
		w_mask_count = IC_0 / PI;
	}
	else{
		mask_count = ceil_div<MW>(Height * Width);
	}
	
	if(enable_pool){
		fc_count = OC_1 * next_c / CPRUNE_FACTOR / PI;
	}
	if (!skip_0){
		w0_count = 9 * IC_0 * OC_0 / PI;
	}
	if (!skip_3){
		w3_count = OC_0 * OC_3 * 9 / PI;
	}
	if (!skip_1){
		w1_count = OC_3 * OC_1 / PI;
	}

	for(int i = 0; i < w_mask_count; i++){
#pragma HLS PIPELINE 
		ap_int<PI * WW> w_read = weight_s.read();
		for(int j = 0; j < PI / PI_0; j++){
			ap_int<PI_0 * WW> w_mask = w_read.range((j + 1) * PI_0 * WW - 1, j * PI_0 * WW);
			w_mask_s.write(w_mask);
		}
	}

	for(int i = 0; i < fc_count; i++){
#pragma HLS PIPELINE 
		ap_int<PI * WW> w_read = weight_s.read();
		for(int j = 0; j < PI / PO_1; j++){
			ap_int<PO_1 * WW> fc = w_read.range((j + 1) * PO_1 * WW - 1, j * PO_1 * WW);
			fc_s.write(fc);
		}
	}


	for(int i = 0; i < w0_count; i++){
#pragma HLS PIPELINE 
		ap_int<PI * WW> w_read = weight_s.read();
		for(int j = 0; j < PI / PI_0; j++){
			ap_int<PI_0 * WW> w_0 = w_read.range((j + 1) * PI_0 * WW - 1, j * PI_0 * WW);
			w_0_s.write(w_0);
		}
	}

	for(int i = 0; i < w3_count; i++){
#pragma HLS PIPELINE 
		ap_int<PI * WW> w_read = weight_s.read();
		for(int j = 0; j < PI / PI_3; j++){
			ap_int<PI_3 * WW> w_3 = w_read.range((j + 1) * PI_3 * WW - 1, j * PI_3 * WW);
			w_3_s.write(w_3);
		}
	}

	for(int i = 0; i < w1_count; i++){
#pragma HLS PIPELINE
		ap_int<PI * WW> w_read = weight_s.read();
		for(int j = 0; j < PI / PI_1; j++){
			ap_int<PI_1 * WW> w_1 = w_read.range((j + 1) * PI_1 * WW - 1, j * PI_1 * WW);
			w_1_s.write(w_1);
		}
	}

	for(int i = 0; i < mask_count; i++){
#pragma HLS PIPELINE
		T_MASK mask_read = weight_s.read();
		mask_s.write(mask_read);
	}

}

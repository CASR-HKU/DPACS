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

	int count = 0;
	for(int rep = 0; rep < REP; rep++){
#pragma HLS pipeline II=1	
		T_IN s_read = s_in.read();
		T_OUT pack;
		for(int po = 0; po < PO; po++ ){
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
void M2S_reduce(T_IN *mem, hls::stream<T_OUT > &s_out, int Height, int Width, T_BATCH batch){
	
	int count = 0;

	for(ap_uint<32> rep = 0; rep < batch * Height * Width * P4 / PI; rep++){
#pragma HLS PIPELINE II=1
		T_IN m_read = mem[rep];
		T_OUT pack;
#pragma HLS ARRAY_PARTITION variable=pack.data complete dim=0

		T_C pi = 0;
		for(T_C i = 0; i < PI / P4; i++ ){
			for(T_C p = 0; p < P4; p++ ){
				pack.data[p] = m_read.range((pi + 1) * W - 1, pi * W);
				pi++;
			}
			s_out.write(pack);
		}
	}
}


template<int PI, int W, typename T_IN, typename T_OUT>
void M2S_F(T_IN *mem, hls::stream<T_OUT > &s_out, int Height, int Width, int IC, T_BATCH batch){
	
	int index = 0;

	for(int rep = 0; rep < batch * Height * Width * IC / PI; rep++){
#pragma HLS pipeline
		T_IN m_read = mem[rep];
		T_OUT pack;
#pragma HLS ARRAY_PARTITION variable=pack.data complete dim=0
		for(T_C pi = 0; pi < PI; pi++ ){
			pack.data[pi] = m_read.range((pi + 1) * W - 1, pi * W);
		}
		s_out.write(pack);
	}
}

template<int PI, int W, typename T_IN, typename T_OUT>
void input_unit(
	T_IN *mem, 
	hls::stream<T_OUT > &s_out, 
	int Height, 
	int Width, 
	int IC, 
	T_BATCH batch,
	bool first_layer
){
	if(first_layer){
		M2S_reduce<PI, W, T_IN, T_OUT>(mem, s_out, Height, Width, batch);
	}
	else{
		M2S_F<PI, W, T_IN, T_OUT>(mem, s_out, Height, Width, IC, batch);
	}
}



template<int PI, int W, typename T_IN, typename T_OUT>
void M2S_residual(T_IN *mem, hls::stream<T_OUT > &s_out, int Height, int Width, int IC, bool residual, T_BATCH batch){
	
	int index = 0;


	for(int rep = 0; rep < batch * Height * Width * IC / PI; rep++){
#pragma HLS pipeline
		if (!residual) break;
		T_IN m_read = mem[rep];
		T_OUT pack;
#pragma HLS ARRAY_PARTITION variable=pack.data complete dim=0
		for(int pi = 0; pi < PI; pi++ ){
			pack.data[pi] = m_read.range((pi + 1) * W - 1, pi * W);
		}
		s_out.write(pack);
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
void S2M_key(hls::stream<T_IN> &s_in, T_OUT *mem, hls::stream<T_K> &key_s, int Width, int OC){

	int index = 0;
	int OCPO = OC / PO;

	for(int rep = 0; rep < MAX_H * MAX_H; rep++){
		T_K key = key_s.read();
		if(key.end == 1) break;

		index = (key.x + key.y * Width) * OCPO;
		
		for(int oc = 0; oc < OCPO; oc++){
#pragma HLS pipeline II=1
			T_IN s_read = s_in.read();
			T_OUT pack;
			for(int po = 0; po < PO; po++ ){
				pack.range((po + 1) * W - 1, po * W) = s_read.data[po];
			}
			mem[index++] = pack;
		}
	}
}


template<int PI, int W, typename T_IN, typename T_OUT, typename T>
void Residual_read(
	T_IN *mem, 
	hls::stream<T> &key_in,
	hls::stream<T_OUT > &s_out, 
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
	int Height,
	int Width,
    T_C  C,
	T_BATCH batch
){

	// assert PI < PO
	T_K key;
	T_C CPI = C / PI;
	
	T_C CPO = ceil_div<PO>(C);
	const int factor = PO / PI;
	T_C CPI_F = ceil_div<factor>(CPI);

	BundleT<PO, T_F> out_pack;
#pragma HLS ARRAY_PARTITION variable=out_pack.data complete dim=0

	BundleT<PI, T_F> read_pack;
#pragma HLS ARRAY_PARTITION variable=read_pack.data complete dim=0


	int read_count = 0;

	for(int rep = 0; rep < batch * Height * Width * C / PO; rep++){
#pragma HLS PIPELINE
		for(T_C i = 0; i < PO / PI; i++){
			read_pack = fmap_in.read();
			for(T_C pi = 0; pi < PI; pi++){
#pragma HLS UNROLL
				out_pack.data[i * PI + pi] = read_pack.data[pi];
			}
		}
		fmap_out.write(out_pack);
	}
}

template<int PI, int PO>
void adjust_stream_larger(
	hls::stream<BundleT<PI, T_F> > &fmap_in,
	hls::stream<BundleT<PO, T_F> > &fmap_out,
	int Height,
	int Width,
    T_C  C,
	T_BATCH batch,
	bool first_layer
){

	T_K key;
	T_C CPI = C / PI;
	
	T_C CPO = ceil_div<PO>(C);
	const int factor = PO / PI;
	T_C CPI_F = ceil_div<factor>(CPI);

	BundleT<PO, T_F> out_pack;
#pragma HLS ARRAY_PARTITION variable=out_pack.data complete dim=0

	BundleT<PI, T_F> read_pack;
#pragma HLS ARRAY_PARTITION variable=read_pack.data complete dim=0


	int read_count = 0;

	for(int rep = 0; rep < batch * Height * Width * C / PO; rep++){
#pragma HLS PIPELINE
		if(!first_layer){
			for(T_C i = 0; i < PO / PI; i++){
				read_pack = fmap_in.read();
				for(T_C pi = 0; pi < PI; pi++){
	#pragma HLS UNROLL
					out_pack.data[i * PI + pi] = read_pack.data[pi];
				}
			}
			fmap_out.write(out_pack);
		}
		else{
			read_pack = fmap_in.read();
			for(T_C pi = 0; pi < PI; pi++){
#pragma HLS UNROLL
				out_pack.data[pi] = read_pack.data[pi];
			}
			fmap_out.write(out_pack);
		}
	}
}



template<int PI, int PO>
void adjust_stream_first_layer(
	hls::stream<BundleT<PI, T_F> > &fmap_in,
	hls::stream<BundleT<PO, T_F> > &fmap_out,
	T_H Height,
	T_H Width,
	T_BATCH batch
){
	// assert PI > PO

	BundleT<PO, T_F> out_pack;
#pragma HLS ARRAY_PARTITION variable=out_pack.data complete dim=0

	BundleT<PI, T_F> read_pack;
#pragma HLS ARRAY_PARTITION variable=read_pack.data complete dim=0

	int count = 0;

	for(int rep = 0; rep <batch * Height * Width; rep++){
#pragma HLS PIPELINE II=1
		read_pack = fmap_in.read();
		for(T_C po = 0; po < P4; po++){
			out_pack.data[po] = read_pack.data[po];
		}
		fmap_out.write(out_pack);
	}
	
}


template<int PI, int PO>
void adjust_stream_smaller(
	hls::stream<BundleT<PI, T_F> > &fmap_in,
	hls::stream<BundleT<PO, T_F> > &fmap_out,
	int Height,
	int Width,
    T_C  C,
	T_BATCH batch
){

	int count = 0;
	T_K key;
	T_C CPO = C / PO;
	T_C CPI = ceil_div<PI>(C);
	const int factor = PI / PO;

	T_C CPO_F = ceil_div<factor>(CPO);

	BundleT<PO, T_F> out_pack;
#pragma HLS ARRAY_PARTITION variable=out_pack.data complete dim=0

	BundleT<PI, T_F> read_pack;
#pragma HLS ARRAY_PARTITION variable=read_pack.data complete dim=0

	int out_count = 0;

	for(int rep = 0; rep < batch * Height * Width * C / PI; rep++){
		read_pack = fmap_in.read();
		for(int i = 0; i < PI / PO; i++){		
#pragma HLS PIPELINE II=1
			for(int po = 0; po < PO; po++){
				out_pack.data[po] = read_pack.data[i * PO + po];
			}
			fmap_out.write(out_pack);		
		}
	}
}


template<int PI, int PO>
void adjust_stream_same(
	hls::stream<BundleT<PI, T_F> > &fmap_in,
	hls::stream<BundleT<PO, T_F> > &fmap_out,
	int Height,
	int Width,
    T_C  C,
	T_BATCH batch
){

	T_K key;
	T_C CPO = C / PO;
	T_C CPI = ceil_div<PI>(C);
	const int factor = PI / PO;

	T_C CPO_F = ceil_div<factor>(CPO);

	BundleT<PO, T_F> out_pack;
#pragma HLS ARRAY_PARTITION variable=out_pack.data complete dim=0

	BundleT<PI, T_F> read_pack;
#pragma HLS ARRAY_PARTITION variable=read_pack.data complete dim=0

	for(int rep = 0; rep < batch * Height * Width * C / PI; rep++){
#pragma HLS PIPELINE
		
		read_pack = fmap_in.read();
		fmap_out.write(read_pack);
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
	bool skip_0,
	bool skip_3,
	bool skip_1,
	bool first_layer
){



	ap_uint<16> w0_count = 0; 
	ap_uint<16> w3_count = 0; 
	ap_uint<16> w1_count = 0; 
	ap_uint<16> w_first_count = 0;

	if (first_layer){
		w_first_count = P4 * IC_0 * 9 / PI;
	}
	if (!skip_0){
		w0_count = IC_0 * OC_0 * 9 / PI;
	}
	if (!skip_3){
		w3_count = OC_0 * OC_3 * 9 / PI;
	}
	if (!skip_1){
		w1_count = OC_3 * OC_1 / PI;
	}

	ap_uint<32> total_count =  w0_count + w3_count + w1_count + w_first_count;

	for(ap_uint<32> rep = 0; rep < total_count; rep++){
#pragma HLS PIPELINE II=1
		ap_int<PI * WW> w_read = weight[rep];
		weight_s.write(w_read);
	}
}




template<int PI>
void route_weight(
	hls::stream<ap_int<PI * WW> > &weight_s,
	hls::stream<ap_int<P4 * WW> > &w_first_s,
	hls::stream<ap_int<PI_0 * WW> > &w_0_s,
	hls::stream<ap_int<PI_3 * WW> > &w_3_s,
	hls::stream<ap_int<PI_1 * WW> > &w_1_s,
    T_C  IC_0,
    T_C  OC_0,
    T_C  OC_3,
    T_C  OC_1,
	bool skip_0,
	bool skip_3,
	bool skip_1,
	bool first_layer
){


	ap_uint<16> w0_count = 0; 
	ap_uint<16> w3_count = 0; 
	ap_uint<16> w1_count = 0;
	ap_uint<16> w_first_count = 0;  

	int global_count = 0;

	if (first_layer){
		w_first_count = P4 * IC_0 * 9 / PI;
	}
	if (!skip_0){
		w0_count = IC_0 * OC_0 * 9 / PI;
	}
	if (!skip_3){
		w3_count = OC_0 * OC_3 * 9 / PI;
	}
	if (!skip_1){
		w1_count = OC_3 * OC_1 / PI;
	}


	for(int i = 0; i < w_first_count; i++){
#pragma HLS PIPELINE 
		ap_int<PI * WW> w_read = weight_s.read();
		for(int j = 0; j < PI / P4; j++){
			ap_int<P4 * WW> w_0 = w_read.range((j + 1) * P4 * WW - 1, j * P4 * WW);
			w_first_s.write(w_0);
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

}

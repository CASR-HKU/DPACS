

template<int PI, int W, typename T_IN, typename T_OUT>
void M2S_from_key(
	T_IN *mem, 
	hls::stream<T_OUT > &s_out, 
	hls::stream<T_K> &key_in,
	hls::stream<T_K> &key_out,
	int Height, 
	int Width, 
	int IC
){
	
	int count = 0;
	int index = 0;
	int ICPI = IC / PI;

	for(int rep = 0; rep < MAX_H * MAX_H; rep++){
		T_K key = key_in.read();
		key_out.write(key);
		if(key.end == 1) break;

		index = (key.x + key.y * Width) * ICPI;
		for(T_C ic = 0; ic < IC / PI; ic++){
#pragma HLS pipeline II=1
			T_OUT pack;
#pragma HLS ARRAY_PARTITION variable=pack.data complete dim=0
			T_IN m_read = mem[index++];
			for(ap_uint<7> pi = 0; pi < PI; pi++ ){
				pack.data[pi] = m_read.range((pi + 1) * W - 1, pi * W);
			}
			s_out.write(pack);
		}
	}
	
}


template<int PI, int W, typename T_IN, typename T_OUT>
void M2S_dense_bundle(T_IN *mem, hls::stream<T_OUT > &s_out, int Height, int Width, int IC){
	
	int index = 0;

	for(T_H h = 0; h < Height; h++){
		for(T_H w = 0; w < Width; w++){
			for(T_C ic = 0; ic < IC / PI; ic++){
#pragma HLS pipeline
				T_IN m_read = mem[index++];
				T_OUT pack;
#pragma HLS ARRAY_PARTITION variable=pack.data complete dim=0
				for(ap_uint<7> pi = 0; pi < PI; pi++ ){
					pack.data[pi] = m_read.range((pi + 1) * W - 1, pi * W);
				}
				s_out.write(pack);
			}
		}
	}
}


template<typename T_IN, typename T_OUT>
void M2S_mask(
	hls::stream<T_IN> &mask_s, 
	hls::stream<T_OUT > 
	&s_out, 
	T_H Height, 
	T_H Width
){
	

	T_K key;
	int REP = ceil_div<MW>(Height * Width);
	T_MASK mask_buffer[MAX_H * MAX_H / MW];
	T_MASK mask_pack = 0;
	int read_count = 0;

	for(int rep = 0; rep < REP; rep++){
#pragma HLS PIPELINE II=1
		mask_buffer[rep] = mask_s.read();
	}
	int count = 0;
	T_H x_index = 0;
	T_H y_index = 0;
	for(int rep = 0; rep < REP; rep++){
		mask_pack = mask_buffer[rep];
		for(int i = 0; i < MW; i++){
#pragma HLS PIPELINE II=1
			bool nz_flag = mask_pack[i];
			key.x = x_index;
			key.y = y_index;
			key.end = 0;
			if (nz_flag){
				s_out.write(key);
			}
			count++;
			if(count == Height * Width) break;
			x_index++;
			if (x_index == Width){
				x_index = 0;
				y_index++;
			}
		}
	}
	key.end = 1;
	s_out.write(key);
	
}




template<int PI, int W, typename T_IN, typename T_OUT>
void M2S_mask_merge(
	T_IN *mem, 
	T_MASK *mask, 
	hls::stream<T_OUT > &s_out,
	hls::stream<T_K > &s_key,
	int Height, 
	int Width,
	int IC
){
#pragma HLS INLINE

	T_K key;
	int REP = ceil_div<MW>(Height * Width);
	T_MASK mask_buffer[MAX_H * MAX_H / MW];
	T_MASK mask_pack = 0;

	for(int rep = 0; rep < REP; rep++){
#pragma HLS PIPELINE II=1
		mask_buffer[rep] = mask[rep];
	}

	int count = 0;
	T_H x_index = 0;
	T_H y_index = 0;
	int index = 0;
	T_C ICPI = IC / PI;

	T_OUT pack;
#pragma HLS ARRAY_PARTITION variable=pack.data complete dim=0

	for(int rep = 0; rep < REP; rep++){
		mask_pack = mask_buffer[rep];
		for(int i = 0; i < MW; i++){
#pragma HLS PIPELINE II=1
			bool nz_flag = mask_pack[i];
			key.x = x_index;
			key.y = y_index;
			key.end = 0;
			if (nz_flag){
				s_key.write(key);
				index = (x_index + y_index * Width) * ICPI;
				for(T_C ic = 0; ic < ICPI; ic++){
#pragma HLS pipeline II=1
					T_IN m_read = mem[index++];
					for(ap_uint<7> pi = 0; pi < PI; pi++ ){
						pack.data[pi] = m_read.range((pi + 1) * W - 1, pi * W);
					}
					s_out.write(pack);
				}
			}
			x_index++;
			if (x_index == Width){
				x_index = 0;
				y_index++;
			}
		}
	}

	key.end = 1;
	s_key.write(key);
}



// Spatial mask unit
template<int PI, int W, typename T_IN, typename T_OUT>
void conv_1x1_mask_wrap(
	T_IN *fin,
	hls::stream<ap_int<PI * WW> > &w_mask,
	hls::stream<T_K> &s_key, 
	hls::stream<T_OUT> &fout,
	T_H Height,
	T_H Width,
	T_C IC
){

	T_K key;	
	int count = 0;
	bool out_flag = 1;
	T_OUT out_pack;

	T_IN in_buffer[MAX_IC / PI];
	ap_int<WW * PI> w_buffer[MAX_IC / PI];

	ap_uint<32> index = 0;

	int w_index = 0;
	for(T_C ic = 0; ic < IC / PI; ic++){
#pragma HLS PIPELINE
		w_buffer[ic] = w_mask.read();
	}

	for(T_H h = 0; h < Height; h++){
		for(T_H w = 0; w < Width; w++){
			key.x = w;
			key.y = h;
			key.end = 0;
			T_S sum = 0;
			for(T_C ic = 0; ic < IC / PI; ic++){
#pragma HLS PIPELINE II=1
				T_IN f_read = fin[index++];
				in_buffer[ic] = f_read;
				ap_int<WW * PI> w_pack = w_buffer[ic];

				for(ap_uint<7> pi = 0; pi < PI; pi++){
					ap_int<FW> in_pi = f_read.range((pi + 1) * FW - 1, pi * FW);
					T_W w = w_pack.range((pi + 1) * WW - 1, pi * WW);
					sum += in_pi * w;
				}
			}
			out_flag = (sum > 0) ? 1 : 0;
			if(out_flag) s_key.write(key);
			for(T_C ic = 0; ic < IC / PI; ic++){
#pragma HLS PIPELINE II=1
				if(!out_flag) break;
				T_IN out_f_ready = in_buffer[ic];
				for(ap_uint<7> pi = 0; pi < PI; pi++){
					out_pack.data[pi] = out_f_ready.range((pi + 1) * FW - 1, pi * FW);
				}
				fout.write(out_pack);
			}			
		}
	}
	key.end = 1;
	s_key.write(key);
	

}

template<int PI, int W, typename T, typename T_OUT>
void load_mask_wrap(
	hls::stream<T_MASK> &mask,
	ap_int<FW * PI> *fin,
	hls::stream<T> &s_key, 
	hls::stream<T_OUT> &fout,
	int Height,
	int Width,
	int IC
){
#pragma HLS DATAFLOW

    hls::stream<BundleT<PI, T_F> > fin_s;
#pragma HLS STREAM variable=fin_s depth=8

	hls::stream<T_K > key_mask;
#pragma HLS STREAM variable=key_mask depth=4

	hls::stream<ap_uint<1> > s_bool;
#pragma HLS STREAM variable=s_bool depth=16

    M2S_mask<>(mask, key_mask, Height, Width);
    M2S_from_key<PI, W>(fin, fout, key_mask, s_key, Height, Width, IC);

}



template<int PI, int W>
void M2S_reduce(
	ap_int<FW * PI> *mem, 
	hls::stream<BundleT<PI, T_F> > &s_out, 
	T_H Height, 
	T_H Width
){
	
	for(ap_uint<32> rep = 0; rep < Height * Width * P4 / PI; rep++){
#pragma HLS pipeline II = 1
		ap_int<FW * PI> m_read = mem[rep];
		T_C ic_index = 0;
		BundleT<PI, T_F> out_pack;

		for(ap_uint<7> i = 0; i < PI / P4; i++){ 
			for(ap_uint<7> pi = 0; pi < P4; pi++){
				out_pack.data[pi] = m_read.range((ic_index + 1) * FW - 1, ic_index * FW);
				ic_index++;
			}
			s_out.write(out_pack);
		}
	}
}


template<int PI, int W>
void first_layer_key(
	hls::stream<BundleT<PI, T_F> > &s_in, 
	hls::stream<BundleT<PI, T_F> > &s_out, 
	hls::stream<T_K> &s_key,
	T_H Height, 
	T_H Width
){
	
	T_K out_key;
	int count = 0;

	for(T_H h = 0; h < Height; h++){
		for(T_H w = 0; w < Width; w++){
#pragma HLS PIPELINE II=1
			out_key.x = w;
			out_key.y = h;
			out_key.end = 0;
			s_key.write(out_key);
			s_out.write(s_in.read());
		}
	}
	out_key.end = 1;
	s_key.write(out_key);
	
}



template<int PI, int W, typename T, typename T_OUT>
void first_layer_unit(
	ap_int<FW * PI> *fin,
	hls::stream<T> &s_key, 
	hls::stream<T_OUT> &fout,
	int Height,
	int Width
){
#pragma HLS DATAFLOW

	hls::stream<BundleT<PI, T_F> > fin_s; 
#pragma HLS STREAM variable=fin_s depth=2
	M2S_reduce<PI, W>(fin, fout, Height, Width);
}




template<int PI, int W, typename T, typename T_OUT>
void input_unit(
	hls::stream<T_MASK> &mask,
	hls::stream<ap_int<PI * WW> > &w_mask,
	ap_int<FW * PI> *fin,
	hls::stream<T> &s_key, 
	hls::stream<T_OUT> &fout,
	int Height,
	int Width,
	int IC,
	bool use_mask,
	bool first_layer
){

	if(first_layer){
		first_layer_unit<PI, W, T, T_OUT>(fin, s_key, fout, Height, Width);
	}
	else if(use_mask){ // if obtained spatial mask, load sparse input
		load_mask_wrap<PI, W>(mask, fin, s_key, fout, Height, Width, IC);
	}
	else{ // if not obtained spatial mask, load dense input and generate spatial mask
		conv_1x1_mask_wrap<PI, W>(fin, w_mask, s_key, fout, Height, Width, IC);
	}
}
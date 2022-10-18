// A small linebuffer for first conv layer with small IC
template<int PI>
void line_buffer_first_layer(
	hls::stream<BundleT<PI, T_F> > &fmap_in,
	hls::stream<BundleT<9, ap_int<PI * FW> > > &fmap_out,
	T_H  Height,
    T_H  Width,
	bool STRIDE,
	T_BATCH batch
){

	typedef ap_int<PI * FW> T_FPI;
	typedef ap_int<P4 * FW> T_FP4;

	typedef BundleT<PI, T_F> TB_FPI;
	typedef BundleT<9, ap_int<PI * FW> > TB_OUT;

	ap_uint<2> stride = (STRIDE == 1) ? 2 : 1;

	const int BUFFER_ROWS = 3;
	const int BUFFER_WIDTH = 256;

	T_FP4 line_buff[BUFFER_ROWS][BUFFER_WIDTH];
#pragma HLS ARRAY_PARTITION variable=line_buff complete dim=1

	TB_OUT win;
#pragma HLS ARRAY_PARTITION variable=win.data complete dim=0

	T_K key_read, key_write;

	for(T_BATCH b = 0; b < batch; b++){

		ap_uint<4> pointer = 0;
		bool out_ready = 0; 
		bool valid_out = 0; 
		bool read_flag = 0;
		int read_count = 0;
		
		for(ap_uint<16> bw = 0; bw < BUFFER_WIDTH; bw++){
#pragma HLS pipeline
			for(ap_uint<3> r = 0; r < BUFFER_ROWS; r++){
				line_buff[r][bw] = 0;
			}
		}

		for(ap_int<9> h = -1; h < Height; h++){
			for(ap_int<9> w = -1; w < Width; w++){
				valid_out = (h >= 0) && (w >= 0) && (h % 2 == 0) && (w % 2 == 0);
				read_flag = (h < Height - 1) && (w < Width - 1); 		
#pragma HLS PIPELINE II=1
				if (read_flag){
					TB_FPI f_read = fmap_in.read();
					
					T_FP4 f_pack = 0;
					for(T_C p4 = 0; p4 < P4; p4++){
#pragma HLS UNROLL
						f_pack.range((p4 + 1) * FW - 1, p4 * FW) = f_read.data[p4];
					}	
					line_buff[pointer][w + 1] = f_pack;
				}
				if(valid_out){
					for(ap_uint<3> ki = 0; ki < 3; ki++){
						for(ap_uint<3> kj = 0; kj < 3; kj++){
							if(ki - 1 + h < 0 || ki - 1 + h >= Height || kj - 1 + w < 0 || kj - 1 + w > Width){
								win.data[ki * 3 + kj] = 0; 
							}
							else{
								T_FPI out_pi = 0;
								out_pi.range(P4 * FW - 1, 0) = line_buff[(ki - 1 + h) % BUFFER_ROWS][w + kj - 1];
								win.data[ki * 3 + kj] = out_pi;
							}
						}
					}
					fmap_out.write(win);
				}
			}
			pointer++;
			if(pointer > BUFFER_ROWS - 1) pointer = 0; 
		}

	}
}



template<int PI, int BUFFER_WIDTH>
void conv_3x3_line_buffer(
	hls::stream<BundleT<PI, T_F> > &fmap_in,
	hls::stream<BundleT<9, ap_int<PI * FW> > > &fmap_out,
	T_H  Height,
    T_H  Width,
    T_C  IC,
	T_BATCH batch
){
	typedef ap_int<PI * FW> T_FPI;
	typedef BundleT<PI, T_F> TB_FPI;
	typedef BundleT<9, ap_int<PI * FW> > TB_OUT;

	const int BUFFER_ROWS = 3;
	T_FPI line_buff[BUFFER_ROWS][BUFFER_WIDTH];
#pragma HLS ARRAY_PARTITION variable=line_buff complete dim=1

	TB_OUT win;
#pragma HLS ARRAY_PARTITION variable=win.data complete dim=0

	int count = 0;

	for(T_BATCH b = 0; b < batch; b++){

		ap_uint<4> pointer = 0;
		bool out_ready = 0; 
		bool valid_out = 0; 
		bool read_flag = 0;
		int read_count = 0;
		
		for(int bw = 0; bw < BUFFER_WIDTH; bw++){
#pragma HLS pipeline
			for(int r = 0; r < BUFFER_ROWS; r++){
				line_buff[r][bw] = 0;
			}
		}

		for(ap_int<9> h = -1; h < Height; h++){
			for(ap_int<9> w = -1; w < Width; w++){
				valid_out = (h >= 0) && (w >= 0);
				read_flag = (h < Height - 1) && (w < Width - 1); 
			
				T_C ICPI = IC / PI;
				
				for(T_C ic = 0; ic < ICPI; ic++){ 	
#pragma HLS PIPELINE II=1
					if (read_flag){
						TB_FPI f_read = fmap_in.read();
						T_FPI f_pack = 0;
						for(T_C pi = 0; pi < PI; pi++){
#pragma HLS UNROLL
							f_pack.range((pi + 1) * FW - 1, pi * FW) = f_read.data[pi];
						}	
						line_buff[pointer][(w + 1) * ICPI + ic] = f_pack;
					}
					if(valid_out){
						for(ap_uint<3> ki = 0; ki < 3; ki++){
							for(ap_uint<3> kj = 0; kj < 3; kj++){
								if(ki - 1 + h < 0 || ki - 1 + h >= Height || kj - 1 + w < 0 || kj - 1 + w > Width){
									win.data[ki * 3 + kj] = 0; 
								}
								else{
									win.data[ki * 3 + kj] = line_buff[((ki - 1) + (h)) % BUFFER_ROWS][((w) + (kj - 1)) * ICPI + ic];
								}
							}
						}
						fmap_out.write(win);
					}
				}
			}
			pointer++;
			if(pointer > BUFFER_ROWS - 1) pointer = 0; 
		}
	}
}


template<int PI, int BUFFER_WIDTH>
void conv_3x3_line_buffer_stride(
	hls::stream<BundleT<PI, T_F> > &fmap_in,
	hls::stream<BundleT<9, ap_int<PI * FW> > > &fmap_out,
	T_H  Height,
    T_H  Width,
    T_C  IC,
	bool STRIDE,
	T_BATCH batch
){
	typedef ap_int<PI * FW> T_FPI;
	typedef BundleT<PI, T_F> TB_FPI;
	typedef BundleT<9, ap_int<PI * FW> > TB_OUT;

	ap_uint<2> stride = (STRIDE == 1) ? 2 : 1;

	const int BUFFER_ROWS = 3;
	T_FPI line_buff[BUFFER_ROWS][BUFFER_WIDTH];
#pragma HLS ARRAY_PARTITION variable=line_buff complete dim=1

	TB_OUT win;
#pragma HLS ARRAY_PARTITION variable=win.data complete dim=0


	for(T_BATCH b = 0; b < batch; b++){

		ap_uint<4> pointer = 0;
		bool out_ready = 0; 
		bool valid_out = 0; 
		bool read_flag = 0;
		int read_count = 0;
		
		for(int bw = 0; bw < BUFFER_WIDTH; bw++){
#pragma HLS pipeline
			for(int r = 0; r < BUFFER_ROWS; r++){
				line_buff[r][bw] = 0;
			}
		}

		for(ap_int<9> h = -1; h < Height; h++){
			for(ap_int<9> w = -1; w < Width; w++){
				valid_out = (h >= 0) && (w >= 0) && (h % stride == 0) && (w % stride == 0);
				read_flag = (h < Height - 1) && (w < Width - 1); 
			
				T_C ICPI = IC / PI;
				
				for(T_C ic = 0; ic < ICPI; ic++){ 	
#pragma HLS PIPELINE II=1
					if (read_flag){
						TB_FPI f_read = fmap_in.read();
						T_FPI f_pack = 0;
						for(T_C pi = 0; pi < PI; pi++){
#pragma HLS UNROLL
							f_pack.range((pi + 1) * FW - 1, pi * FW) = f_read.data[pi];
						}	
						line_buff[pointer][(w + 1) * ICPI + ic] = f_pack;
					}
					if(valid_out){
						for(ap_uint<3> ki = 0; ki < 3; ki++){
							for(ap_uint<3> kj = 0; kj < 3; kj++){
								if(ki - 1 + h < 0 || ki - 1 + h >= Height || kj - 1 + w < 0 || kj - 1 + w > Width){
									win.data[ki * 3 + kj] = 0; 
								}
								else{
									win.data[ki * 3 + kj] = line_buff[((ki - 1) + (h)) % BUFFER_ROWS][((w) + (kj - 1)) * ICPI + ic];
								}
							}
						}
						fmap_out.write(win);
					}
				}
			}
			pointer++;
			if(pointer > BUFFER_ROWS - 1) pointer = 0; 
		}
	}
}


template<int PI, int BUFFER_WIDTH>
void conv_3x3_line_buffer_wrap(
	hls::stream<BundleT<PI, T_F> > &fmap_in,
	hls::stream<BundleT<9, ap_int<PI * FW> > > &fmap_out,
	T_H  Height,
    T_H  Width,
    T_C  IC,
	bool STRIDE,
	T_BATCH batch,
	bool first_layer
){
    if (first_layer){
		line_buffer_first_layer<PI>(fmap_in, fmap_out, Height, Width, STRIDE, batch);
	}
	else{
		conv_3x3_line_buffer_stride<PI, 1024>(fmap_in, fmap_out, Height, Width, IC, STRIDE, batch);
	}
}

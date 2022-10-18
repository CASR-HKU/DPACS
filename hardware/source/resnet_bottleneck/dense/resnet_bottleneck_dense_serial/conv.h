template<int _W_1, int _W_2>
ap_int<_W_1 + _W_2> DSP_AM(ap_int<_W_1> in1, ap_int<_W_1> in2, ap_int<_W_2> in3){
#pragma HLS INLINE OFF

	ap_int<_W_1> add_temp = in1 + in2;
	ap_int<_W_1 + _W_2> mul_temp = add_temp * in3;
	return mul_temp;
}

// A small linebuffer of first layer with small ic
template<int PI>
void line_buffer_first_layer(
	hls::stream<BundleT<PI, T_F> > &fmap_in,
	hls::stream<BundleT<9, ap_int<P4 * FW> > > &fmap_out,
	T_H  Height,
    T_H  Width,
	bool STRIDE,
	T_BATCH batch
){

	typedef ap_int<PI * FW> T_FPI;
	typedef ap_int<P4 * FW> T_FP4;

	typedef BundleT<PI, T_F> TB_FPI;
	typedef BundleT<9, ap_int<P4 * FW> > TB_OUT;


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
								win.data[ki * 3 + kj] = line_buff[(ki - 1 + h) % BUFFER_ROWS][w + kj - 1];
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



template<int PI, int PO, int PF, int M_OC>
void first_layer(
	hls::stream<BundleT<9, ap_int<P4 * FW> > > &fmap_in,
	hls::stream<BundleT<PF, T_P> > &fmap_out,
	hls::stream<ap_int<P4 * WW> >  &w3,
	T_H Height,
	T_H Width,
	T_BATCH batch
){

	typedef BundleT<9, ap_int<P4 * FW> > TIN_FPI;
	typedef BundleT<9 * PF, ap_int<P4 * WW> > TB_W;
	typedef ap_int<P4 * WW> T_WPI;


	const int OC = 64;

	TIN_FPI win;
#pragma HLS ARRAY_PARTITION variable=win.data complete dim=0
	BundleT<PF, T_P > out;
#pragma HLS ARRAY_PARTITION variable=out.data complete dim=0
	TB_W w_vec;
#pragma HLS ARRAY_PARTITION variable=w_vec.data complete dim=0


	T_WPI  w_buffer[9][M_OC];
DO_PRAGMA(HLS ARRAY_PARTITION variable=w_buffer cyclic factor=PF/2 dim=2)
DO_PRAGMA(HLS ARRAY_PARTITION variable=w_buffer complete dim=1)


	int count = 0;

    int read_count = 0;
	for(T_C k = 0; k < 9; k++){
		for(T_C oc = 0; oc < OC; oc++){
#pragma HLS PIPELINE
           	w_buffer[k][oc]= w3.read();
		}
	}

	for(int rep = 0; rep < batch * Height * Width; rep++){		
		win = fmap_in.read();
		for(T_C oc = 0; oc < OC / PF; oc++){	
#pragma HLS PIPELINE II=1
			for(T_C po = 0; po < PF; po++){
				for(ap_uint<4> k = 0; k < 9; k++){
					w_vec.data[k + po * 9] = w_buffer[k][oc * PF + po];
				}
			}

			for(T_C po = 0; po < PF / 2; po++){
				T_P psum_0 = 0;
				T_P psum_1 = 0;

				for(T_C pi = 0; pi < P4; pi++){
					for(T_C k = 0; k < 9; k++){
						T_F in = (T_F) (win.data[k].range(FW * (pi + 1) - 1, FW * pi));
						ap_int<18> in_expend = (ap_int<18>) in;
						ap_int<27> w_pack = 0;

						T_W w_0 = (T_W) (w_vec.data[po * 18 + k].range((pi + 1) * WW - 1, pi * WW));
						T_W w_1 = (T_W) (w_vec.data[po * 18 + 9 + k].range((pi + 1) * WW - 1, pi * WW));
						ap_int<27> w_1_shift = 0;
						ap_int<27> w_0_expend = (ap_int<27>) w_0;
						w_1_shift.range(26,18) = (ap_int<9>) w_1;
						ap_int<48> mul_temp = DSP_AM(w_1_shift, w_0_expend, in_expend);
						ap_int<FW + WW> low = (ap_int<FW + WW>)(mul_temp.range(FW + WW - 1, 0));
						ap_int<FW + WW> high = (ap_int<FW + WW>)(mul_temp.range(FW + WW - 1 + 18, 18)) +  mul_temp.range(FW + WW - 1, FW + WW - 1);
						psum_0 += low;
						psum_1 += high;
					}
				}
				out.data[po * 2] = psum_0;
				out.data[po * 2 + 1] = psum_1;
			}
			fmap_out.write(out);
		}
	}
}


template<int PI, int PO, int M_IC, int M_OC>
void conv_3x3_double(
	hls::stream<BundleT<PI, T_F> > &fmap_in,
	hls::stream<BundleT<PO, T_P> > &fmap_out,
	hls::stream<ap_int<PI * WW> >  &w3,
	T_H Height,
	T_H Width,
	T_C IC,
	T_C OC,
	bool STRIDE,
	T_BATCH batch
){

	typedef BundleT<PI, T_F> TIN_FPI;
	typedef BundleT<PO, T_P > TOUT_FPO;
	typedef BundleT<PO, ap_int<PI * WW> > TB_W;
	typedef ap_int<PI * WW> T_WPI;

	TIN_FPI win;
#pragma HLS ARRAY_PARTITION variable=win.data complete dim=0
	TOUT_FPO out;
#pragma HLS ARRAY_PARTITION variable=out.data complete dim=0
	TB_W w_vec;
#pragma HLS ARRAY_PARTITION variable=w_vec.data complete dim=0

	T_P psum_buffer[M_OC];
DO_PRAGMA(HLS ARRAY_PARTITION variable=psum_buffer cyclic factor=PO dim=1)

	T_WPI  w_buffer[9][M_IC / PI][M_OC];
DO_PRAGMA(HLS ARRAY_PARTITION variable=w_buffer cyclic factor=PO/2 dim=3)


	ap_uint<2> stride = (STRIDE == 1) ? 2 : 1;
	T_H H_bound = (STRIDE == 1) ? Height >> 1 : Height;
	T_H W_bound = (STRIDE == 1) ? Width >> 1 :  Width;


	int count = 0;

    int read_count = 0;
	for(ap_uint<4> k = 0; k < 9; k++){
		for(T_C oc = 0; oc < OC; oc++){
			for(T_C ic = 0; ic < IC / PI; ic++){
#pragma HLS PIPELINE
           		w_buffer[k][ic][oc]= w3.read();
			}
		}
	}

	for(T_C oc = 0; oc < OC / PO; oc++){
#pragma HLS PIPELINE
		for(T_C po = 0; po < PO; po++){
			psum_buffer[po + oc * PO] = 0;
		}
	}

	for(int rep = 0; rep < batch * H_bound * W_bound; rep++){		
		for(ap_uint<4> k = 0; k < 9; k++){
			for(T_C ic = 0; ic < IC / PI; ic++){
				win = fmap_in.read();
				for(T_C oc = 0; oc < OC / PO; oc++){	
#pragma HLS PIPELINE II=1
					for(T_C po = 0; po < PO; po++){
						w_vec.data[po] = w_buffer[k][ic][oc * PO + po];
					}
					// pack 2 MAC on 1 dsp
					for(T_C po = 0; po < PO / 2; po++){
						T_P psum_0 = psum_buffer[oc * PO + po * 2];
						T_P psum_1 = psum_buffer[oc * PO + po * 2 + 1];

						for(T_C pi = 0; pi < PI; pi++){
							T_F in = (T_F) win.data[pi];
							ap_int<18> in_expend = (ap_int<18>) in;
							ap_int<27> w_pack = 0;

							T_W w_0 = (T_W) (w_vec.data[po * 2].range((pi + 1) * WW - 1, pi * WW));
							T_W w_1 = (T_W) (w_vec.data[po * 2 + 1].range((pi + 1) * WW - 1, pi * WW));
							ap_int<27> w_1_shift = 0;
							ap_int<27> w_0_expend = (ap_int<27>) w_0;
							w_1_shift.range(26,18) = (ap_int<9>) w_1;
							ap_int<48> mul_temp = DSP_AM(w_1_shift, w_0_expend, in_expend);
							ap_int<FW + WW> low = (ap_int<FW + WW>)(mul_temp.range(FW + WW - 1, 0));
							ap_int<FW + WW> high = (ap_int<FW + WW>)(mul_temp.range(FW + WW - 1 + 18, 18)) +  mul_temp.range(FW + WW - 1, FW + WW - 1);
							psum_0 += low;
							psum_1 += high;
							
						}
						psum_buffer[oc * PO + po * 2] = psum_0;
						psum_buffer[oc * PO + po * 2 + 1] = psum_1;
					}
				}
			}
		}
		for(T_C oc = 0; oc < OC / PO; oc++){	
#pragma HLS PIPELINE
			for(T_C po = 0; po < PO; po++){
				out.data[po] = psum_buffer[oc * PO + po];
				psum_buffer[oc * PO + po] = 0;
			}
			fmap_out.write(out);
		}
	}
}


template<int PI, int BUFFER_WIDTH>
void conv_3x3_line_buffer(
	hls::stream<BundleT<PI, T_F> > &fmap_in,
	hls::stream<BundleT<PI, T_F> > &fmap_out,
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
	TB_FPI output_pack;
	const int BUFFER_ROWS = 3;
	T_FPI line_buff[BUFFER_ROWS][BUFFER_WIDTH];
// #pragma HLS ARRAY_PARTITION variable=line_buff complete dim=1

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
				}
				if(valid_out){
					for(ap_uint<3> ki = 0; ki < 3; ki++){
						for(ap_uint<3> kj = 0; kj < 3; kj++){
								if(ki - 1 + h < 0 || ki - 1 + h >= Height || kj - 1 + w < 0 || kj - 1 + w > Width){
									for(T_C ic = 0; ic < ICPI; ic++){
										#pragma HLS PIPELINE II=1
										for(T_C pi = 0; pi < PI; pi++){
											output_pack.data[pi] = 0;
										}
										fmap_out.write(output_pack);
									}
								}
								else{
									for(T_C ic = 0; ic < ICPI; ic++){
										#pragma HLS PIPELINE II=1
										T_FPI output_read = line_buff[(ki - 1 + h) % BUFFER_ROWS][(w + kj - 1) * ICPI + ic];
										for(T_C pi = 0; pi < PI; pi++){
											output_pack.data[pi] = output_read.range((pi + 1) * FW - 1, pi * FW);
										}
										fmap_out.write(output_pack);
									}
								
							}
						}
					}
				}
			}
			pointer++;
			if(pointer > BUFFER_ROWS - 1) pointer = 0; 
		}
	}
}


template<int PI, int PO, int M_IC, int M_OC>
void conv1x1_dsp_double(
	hls::stream<BundleT<PI, T_F> > &fmap_in,
	hls::stream<BundleT<PO, T_P> > &fmap_out,
	hls::stream<ap_int<PI * WW> > &w,
	int REP,	
	int IC,
	int OC
){
    typedef ap_int<PI * FW> T_FPI;
    typedef ap_int<PI * WW> T_WPI;
    typedef BundleT<PI, T_F> TB_FPI;
    typedef BundleT<PO, T_P> TB_PPO;
	typedef BundleT<PO, T_WPI> TB_WPIPO;

	T_FPI in_buffer[M_IC / PI];
	T_WPI  w_buffer[M_OC][M_IC / PI];
	TB_PPO sum;
DO_PRAGMA( HLS ARRAY_PARTITION variable=w_buffer cyclic factor=PO/2 dim=1)
#pragma HLS ARRAY_PARTITION variable=sum.data complete dim=0
    TB_WPIPO w_vec;
#pragma HLS ARRAY_PARTITION variable=w_vec.data complete dim=0

	for(int p = 0; p < PO; p++){
#pragma HLS UNROLl
		sum.data[p] = 0;
	}

    int read_count = 0;
	for(T_C oc = 0; oc < OC; oc++){
		for(T_C ic = 0; ic < IC / PI; ic++){
#pragma HLS pipeline
            w_buffer[oc][ic]= w.read();
		}
	}

	for(int rep = 0; rep < REP; rep++){
		for(T_C cpi=0; cpi < IC / PI; cpi++){
#pragma HLS PIPELINE
			TB_FPI f_read = fmap_in.read();
			T_FPI f_tmp = 0;
			for(int pi = 0; pi < PI; pi++){
#pragma HLS UNROLL
				f_tmp.range((pi + 1) * FW - 1, pi * FW) = (ap_int<FW>) f_read.data[pi];
			}
			in_buffer[cpi] = f_tmp;
		}

		for(T_C k = 0; k < OC / PO; k++){
			for(T_C cpi=0; cpi < IC / PI; cpi++){
#pragma HLS PIPELINE
				T_FPI in_pi = in_buffer[cpi];
				for(T_C po = 0; po < PO; po++){
					w_vec.data[po] = w_buffer[k * PO + po][cpi];
				}

				// pack 2 MAC on 1 DSP
    			for(T_C po = 0; po < PO / 2; po++){
#pragma HLS UNROLl
					for(T_C pi = 0; pi < PI; pi++){
#pragma HLS UNROLl
						T_F in = (T_F) in_pi.range(FW * (pi + 1) -1, FW * pi);
						ap_int<18> in_expend = (ap_int<18>) in;

						ap_int<27> w_pack = 0;
						T_W w_0 = (T_W) (w_vec.data[po * 2].range((pi + 1) * WW - 1, pi * WW));
						T_W w_1 = (T_W) (w_vec.data[po * 2 + 1].range((pi + 1) * WW - 1, pi * WW));

						ap_int<27> w_1_shift = 0;
						ap_int<27> w_0_expend = (ap_int<27>) w_0;
						w_1_shift.range(26,18) = (ap_int<9>) w_1;

						ap_int<48> mul_temp = DSP_AM(w_1_shift, w_0_expend, in_expend);
						ap_int<FW + WW> low = (ap_int<FW + WW>)(mul_temp.range(FW + WW - 1, 0));
						ap_int<FW + WW> high = (ap_int<FW + WW>)(mul_temp.range(FW + WW - 1 + 18, 18)) +  mul_temp.range(FW + WW - 1, FW + WW - 1);
						sum.data[po * 2] += low;
						sum.data[po * 2 + 1] += high;
					}
				}
			}
			fmap_out<<sum;
			for(T_C p = 0; p < PO; p++){
#pragma HLS UNROLL
				sum.data[p]=0;
			}
		}
	}
}



/*
	Quantization function. Currently we simply use shift operation to quantize psum to 8-bit.
	Since the  quanzation method can be ad-hoc, you can modify this function to fit your quantization algorithm.
*/
template<int PO>
void quantize_shift(
    hls::stream<BundleT<PO, T_P> > &pin,
	hls::stream<BundleT<PO, T_F> > &fout,
    T_Q    *Q,
	int    rep,
	T_C    C,
	bool   relu
	){

    typedef BundleT<PO, T_P> T_PPO;
    typedef BundleT<PO, T_F> T_FPO;

	int count = 0;
	const int upper = (1 << (FW - 1)) - 1;
	const int lower = -(1 << (FW - 1));
	const int neg_shift = 1 << (SW - 1);

	for(int r = 0; r < rep; r++){
		for(int c = 0; c < C / PO; c++){
#pragma HLS PIPELINE II=1

			T_PPO p_read = pin.read();
			T_FPO out;
#pragma HLS ARRAY_PARTITION variable=p_read.data complete dim=0
#pragma HLS ARRAY_PARTITION variable=out.data complete dim=0

			for(int po = 0; po < PO; po++){
#pragma HLS UNROLL
				T_P psum = p_read.data[po];
				psum =  (psum >> 10);
				if(relu && (psum < 0)) psum = 0;
				if (psum > 127) psum = 127;
				else if (psum < -128) psum = -128; 
				out.data[po] = (T_F) psum;
			}
			fout.write(out);
		}
	}
}

// Quantization function with identity add
template<int PO>
void quantize_shift_res(
    hls::stream<BundleT<PO, T_P> > &pin,
	hls::stream<BundleT<PO, T_F> > &fres,
	hls::stream<BundleT<PO, T_F> > &fout,
    T_Q    *Q,
	int    rep,
	T_C    C,
	bool   relu,
	bool   residual
	){

    typedef BundleT<PO, T_P> T_PPO;
    typedef BundleT<PO, T_F> T_FPO;

	int count = 0;
	const int upper = (1 << (FW - 1)) - 1;
	const int lower = -(1 << (FW - 1));
	const int neg_shift = 1 << (SW - 1);

	T_FPO res_read;

	for(int r = 0; r < rep; r++){
		for(int c = 0; c < C / PO; c++){
#pragma HLS PIPELINE II=1

			T_PPO p_read = pin.read();
			if (residual) res_read = fres.read();
			T_FPO out;
#pragma HLS ARRAY_PARTITION variable=p_read.data complete dim=0
#pragma HLS ARRAY_PARTITION variable=out.data complete dim=0

			for(int po = 0; po < PO; po++){
#pragma HLS UNROLL
				T_P psum = p_read.data[po];
				psum =  (psum >> 10);
				if (residual) psum += res_read.data[po];
				if (psum > 127) psum = 127;
				else if (psum < -128) psum = -128; 
				if(relu && (psum < 0)) psum = 0;
				T_F quant = (T_F) psum;
				out.data[po] = quant;
			}
			fout.write(out);
		}
	}
}
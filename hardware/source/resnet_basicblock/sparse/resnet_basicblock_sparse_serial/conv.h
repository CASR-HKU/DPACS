template<int _W_1, int _W_2>
ap_int<_W_1 + _W_2> DSP_AM(ap_int<_W_1> in1, ap_int<_W_1> in2, ap_int<_W_2> in3){
#pragma HLS INLINE OFF

	ap_int<_W_1> add_temp = in1 + in2;
	ap_int<_W_1 + _W_2> mul_temp = add_temp * in3;
	return mul_temp;
}


template<int PI, int PF, int M_OC>
void first_layer(
	hls::stream<BundleT<9, ap_int<P4 * FW> > > &fmap_in,
	hls::stream<BundleT<PF, T_P> > &fmap_out,
	hls::stream<T_K > &out_key,
	hls::stream<ap_int<P4 * WW> > &w,
	T_H  Height,
	T_H  Width
){


	typedef BundleT<9, ap_int<P4 * FW> > TIN_FPI;
	typedef BundleT<PF, T_P > TOUT_FPO;
	typedef BundleT<9 * PF, ap_int<P4 * WW> > TB_W;
	typedef ap_int<P4 * WW> T_WPI;

	TIN_FPI win;
#pragma HLS ARRAY_PARTITION variable=win.data complete dim=0
	BundleT<PF, T_P> out;
#pragma HLS ARRAY_PARTITION variable=out.data complete dim=0
	TB_W w_vec;
#pragma HLS ARRAY_PARTITION variable=w_vec.data complete dim=0


	T_WPI  w_buffer[9][M_OC];
DO_PRAGMA(HLS ARRAY_PARTITION variable=w_buffer cyclic factor=PF/2 dim=2)
DO_PRAGMA(HLS ARRAY_PARTITION variable=w_buffer complete dim=1)

	int count = 0;

    int read_count = 0;
	T_C pi_factor, po_factor, ic_ceil, oc_ceil;

	T_K key;

	const T_H IC=64, OC=64;

	for(T_C k = 0; k < 9; k++){
		for(T_C oc = 0; oc < OC; oc++){
#pragma HLS PIPELINE II=1
			w_buffer[k][oc]= w.read();
		}
	}

	for(ap_int<9> h = 0; h < Height; h++){
		for(ap_int<9> w = 0; w < Width; w++){
			key.x = w; key.y = h; key.end = 0;
			out_key.write(key);

			win = fmap_in.read();
			for(ap_uint<6> oc = 0; oc < OC / PF; oc++){	
#pragma HLS PIPELINE II=1
				for(ap_uint<6> po = 0; po < PF; po++){
					for(ap_uint<4> k = 0; k < 9; k++){
						w_vec.data[k + po * 9] = w_buffer[k][oc * PF + po];
					}
				}
				for(ap_uint<6> po = 0; po < PF / 2; po++){
					T_P psum_0 = 0;
					T_P psum_1 = 0;

					for(ap_uint<3> pi = 0; pi < P4; pi++){
						for(ap_uint<4> k = 0; k < 9; k++){
							T_F in = (T_F) win.data[k].range(FW * (pi + 1) -1, FW * pi);
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
	key.end = 1;
	out_key.write(key);

}

template<int PI, int PO, int M_IC, int M_OC>
void conv_3x3_double_serial(
	hls::stream<BundleT<PI, T_F> > &fmap_in,
	hls::stream<BundleT<PO, T_P> > &fmap_out,
	hls::stream<T_OFFSET> &offset_s,
	hls::stream<T_K > &in_key,
	hls::stream<T_K > &out_key,
	hls::stream<ap_int<PI * WW> > &w,
	T_C IC,
	T_C OC,
	bool use_cprune
){

	typedef BundleT<PI, T_F > TIN_FPI;
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

	int count = 0;

    int read_count = 0;
	T_C pi_factor, po_factor, ic_ceil, oc_ceil;
	if(use_cprune){
		pi_factor = CPRUNE_FACTOR / PI;
		po_factor = CPRUNE_FACTOR;
		ic_ceil = ceil_div<CPRUNE_FACTOR>(IC);
		oc_ceil = ceil_div<CPRUNE_FACTOR>(OC);
	}
	else{
		pi_factor = 1;
		po_factor = PO;
		ic_ceil   = IC / PI;
		oc_ceil   = OC / PO;
	}

	for(T_C k = 0; k < 9; k++){
		for(T_C oc = 0; oc < oc_ceil; oc++){
			for(T_C ic = 0; ic < ic_ceil; ic++){
				for(T_C po = 0; po < po_factor; po++){
					for(T_C pi = 0; pi < pi_factor; pi++){
#pragma HLS PIPELINE II=1
						w_buffer[k][ic * pi_factor + pi][oc * po_factor + po]= w.read();
					}
				}
			}
		}
	}



	for(T_C oc = 0; oc < OC / PO; oc++){
#pragma HLS PIPELINE
		for(ap_uint<7> po = 0; po < PO; po++){
			psum_buffer[po + oc * PO] = 0;
		}
	}

	for(ap_uint<19> rep = 0; rep < MAX_H * MAX_H; rep++){
		
		T_K key = in_key.read();
		out_key.write(key);
		if (key.end == 1) break;
		for(T_C ki = 0; ki < 10; ki++){
			T_OFFSET k = offset_s.read();
			
			if (k == end_3x3) break;
			

			for(T_C ic = 0; ic < IC / PI; ic++){
				win = fmap_in.read();
				for(T_C oc = 0; oc < OC / PO; oc++){	
#pragma HLS PIPELINE II=1
					for(ap_uint<7> po = 0; po < PO; po++){
						w_vec.data[po] = w_buffer[k][ic][oc * PO + po];
					}

					// pack 2 MAC on 1 DSP
					for(ap_uint<7> po = 0; po < PO / 2; po++){
						T_P psum_0 = psum_buffer[oc * PO + po * 2];
						T_P psum_1 = psum_buffer[oc * PO + po * 2 + 1];

						for(ap_uint<7> pi = 0; pi < PI; pi++){
							
								T_F in = (T_F) win.data[pi];
								ap_int<18> in_expend = (ap_int<18>) in;
								ap_int<27> w_pack = 0;

								T_W w_0 = (T_W) (w_vec.data[po * 2].range((pi + 1) * WW - 1, pi * WW));
								T_W w_1 = (T_W) (w_vec.data[po * 2 + 1].range((pi + 1) * WW - 1, pi * WW));
								ap_int<27> w_1_shift = 0;
								ap_int<27> w_0_expend = (ap_int<27>) w_0;
								w_1_shift.range(26,18) = (ap_int<9>) w_1;

								// cout<<"w_0:"<<w_0<<" w_1:"<<w_1<<endl;;
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
			for(ap_uint<7> po = 0; po < PO; po++){
				out.data[po] = psum_buffer[oc * PO + po];
				psum_buffer[oc * PO + po] = 0;
			}
			fmap_out.write(out);
		}
	}
	
}

template<int PI, int PO, int M_IC, int M_OC>
void conv1x1_dsp_double(
		hls::stream<BundleT<PI, T_F> > &fmap_in,
		hls::stream<BundleT<PO, T_P> > &fmap_out,
		hls::stream<T_K > &in_key,
		hls::stream<T_K > &out_key,
		hls::stream<ap_int<PI * WW> > &w,
		int IC,
		int OC,
		bool use_cprune
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


    int read_count = 0;
	T_C pi_factor, po_factor, ic_ceil, oc_ceil;
	if(use_cprune){
		pi_factor = CPRUNE_FACTOR / PI;
		po_factor = CPRUNE_FACTOR;
		ic_ceil = ceil_div<CPRUNE_FACTOR>(IC);
		oc_ceil = ceil_div<CPRUNE_FACTOR>(OC);
	}
	else{
		pi_factor = 1;
		po_factor = PO;
		ic_ceil   = IC / PI;
		oc_ceil   = OC / PO;
	}

	for(T_C oc = 0; oc < oc_ceil; oc++){
		for(T_C ic = 0; ic < ic_ceil; ic++){
			for(T_C po = 0; po < po_factor; po++){
				for(T_C pi = 0; pi < pi_factor; pi++){
#pragma HLS PIPELINE II=1
					w_buffer[oc * po_factor + po][ic * pi_factor + pi]= w.read();
				}
			}
		}
	}


	for(T_C p = 0; p < PO; p++){
#pragma HLS UNROLl
		sum.data[p] = 0;
	}
	for(ap_uint<19> rep = 0; rep < MAX_H * MAX_H; rep++){
		T_K key = in_key.read();
		out_key.write(key);

		if(key.end == 1) break;


		for(T_C cpi=0; cpi < IC / PI; cpi++){
#pragma HLS PIPELINE
			TB_FPI f_read = fmap_in.read();
			T_FPI f_tmp = 0;
			for(ap_uint<7> pi = 0; pi < PI; pi++){
#pragma HLS UNROLL
				f_tmp.range((pi + 1) * FW - 1, pi * FW) = (ap_int<FW>) f_read.data[pi];
			}
			in_buffer[cpi] = f_tmp;
		}

		for(T_C k = 0; k < OC / PO; k++){
			for(T_C cpi=0; cpi < IC / PI; cpi++){
#pragma HLS PIPELINE
				T_FPI in_pi = in_buffer[cpi];
				for(ap_uint<7> po = 0; po < PO; po++){
					w_vec.data[po] = w_buffer[k * PO + po][cpi];
				}

				// pack 2 MAC on 1 DSP
				for(ap_uint<7> po = 0; po < PO / 2; po++){
#pragma HLS UNROLl
					for(ap_uint<7> pi = 0; pi < PI; pi++){
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
			for(ap_uint<7> p = 0; p < PO; p++){
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
	hls::stream<T_K > &in_key,
	hls::stream<T_K > &out_key,
    T_Q    *Q,
	T_C    C,
	bool   relu
	){

    typedef BundleT<PO, T_P> T_PPO;
    typedef BundleT<PO, T_F> T_FPO;

	int count = 0;
	T_F quant;

	T_PPO p_read;
	T_FPO out;
#pragma HLS ARRAY_PARTITION variable=p_read.data complete dim=0
#pragma HLS ARRAY_PARTITION variable=out.data complete dim=0



	for(int r = 0; r < MAX_H * MAX_H; r++){

		T_K key = in_key.read();
		out_key.write(key);
		if (key.end == 1) break;

		for(T_C c = 0; c < C / PO; c++){
#pragma HLS PIPELINE II=1
			p_read = pin.read();

			for(ap_uint<7> po = 0; po < PO; po++){
#pragma HLS UNROLL
				T_P psum = p_read.data[po];
				psum = psum >> 10;
				if(relu && (psum < 0)) psum = 0;
				if (psum > 127) psum = 127;
				else if (psum < -128) psum = -128; 
				out.data[po] = (T_F) psum;
			}
			fout.write(out);
		}
	}
	
}

/*
	Quantization function and identity add.
*/
template<int PO>
void quantize_shift_res(
    hls::stream<BundleT<PO, T_P> > &pin,
	hls::stream<BundleT<PO, T_F> > &fout,
	hls::stream<BundleT<PO, T_F> > &fres,
	hls::stream<T_K > &in_conv_key,
	hls::stream<T_K > &out_key,
    T_Q    *Q,
	T_C    C,
	bool   relu,
	bool   residual
	){

    typedef BundleT<PO, T_P> T_PPO;
    typedef BundleT<PO, T_F> T_FPO;

	int count = 0;
	T_PPO p_read;
	T_FPO out, res_read;
#pragma HLS ARRAY_PARTITION variable=p_read.data complete dim=0
#pragma HLS ARRAY_PARTITION variable=out.data complete dim=0
#pragma HLS ARRAY_PARTITION variable=res_read.data complete dim=0

	T_K	key_res, key;
	T_F quant;
	for(int r = 0; r < MAX_H * MAX_H; r++){
		key = in_conv_key.read();
		out_key.write(key);
		if (key.end == 1) break;

		for(T_C c = 0; c < C / PO; c++){
#pragma HLS PIPELINE II=1
			if (residual) res_read = fres.read();
			p_read = pin.read();

			for(ap_uint<7> po = 0; po < PO; po++){
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

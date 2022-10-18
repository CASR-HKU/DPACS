
template<int PO, int M_IC, int M_OC>
void max_pool(
    hls::stream<BundleT<PO, T_F> > &f_in,
	hls::stream<T_K > &in_key,
	hls::stream<ap_int<WW * PO> > &fc_s,
	ap_int<PW> *cmask_out,
	bool enable_pool,
	int IC,
	int OC
){

	typedef BundleT<PO, T_F> T_BPO;
    typedef ap_int<PO * FW> T_PO;
	typedef ap_int<WW * PO> T_POW;

	T_BPO f_read;
#pragma HLS ARRAY_PARTITION variable=f_read.data complete dim=0

	T_PO pool_buff[M_IC / PO];
	T_POW fc_buff[M_OC / CPRUNE_FACTOR][M_IC / PO];
	T_PO max_po;

	


	if(enable_pool){
		int read_count = 0;
		for(T_C oc = 0; oc < OC / CPRUNE_FACTOR; oc++){
			for(T_C ic = 0; ic < IC / PO; ic++){
#pragma HLS PIPELINE II=1
				fc_buff[oc][ic] = fc_s.read();
				read_count++;
			}
		}
		// cout<<"fc read weights:"<<read_count<<endl;

		for(T_C i = 0; i < M_IC / PO; i++){
#pragma HLS PIPELINE II=1
			for(T_C j = 0; j < PO; j++){
#pragma HLS UNROLL
				pool_buff[i].range((j + 1) * FW - 1, j * FW) = -128;
			}
		}		
		for(ap_uint<32> r = 0; r < MAX_H * MAX_H; r++){

			T_K key = in_key.read();
			if (key.end == 1) break;
			for(T_C c = 0; c < IC / PO; c++){
#pragma HLS PIPELINE II=1
				f_read = f_in.read();
				max_po = pool_buff[c];
				for(T_C po = 0; po < PO; po++){
#pragma HLS UNROLL
					T_F new_f = f_read.data[po];
					T_F current_max = max_po.range((po + 1) * FW - 1, po * FW);
					if(new_f > current_max){
						max_po.range((po + 1) * FW - 1, po * FW) = new_f;
					}
				}
				pool_buff[c] = max_po;
			}
		}

		for(T_C oc = 0; oc < OC / CPRUNE_FACTOR; oc++){
			T_P psum = 0;
			for(T_C ic = 0; ic < IC / PO; ic++){
#pragma HLS PIPELINE II=1
				T_POW weight = fc_buff[oc][ic];
				T_PO f_pool = pool_buff[ic];
				for(T_C po = 0; po < PO; po ++){
#pragma HLS UNROLL
					psum += (T_F) f_pool.range((po + 1) * FW - 1, po * FW) * (T_W) weight.range((po + 1) * WW - 1, po * WW);
				}
			}
			cmask_out[oc] = psum;
		}
	}
	
}
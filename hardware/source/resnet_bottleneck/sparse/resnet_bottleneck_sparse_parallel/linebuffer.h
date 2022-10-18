

// Small linebuffer optimized for first layer
template<int PI>
void line_buffer_first_layer(
	hls::stream<BundleT<PI, T_F> > &fmap_in,
	hls::stream<BundleT<9, ap_int<PI * FW> > > &fmap_out,
	hls::stream<T_K > &in_key,
	hls::stream<T_K > &out_key,
	T_H  Height,
    T_H  Width
){

	typedef ap_int<PI * FW> T_FPI;
	typedef ap_int<P4 * FW> T_FP4;

	typedef BundleT<PI, T_F> TB_FPI;
	typedef BundleT<9, ap_int<PI * FW> > TB_OUT;

	const int BUFFER_ROWS = 3;
	const int BUFFER_WIDTH = 256;

	T_FP4 line_buff[BUFFER_ROWS][BUFFER_WIDTH];
#pragma HLS ARRAY_PARTITION variable=line_buff complete dim=1

	TB_OUT win;
#pragma HLS ARRAY_PARTITION variable=win.data complete dim=0

	T_K key_read, key_write;

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
                key_read = in_key.read();
                TB_FPI f_read = fmap_in.read();
                
                T_FP4 f_pack = 0;
                for(T_C p4 = 0; p4 < P4; p4++){
#pragma HLS UNROLL
                    f_pack.range((p4 + 1) * FW - 1, p4 * FW) = f_read.data[p4];
                }	
                line_buff[pointer][w + 1] = f_pack;
            }
            if(valid_out){
                key_write.x = w >> 1;
                key_write.y = h >> 1;
                key_write.end = 0;
                out_key.write(key_write);

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

    key_write = in_key.read();
    key_write.end = 1;
    out_key.write(key_write);

}



// Elastic Linebuffer with identical input and output non-zero index
template<int PI, int BUFFER_WIDTH>
void conv_3x3_line_buffer_residual_stride(
	hls::stream<BundleT<PI, T_F> > &fmap_in,
	hls::stream<BundleT<9, ap_int<PI * FW> > > &fmap_out,
	hls::stream<T_K > &in_key,
	hls::stream<T_K > &out_key,
	hls::stream<T_K > &res_key,
	T_H  Height,
    T_H  Width,
    T_C  IC,
	bool STRIDE2,
	bool residual
){
	typedef ap_int<PI * FW> T_FPI;
	typedef BundleT<PI, T_F> TB_FPI;
	typedef BundleT<9, ap_int<PI * FW> > TB_OUT;

	const int BUFFER_ROWS = 3;
	T_FPI line_buff[BUFFER_ROWS][BUFFER_WIDTH];
#pragma HLS ARRAY_PARTITION variable=line_buff complete dim=1

	bool valid[BUFFER_ROWS][MAX_H];
#pragma HLS ARRAY_RESHAPE variable=valid complete dim=2

	bool win_valid[3][3];
#pragma HLS ARRAY_PARTITION variable=win_valid complete dim=0
	
	TB_OUT win;
#pragma HLS ARRAY_PARTITION variable=win.data complete dim=0

	const int FIFO_DEPTH = MAX_H * (BUFFER_ROWS);
	T_K key_fifo[FIFO_DEPTH];
	const T_K empty_key = {.end = 0, .x = 0, .y = 0};
	ap_uint<2> stride = (STRIDE2 == 1) ? 2 : 1;

    ap_uint<10> start_pointer = 0, end_pointer = 0;  
    bool fifo_full = 0, fifo_empty = 1, read_end = 0, padding = 0, invalid = 0;
    int jump_line_diff = 0, read_count = 0;
    bool read_enable = 1, read_ready = 0, out_enable = 0, out_ready = 0, out_empty_check = 0;
    T_H x_anchor=0, y_anchor=0;
    T_C ICPI = IC / PI;
    T_K new_key = {.end = 0, .x = 0, .y = 0}, old_key = {.end = 0, .x = 0, .y = 0}, key_stride = {.end = 0, .x = 0, .y = 0};
    T_H x = 0, y = 0;
    
    for(int i = 0; i < FIFO_DEPTH; i++){
#pragma HLS PIPELINE
        key_fifo[i] = empty_key;
    }

    for(ap_uint<4> r = 0; r < 3; r++){
#pragma HLS UNROLL
        for(ap_uint<4> h = 0; h < 3; h++){
#pragma HLS UNROLL
            win_valid[r][h] = 0;
        }
    }


    for(ap_uint<4> r = 0; r < BUFFER_ROWS; r++){
#pragma HLS UNROLL
        for(T_H h = 0; h < MAX_H; h++){
#pragma HLS UNROLL
            valid[r][h] = 0;
        }
    }

    for(int h = 0; h < BUFFER_WIDTH; h++){
#pragma HLS PIPELINE
        for(ap_uint<4> r = 0; r < BUFFER_ROWS; r++){
#pragma HLS UNROLL
            line_buff[r][h] = 0;
        }
    }


    for(ap_uint<20> rep = 0; rep < (MAX_H + 2) * (MAX_H + 2); rep++){

        if(read_enable){ 
            T_K key_read = in_key.read();
            jump_line_diff = key_read.y - new_key.y;
            new_key = key_read;
            key_fifo[end_pointer] = new_key;
            end_pointer = (end_pointer + 1) % FIFO_DEPTH;
        }

        fifo_full = ((end_pointer - start_pointer) == (FIFO_DEPTH - 1)) || ((end_pointer - start_pointer) == -1);
        fifo_empty = (end_pointer == start_pointer);
        read_end = new_key.end;
        
        // peak oldest index
        if (STRIDE2){
            old_key.x = x_anchor;
            old_key.y = y_anchor;
            old_key.end = (y_anchor > Height) ? 1 : 0;
        }
        else{
            if(!fifo_empty) old_key = key_fifo[start_pointer];
        }
        
        
        out_ready = (((new_key.y - old_key.y) == 1) && ((new_key.x - old_key.x) >= 2)) || ((new_key.y - old_key.y) >= 2) || read_end; //to be more general
        read_ready = !read_end && (((new_key.y - old_key.y) <= 1) || fifo_empty);
        read_enable = read_ready && !fifo_full;
        out_enable = out_ready;

        // update bitmap of the buffer
        if(read_enable){
            jump_line_diff = jump_line_diff > 2 ? 2 : jump_line_diff;
            for(int l = 0; l < jump_line_diff; l++){
#pragma HLS PIPELINE II=1
                for(int k = 0; k < MAX_H; k++){
#pragma HLS UNROLL
                    valid[(new_key.y - l) % BUFFER_ROWS][k] = 0;
                }
            }
            valid[new_key.y % BUFFER_ROWS][new_key.x] = 1;
        }

        
        out_empty_check = 0;
        for(ap_uint<4> ki = 0; ki < 3; ki++){
#pragma HLS UNROLL
            for(ap_uint<4> kj = 0; kj < 3; kj++){
#pragma HLS UNROLL
                padding = (ki - 1 + old_key.y < 0) || (ki - 1 + old_key.y >= Height) || (kj - 1 + old_key.x < 0) || (kj - 1 + old_key.x > Width);
                if(!padding) invalid = valid[(ki - 1 + old_key.y) % BUFFER_ROWS][old_key.x + kj - 1] == 0;
                if(padding || invalid){
                    win_valid[ki][kj] = 0; 
                }
                else{
                    win_valid[ki][kj] = 1;
                }
                out_empty_check = out_empty_check || win_valid[ki][kj];
            }
        }

        out_enable = out_enable && out_empty_check;
        out_enable = out_enable || old_key.end;

        // pop oldest index
        if(out_enable) {
            if(residual) res_key.write(old_key);
            key_stride.x = (STRIDE2) ? old_key.x >> 1: old_key.x;
            key_stride.y = (STRIDE2) ? old_key.y >> 1: old_key.y;
            key_stride.end = old_key.end;
            out_key.write(key_stride);
        }

        if(out_enable && key_stride.end == 1) break;
        

        for(T_C ic = 0; ic < ICPI; ic++){ 
#pragma HLS PIPELINE II=1
            if(read_enable){
                TB_FPI f_read = fmap_in.read();
                T_FPI f_pack = 0;
                for(T_C pi = 0; pi < PI; pi++){
                    f_pack.range((pi + 1) * FW - 1, pi * FW) = f_read.data[pi];
                }	
                line_buff[new_key.y % BUFFER_ROWS][new_key.x * ICPI + ic] = f_pack;		
            }

            if(out_enable){
                T_H x = old_key.x;
                T_H y = old_key.y;
                for(ap_uint<4> ki = 0; ki < 3; ki++){
                    for(ap_uint<4> kj = 0; kj < 3; kj++){
                        if(win_valid[ki][kj] == 0){
                            win.data[ki * 3 + kj] = 0; 
                        }
                        else{
                            win.data[ki * 3 + kj] = line_buff[(ki - 1 + y) % BUFFER_ROWS][(x + kj - 1) * ICPI + ic];
                        }
                    }
                }
                fmap_out.write(win);
            }
        }
        bool pop_enable = STRIDE2 ? read_enable : (!fifo_empty && out_enable);
        if (pop_enable) start_pointer = (start_pointer + 1) % FIFO_DEPTH;
        if(out_ready){
            x_anchor += 2;
            if(x_anchor >= Width){
                x_anchor = 0;
                y_anchor += 2;
            }
        }
    }
	
}
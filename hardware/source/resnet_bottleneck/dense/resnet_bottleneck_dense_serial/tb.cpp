#include "top.h"
#include "tb_func.h"

int main(){
    // freopen("../../../../log.txt","w",stdout);
    static ap_int<FW * PI_0> fin[65536];
    static ap_int<FW * PO_1> fout[65536];
    static ap_int<W_FACTOR * WW> weight_buffer[65536];

    ap_uint<16> flags = 0;
    int H, W, IC_0, OC_0, OC_3, OC_1, batch;
    bool skip_0, skip_3, skip_1, relu, residual, stride2, first_layer, relu_0, relu_3, relu_1;
    int w_index = 0, count = 0; 
    bool pass;

    cout<<"-------------start testing dense input--------------"<<endl;
    H=8, W=8, IC_0=128, OC_0=128, OC_3=128, OC_1=128, batch=1;
    skip_0=0, skip_3=0, skip_1=0, residual=1, stride2=0, first_layer=0;
    relu_0 = 1, relu_3 = 1, relu_1 = 1;


    flags = 0;
    flags[0] = skip_0;
    flags[1] = skip_3;
    flags[2] = skip_1;
    flags[3] = stride2;     
    flags[4] = residual;    
    flags[5] = first_layer;     
    flags[6] = relu_0;      
    flags[7] = relu_3;      
    flags[8] = relu_1; 
    cout<<"flags: "<<flags<<endl;

    w_index = 0;
    readfile<FW, PI_0>(fin, batch * H * W * IC_0, "TXT_FILES_dense_input.txt");
    if (first_layer)
        w_index = read_file_static<WW, W_FACTOR>(weight_buffer,  IC_0 * P4 * 9, "TXT_FILES_w3_dense_layout.txt", w_index);
    if (!skip_0)
        w_index = read_file_static<WW, W_FACTOR>(weight_buffer,  IC_0 * OC_0, "TXT_FILES_w0_dense_layout.txt", w_index);
    if (!skip_3)
	    w_index = read_file_static<WW, W_FACTOR>(weight_buffer,  OC_0 * 9 * OC_3, "TXT_FILES_w3_dense_layout.txt", w_index);
    if (!skip_1)
        w_index = read_file_static<WW, W_FACTOR>(weight_buffer,  OC_3 * OC_1, "TXT_FILES_w1_dense_layout.txt", w_index);

    top(fin, fout, fin, weight_buffer, H, W, IC_0, OC_0, OC_3, OC_1, flags, batch);
        
    FILE *f_gt;
    f_gt = fopen("TXT_FILES_dense_out_1.txt", "r");
    cout<<"Comparing with TXT_FILES_dense_out_1.txt"<<endl;

    count = 0;  pass = 1;
    for (int i = 0; i < batch * H * W * OC_1 / PO_1; i++){
        ap_uint<FW * PO_1> rd = fout[i];
        for(int j = 0; j < PO_1; j++){
            int out_read = (T_F) (rd.range((j + 1) * FW - 1, j * FW));
            int tmp;
		    fscanf(f_gt, "%d", &tmp);
            if (tmp != out_read){
                cout<<"failed at:"<<count<<" gt:"<<tmp<<" out:"<<out_read<<endl;
                pass = 0;
            }
            count++;
   	    }
    }

    if(pass) cout<<"Passed dense tb!"<<endl; 
    fclose(f_gt);
    cout<<"-------------end testing dense input--------------\n\n";

}

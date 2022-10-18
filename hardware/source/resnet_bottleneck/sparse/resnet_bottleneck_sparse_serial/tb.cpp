#include "top.h"
#include "tb_func.h"

int main(){

    // freopen("../../../../log.txt","w",stdout);
    static ap_int<FW * PI_0> fin[65536];
    static ap_int<FW * PO_1> identity[65536];
    static ap_int<W_FACTOR * WW> weight_buffer[65536];
    static ap_int<PW>  cmask_out[65536];
    ap_uint<16> flags = 0;
    int H, W, IC_0, OC_0, OC_3, OC_1, next_c, batch;
    bool skip_0, skip_3, skip_1, relu, residual, stride2, use_mask, enable_pool, use_cprune, first_layer, return_mask, relu_0, relu_3, relu_1;
    int w_index = 0, count = 0; 
    bool pass;
    ap_uint<8> cmask_ic = 0, cmask_oc = 0;

  
    cout<<"-------------start testing dense input--------------"<<endl;
    H=8, W=8, IC_0=128, OC_0=128, OC_3=128, OC_1=128, next_c=128, batch=1;
    skip_0=0, skip_3=0, skip_1=0, residual=1, stride2=0, use_mask=1, enable_pool=0, use_cprune=0, first_layer=0, return_mask=1;
    relu_0 = 1, relu_3 = 1, relu_1 = 1;


    flags = 0;
    flags[0] = skip_0;
    flags[1] = skip_3;
    flags[2] = skip_1;
    flags[3] = enable_pool; 
    flags[4] = stride2;     
    flags[5] = residual;    
    flags[6] = use_mask;    
    flags[7] = use_cprune;
    flags[8] = first_layer;     
    flags[9] = return_mask;  
    flags[10] = relu_0;      
    flags[11] = relu_3;      
    flags[12] = relu_1;   
    cout<<"flags: "<<flags<<endl;

    w_index = 0;
    readfile<FW, PI_0>(fin, batch * H * W * IC_0, "TXT_FILES_dense_input.txt");
    readfile<FW, PI_0>(identity, batch * H * W * IC_0, "TXT_FILES_dense_input.txt");

    if (!use_mask)
        w_index = read_file_static<WW, W_FACTOR>(weight_buffer,  IC_0, "TXT_FILES_w_mask.txt", w_index);
    if (enable_pool)
        w_index = read_file_static<WW, W_FACTOR>(weight_buffer,  OC_1 * OC_1 / CPRUNE_FACTOR, "TXT_FILES_fc.txt", w_index);
    if(first_layer)
        w_index = read_file_static<WW, W_FACTOR>(weight_buffer,  IC_0 * P4 * 9, "TXT_FILES_w3_dense_layout.txt", w_index);
    if (!skip_0)
        w_index = read_file_static<WW, W_FACTOR>(weight_buffer,  IC_0 * OC_0, "TXT_FILES_w0_dense_layout.txt", w_index);
    if (!skip_3)
	    w_index = read_file_static<WW, W_FACTOR>(weight_buffer,  OC_0 * 9 * OC_3, "TXT_FILES_w3_dense_layout.txt", w_index);
    if (!skip_1)
        w_index = read_file_static<WW, W_FACTOR>(weight_buffer,  OC_3 * OC_1, "TXT_FILES_w1_dense_layout.txt", w_index);
    if (use_mask)
        w_index = read_file_static<1, MW>(weight_buffer,  H * W, "TXT_FILES_full_spatial_mask.txt", w_index);
    

    top(fin, fin, identity, weight_buffer, cmask_out, H, W, IC_0, OC_0, OC_3, OC_1, next_c, flags, cmask_ic, cmask_oc, batch);
        
    FILE *f_gt;
    f_gt = fopen("TXT_FILES_dense_out_1.txt", "r");
    cout<<"Comparing with TXT_FILES_dense_out_1.txt"<<endl;


    count = 0;  pass = 1;
    for (int i = 0; i < batch * H * W * OC_1 / PO_1; i++){
        ap_uint<FW * PO_1> rd = fin[i];
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



    cout<<"-------------start testing DP with given spatial mask--------------"<<endl;
    H=8, W=8, IC_0=128, OC_0=128, OC_3=128, OC_1=128, next_c=128, batch=1;
    skip_0=0, skip_3=0, skip_1=0, residual=1, stride2=0, use_mask=1, enable_pool=1, use_cprune=1, first_layer=0, return_mask=1;
    relu_0 = 1, relu_3 = 1, relu_1 = 1;


    flags[0] = skip_0;      
    flags[1] = skip_3;      
    flags[2] = skip_1;      
    flags[3] = enable_pool; 
    flags[4] = stride2;     
    flags[5] = residual;    
    flags[6] = use_mask;    
    flags[7] = use_cprune;
    flags[8] = first_layer;     
    flags[9] = return_mask;  
    flags[10] = relu_0;      
    flags[11] = relu_3;      
    flags[12] = relu_1;   

    cout<<"flags: "<<flags<<endl;

    cmask_ic = read_cprune<ap_uint<8> >("TXT_FILES_channel_mask.txt", OC_0/CPRUNE_FACTOR);
    cmask_oc = cmask_ic;
    


    w_index = 0;
    readfile<FW, PI_0>(fin, batch * H * W * IC_0, "TXT_FILES_dense_input.txt");
    readfile<FW, PI_0>(identity, batch * H * W * IC_0, "TXT_FILES_dense_input.txt");

    if (!use_mask)
        w_index = read_file_static<WW, W_FACTOR>(weight_buffer,  IC_0, "TXT_FILES_w_mask.txt", w_index);
    if (enable_pool)
        w_index = read_file_static<WW, W_FACTOR>(weight_buffer,  OC_1 * OC_1 / CPRUNE_FACTOR, "TXT_FILES_fc.txt", w_index);
    if(first_layer)
        w_index = read_file_static<WW, W_FACTOR>(weight_buffer,  IC_0 * P4 * 9, "TXT_FILES_w3_sparse_layout.txt", w_index);
    if (!skip_0)
        w_index = read_file_static<WW, W_FACTOR>(weight_buffer,  IC_0 * OC_0, "TXT_FILES_w0_sparse_layout.txt", w_index);
    if (!skip_3)
	    w_index = read_file_static<WW, W_FACTOR>(weight_buffer,  OC_0 * 9 * OC_3, "TXT_FILES_w3_sparse_layout.txt", w_index);
    if (!skip_1)
        w_index = read_file_static<WW, W_FACTOR>(weight_buffer,  OC_3 * OC_1, "TXT_FILES_w1_sparse_layout.txt", w_index);
    if (use_mask)
        w_index = read_file_static<1, MW>(weight_buffer,  H * W, "TXT_FILES_spatial_mask.txt", w_index);

    top(fin, fin, identity, weight_buffer, cmask_out, H, W, IC_0, OC_0, OC_3, OC_1, next_c, flags, cmask_ic, cmask_oc, batch);
    

    FILE *sp_gt;

    sp_gt = fopen("TXT_FILES_sparse_out_1.txt", "r");
    cout<<"Comparing with TXT_FILES_sparse_out_1.txt"<<endl;


    count = 0; pass = 1;
    for (int i = 0; i < batch * H * W * OC_1 / PO_1; i++){
        ap_uint<FW * PO_1> rd = fin[i];
        for(int j = 0; j < PO_1; j++){
            int out_read = (T_F) (rd.range((j + 1) * FW - 1, j * FW));
            int tmp;
		    fscanf(sp_gt, "%d", &tmp);
            if (tmp != out_read){
                cout<<"failed at:"<<count<<" gt:"<<tmp<<" out:"<<out_read<<endl;
                pass = 0;
            }
            count++;
   	    }
    }
    if(pass) cout<<"Passed DP tb!"<<endl; 
    fclose(sp_gt);

    if(cmask_out[0] > cmask_out[1]){
        cmask_out[0] = 1;
        cmask_out[1] = 0;
    }
    else{
        cmask_out[1] = 1;
        cmask_out[0] = 0;        
    }
    cout<<"next channsl mask:\t"<<cmask_out[0]<<"\t"<<cmask_out[1]<<endl;

    // cout<<"out spatial mask"<<endl;
    // int mask_count = 0;
    // for(int i = 0; i < ceil_div<MW>(H * W); i++){
    //     for(int j = 0; j < MW; j++){
    //         cout<<smask_out[i][j]<<'\t';
    //         if(mask_count++ == H * W - 1) break;
    //         if(mask_count % W == 0) cout<<endl;
    //     }
    //     cout<<endl;
    // }

    cout<<"-------------end testing dp with given spatial mask--------------\n\n";




    cout<<"-------------start testing DP with spatial mask unit--------------"<<endl;
    H=8, W=8, IC_0=128, OC_0=128, OC_3=128, OC_1=128, next_c=128, batch=1;
    skip_0=0, skip_3=0, skip_1=0, residual=1, stride2=0, use_mask=0, enable_pool=1, use_cprune=1, first_layer=0, return_mask=1;
    relu_0 = 1, relu_3 = 1, relu_1 = 1;



    flags[0] = skip_0;      
    flags[1] = skip_3;      
    flags[2] = skip_1;      
    flags[3] = enable_pool; 
    flags[4] = stride2;     
    flags[5] = residual;    
    flags[6] = use_mask;    
    flags[7] = use_cprune;
    flags[8] = first_layer;     
    flags[9] = return_mask;  
    flags[10] = relu_0;      
    flags[11] = relu_3;      
    flags[12] = relu_1;   


    cout<<"flags: "<<flags<<endl;

    cmask_ic = read_cprune<ap_uint<8> >("TXT_FILES_channel_mask.txt", 2);
    cmask_oc = cmask_ic;
    
    w_index = 0;
    readfile<FW, PI_0>(fin, batch * H * W * IC_0, "TXT_FILES_sparse_input.txt");
    readfile<FW, PI_0>(identity, batch * H * W * IC_0, "TXT_FILES_sparse_input.txt");

    if (!use_mask)
        w_index = read_file_static<WW, W_FACTOR>(weight_buffer,  IC_0, "TXT_FILES_w_mask.txt", w_index);
    if (enable_pool)
        w_index = read_file_static<WW, W_FACTOR>(weight_buffer,  OC_1 * OC_1 / CPRUNE_FACTOR, "TXT_FILES_fc.txt", w_index);
    if(first_layer)
        w_index = read_file_static<WW, W_FACTOR>(weight_buffer,  IC_0 * P4 * 9, "TXT_FILES_w3_sparse_layout.txt", w_index);
    if (!skip_0)
        w_index = read_file_static<WW, W_FACTOR>(weight_buffer,  IC_0 * OC_0, "TXT_FILES_w0_sparse_layout.txt", w_index);
    if (!skip_3)
	    w_index = read_file_static<WW, W_FACTOR>(weight_buffer,  OC_0 * 9 * OC_3, "TXT_FILES_w3_sparse_layout.txt", w_index);
    if (!skip_1)
        w_index = read_file_static<WW, W_FACTOR>(weight_buffer,  OC_3 * OC_1, "TXT_FILES_w1_sparse_layout.txt", w_index);
    if (use_mask)
        w_index = read_file_static<1, MW>(weight_buffer,  H * W, "TXT_FILES_spatial_mask.txt", w_index);


    top(fin, fin, identity, weight_buffer, cmask_out, H, W, IC_0, OC_0, OC_3, OC_1, next_c, flags, cmask_ic, cmask_oc, batch);
    

    FILE *spu_gt;

    spu_gt = fopen("TXT_FILES_sparse_out_1_sparse.txt", "r");
    cout<<"Comparing with TXT_FILES_sparse_out_1_sparse.txt"<<endl;


    count = 0; pass = 1;
    for (int i = 0; i < batch * H * W * OC_1 / PO_1; i++){
        ap_uint<FW * PO_1> rd = fin[i];
        for(int j = 0; j < PO_1; j++){
            int out_read = (T_F) (rd.range((j + 1) * FW - 1, j * FW));
            int tmp;
		    fscanf(spu_gt, "%d", &tmp);
            if (tmp != out_read){
                cout<<"failed at:"<<count<<" gt:"<<tmp<<" out:"<<out_read<<endl;
                pass = 0;
            }
            count++;
   	    }
    }
    if(pass) cout<<"Passed DP spatial unit tb!"<<endl; 
    fclose(spu_gt);

    if(cmask_out[0] > cmask_out[1]){
        cmask_out[0] = 1;
        cmask_out[1] = 0;
    }
    else{
        cmask_out[1] = 1;
        cmask_out[0] = 0;        
    }
    cout<<"next channsl mask:\t"<<cmask_out[0]<<"\t"<<cmask_out[1]<<endl;

    // cout<<"out spatial mask"<<endl;
    // mask_count = 0;
    // for(int i = 0; i < ceil_div<MW>(H * W); i++){
    //     for(int j = 0; j < MW; j++){
    //         cout<<smask_out[i][j]<<'\t';
    //         if(mask_count++ == H * W - 1) break;
    //         if(mask_count % W == 0) cout<<endl;
    //     }
    //     cout<<endl;
    // }

    cout<<"-------------end testing dp with spatial mask unit--------------"<<endl;


}

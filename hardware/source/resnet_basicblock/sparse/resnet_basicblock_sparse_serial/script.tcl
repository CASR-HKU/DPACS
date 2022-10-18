############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
############################################################
open_project sp_hls_new_proj
set_top top
add_files top.cpp
add_files -tb tb_file/TXT_FILES/TXT_FILES_dense_input.txt
add_files -tb tb_file/TXT_FILES/TXT_FILES_sparse_input.txt
add_files -tb tb_file/TXT_FILES/TXT_FILES_dense_out_0.txt
add_files -tb tb_file/TXT_FILES/TXT_FILES_dense_out_3.txt
add_files -tb tb_file/TXT_FILES/TXT_FILES_sparse_out_0.txt
add_files -tb tb_file/TXT_FILES/TXT_FILES_sparse_out_3.txt
add_files -tb tb_file/TXT_FILES/TXT_FILES_sparse_out_3_sparse.txt
add_files -tb tb_file/TXT_FILES/TXT_FILES_w0_dense_layout.txt
add_files -tb tb_file/TXT_FILES/TXT_FILES_w3_dense_layout.txt
add_files -tb tb_file/TXT_FILES/TXT_FILES_w0_sparse_layout.txt
add_files -tb tb_file/TXT_FILES/TXT_FILES_w3_sparse_layout.txt
add_files -tb tb_file/TXT_FILES/TXT_FILES_w_mask.txt
add_files -tb tb_file/TXT_FILES/TXT_FILES_spatial_mask.txt
add_files -tb tb_file/TXT_FILES/TXT_FILES_channel_mask.txt
add_files -tb tb_file/TXT_FILES/TXT_FILES_full_spatial_mask.txt
add_files -tb tb_file/TXT_FILES/TXT_FILES_fc.txt
add_files -tb tb.cpp -cflags "-Wno-unknown-pragmas"
open_solution "solution1" -flow_target vivado
set_part {xczu9eg-ffvb1156-2-e}
create_clock -period 3 -name default
#source "./sp_hls/solution1/directives.tcl"
csim_design
csynth_design
cosim_design
export_design -format ip_catalog
quit

SHELL := /bin/bash
VHLS_PATH := $(dir $(shell which vivado_hls))/..
VHLS_INC ?= ${VHLS_PATH}/include
proj_dir = $(shell pwd)
bifile_dir = ../../../../../drive/ZCU102/bitfiles
name = resnet_basicblock_sparse_parallel
hls_name = sp_hls_new_proj
tcl_name = sp_new_vivado_proj
xsa_dir = bitfile

CFLAGS = -g -I${VHLS_INC}

all: gen_tb hls bitstream unpack

gen_tb:
	cd ./tb_file; python tb_generator.py

csim:
	faketime -f "-1y" vitis_hls -f script_csim.tcl

hls:
	faketime -f "-1y" vitis_hls -f script.tcl

bitstream:
	
	cp ./vivado/design_1_wrapper.v ./$(hls_name)/solution1/impl/ip/
	cp ./vivado/$(tcl_name).tcl ./$(hls_name)/solution1/impl/ip/
	cd ./$(hls_name)/solution1/impl/ip/; vivado -mode tcl -source $(tcl_name).tcl

unpack:
	cd ./$(xsa_dir); unzip $(xsa_dir).xsa -d xsa; mv ./xsa/design_1.hwh $(name).hwh; mv ./xsa/$(xsa_dir).bit $(name).bit; cp $(name).bit $(bifile_dir); cp $(name).hwh $(bifile_dir); rm -r xsa
	
clean:
	rm -rf $(hls_name) $(xsa_dir) *.prj *.log .Xil *.jou

cleanv:
	rm -rf $(hls_name)/solution1/impl/ip/sp_new_vivado_proj
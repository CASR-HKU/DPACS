
#define FW 8   //input bitwidth
#define WW 8   //weight bitwidth
#define PW 24  //partial sum bitwidth
#define SW 16  //quantization shift bitwidth
#define BW 16  //bias bitwidth
#define CW 12  //channel bitwidth
#define HW 10  //image height
#define SHIFT_W 16  //quantization shift bits



typedef ap_int<FW> T_F; 
typedef ap_int<WW> T_W; 
typedef ap_int<PW> T_P; 
typedef ap_int<SW> T_S; 
typedef ap_int<BW> T_B; 
typedef ap_uint<CW> T_C; 
typedef ap_int<HW> T_H; 
typedef ap_int<SW + BW> T_Q;
typedef ap_uint<4> T_BASE;

typedef ap_uint<6> T_BATCH;
typedef ap_int<256> int256; 

// typedef ap_uint<KW> T_K; //key type

typedef ap_uint<4> T_OFFSET;



typedef struct T_K{
	ap_uint<1> end;
	ap_uint<HW> x;
	ap_uint<HW> y;
} T_K;

template <unsigned int N, typename T>
struct BundleT {
	T data[N];
};

// parallel factors for different kernels
#define PI_0 32 
#define PO_0 32
#define PI_3 32
#define PO_3 32
#define PI_1 32
#define PO_1 32
#define P4 4
#define P8 4
#define CPRUNE_FACTOR 32

#define W_FACTOR CPRUNE_FACTOR

#define MW (CPRUNE_FACTOR * WW)
typedef ap_uint<MW> T_MASK;

#define MAX_IC 512
#define MAX_C  256
#define MAX_OC 512
#define KERNEL 3
#define MAX_H 256    

#define end_3x3 15
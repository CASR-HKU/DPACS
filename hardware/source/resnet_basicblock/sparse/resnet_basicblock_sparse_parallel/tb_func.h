
template<int W, int PA, typename T_OUT>
void readfile(T_OUT *mem, int length, const char* name){
	int count = 0;
	FILE *f_s;
	f_s=fopen(name,"r");
	for (int i=0; i<length/PA; i++){
	  T_OUT temp;
	  for(int j=0;j<PA;j++){
		  int tmp;
		  fscanf(f_s, "%d", &tmp);
		  temp.range(W*(j+1)-1,W*j)=tmp;
	  }
	  mem[count++] = temp;
	//   cout<<temp<<endl;
	}
	fclose(f_s);
}

template<int W, int PA, typename T_OUT>
int read_file_static(T_OUT *mem, int length, const char* name, int start_index){
	
	int count = start_index;
	FILE *f_s;
	f_s = fopen(name,"r");
	int rep = ceil_div<PA>(length);
	for (int i = 0; i < rep; i++){
	  T_OUT temp;
	  for(int j = 0; j < PA; j++){
		  int tmp;
		  fscanf(f_s, "%d", &tmp);
		  temp.range(W * (j + 1) - 1, W * j) = tmp;
	  }
	  mem[count++] = temp;
	//   cout<<temp<<endl;
	}
	fclose(f_s);
	return count;
}


template<typename T_OUT>
T_OUT read_cprune(const char* name, int N){
	int count = 0;
	FILE *f_s;
	f_s=fopen(name,"r");
	T_OUT cprune = 0;

	for (int i = 0; i < N; i++){
		int tmp;
		fscanf(f_s, "%d", &tmp);
		cprune[i] = tmp;
	}
	fclose(f_s);
	return cprune;
}
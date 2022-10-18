template<int W, int PA, typename T_OUT>
void readfilepack(T_OUT *mem, int length, const char* name){
	static int count = 0;
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
	  //cout<<endl;
	}
	fclose(f_s);
}
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

#include<stdio.h>
#include<string.h>
#include<random>
#include<string>
#include<algorithm>
#include<set>

std::mt19937 generator(1234567);
std::uniform_int_distribution<long long> dist(0,std::numeric_limits<long long>::max());

long long fast_pow(long long A, long long X,long long M){
	__int128_t B = 1,exp_a = A;
	while(X){
		if(X & 1)
			B = B * exp_a % M;
		exp_a = exp_a * exp_a % M;
		X >>= 1;
	}
	return B;
}

void dump_func(int cas,long long M){
	printf("dumping case %d\n",cas);
	std::string file_name = "sample" + std::to_string(cas) + ".in";
	FILE* fp = fopen(file_name.c_str(),"w");
	long long A = dist(generator) % M % 100000;
	long long X = dist(generator) % M;
	long long B = fast_pow(A,X,M);
	fprintf(fp,"%lld %lld %lld\n",A,B,M);
	fclose(fp);
	file_name = "sample" + std::to_string(cas) + ".out";
	fp = fopen(file_name.c_str(),"w");
	fprintf(fp,"%lld\n",X);
	fclose(fp);
}

int main(){
	dump_func(1,11LL);
	dump_func(2,10000019LL);
	dump_func(3,1000000007LL);
	dump_func(4,10000000019LL);
	dump_func(5,20184892661LL);
	dump_func(6,30241095613LL);
	for(int cas = 7;cas <= 10;++cas){
		dump_func(cas,40929166261LL);
	}
	return 0;
}


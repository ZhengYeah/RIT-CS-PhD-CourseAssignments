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

int main(int argc,char** argv){
	long long A,B,M,X;
	FILE* fin = fopen(argv[1],"r");
	FILE* fout = fopen(argv[2],"r");
	fscanf(fin,"%lld%lld%lld",&A,&B,&M);
	fscanf(fout,"%lld",&X);
	fclose(fin);
	fclose(fout);
	if(fast_pow(A,X,M) == B)
		return 0;
	else
		return 1;
}


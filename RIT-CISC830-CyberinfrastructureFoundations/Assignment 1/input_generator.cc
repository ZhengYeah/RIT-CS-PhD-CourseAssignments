#include<stdio.h>
#include<string.h>
#include<random>
#include<string>
#include<algorithm>

std::mt19937 generator(12356789);
std::uniform_int_distribution<int> dist(0,std::numeric_limits<int>::max());

void dump_func(int cas,int N,int K,int A,int B,int C,int M,int (*f)(int)){
	printf("dumping case %d\n",cas);
	std::string file_name = "sample" + std::to_string(cas) + ".in";
	FILE* fp = fopen(file_name.c_str(),"w");
	fprintf(fp,"%d %d %d %d %d %d\n",N,K,A,B,C,M);
	for(int i = 0;i < K;++i){
		fprintf(fp,"%d\n",f(i) % M);
	}
	fclose(fp);
}

int main(){
	int maxn = 100000000;
	int maxm = 1000000007;
	dump_func(3,maxn,2,1,0,maxm - 1,maxm,[](int x){return 1 - x;});
	dump_func(4,maxn,2,1,0,3,maxm,[](int x){return x;});
	dump_func(5,maxn,maxn,dist(generator),dist(generator),dist(generator),maxm,[](int x){return dist(generator);});
	for(int cas = 6;cas <= 10;++cas){
		dump_func(cas,maxn,100,dist(generator),dist(generator),dist(generator),maxm,[](int x){return dist(generator);});
	}
	return 0;
}


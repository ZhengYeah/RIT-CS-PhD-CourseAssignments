#include<stdio.h>
#include<string.h>
#include<algorithm>

int X[10000000];

int main(int argc,char** argv){
	int N,K,A,B,C,M;
	FILE* fin = fopen(argv[1],"r");
	fscanf(fin,"%d%d%d%d%d%d",&N,&K,&A,&B,&C,&M);
	for(int i = 0;i < K;++i)
		fscanf(fin,"%d",&X[i]);
	fclose(fin);

	FILE* fout = fopen(argv[2],"w");
	for(int i = K;i < N;++i)
		X[i] = ((long long)A * X[i - 1] + (long long)B * X[i - 2] + C) % M;
	std::sort(X,X + N);
	for(int i = 0;i < N;++i)
		fprintf(fout,"%d\n",X[i]);
	fclose(fout);
	return 0;
}


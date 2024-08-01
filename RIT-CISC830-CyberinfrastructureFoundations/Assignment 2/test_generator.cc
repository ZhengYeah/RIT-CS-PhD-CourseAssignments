#include<stdio.h>
#include<string.h>
#include<random>
#include<string>
#include<algorithm>
#include<set>

std::mt19937 generator(22356789);
std::uniform_int_distribution<int> dist(0,std::numeric_limits<int>::max());

void dump_func_hard2(int cas,int V,int D,int E,int L,int K,int A,int B,int C,int M,int Q,int (*f)(int)){
        printf("dumping case %d\n",cas);
        std::string file_name = "sample" + std::to_string(cas) + ".in";
        FILE* fp = fopen(file_name.c_str(),"w");
        fprintf(fp,"%d %d %d %d %d %d %d %d %d %d\n",V,D,E,L,K,A,B,C,M,Q);
        for(int i = 0;i < K;++i){
                fprintf(fp,"%d\n",dist(generator) % M);
        }
        std::set<std::pair<int,int>> visited;
        std::vector<int> degree(V,0);
        int nextid = 1;
        for(int i = 0;i < 1000;++i){
                for(int j = 0;j < 32;++j){
                        fprintf(fp,"%d %d\n",i,nextid);
                        visited.insert(std::make_pair(i,nextid));
                        ++nextid;
                        ++degree[i];
                }
        }
        for(int i = 1000 * 32;i < E;++i){
                while(1){
                        int u = dist(generator) % V;
                        int v = dist(generator) % V;
                        if(u > v)
                                std::swap(u,v);
                        if(u == v || degree[u] >= L || visited.count(std::make_pair(u,v)))
                                continue;
                        visited.insert(std::make_pair(u,v));
                        ++degree[u];
                        fprintf(fp,"%d %d\n",u,v);
                        break;
                }
        }
        for(int i = 0;i < Q;++i){
                fprintf(fp,"%d %d",dist(generator) % 64,3);
                for(int j = 0;j < D;++j)
                        fprintf(fp," %d",dist(generator) % M);
                fprintf(fp,"\n");
        }
        fclose(fp);
}

void dump_func_hard(int cas,int V,int D,int E,int L,int K,int A,int B,int C,int M,int Q,int (*f)(int)){
        printf("dumping case %d\n",cas);
        std::string file_name = "sample" + std::to_string(cas) + ".in";
        FILE* fp = fopen(file_name.c_str(),"w");
        fprintf(fp,"%d %d %d %d %d %d %d %d %d %d\n",V,D,E,L,K,A,B,C,M,Q);
        for(int i = 0;i < K;++i){
                fprintf(fp,"%d\n",dist(generator) % M);
        }
        std::set<std::pair<int,int>> visited;
        std::vector<int> degree(V,0);
        for(int i = 0;i < 10000;++i){
                for(int j = 0;j < 63;++j){
                        fprintf(fp,"%d %d\n",i,i + j + 1);
                        visited.insert(std::make_pair(i,i + j + 1));
                        ++degree[i];
                }
        }
        for(int i = 10000 * 63;i < E;++i){
                while(1){
                        int u = dist(generator) % V;
                        int v = dist(generator) % V;
                        if(u > v)
                                std::swap(u,v);
                        if(u == v || degree[u] >= L || visited.count(std::make_pair(u,v)))
                                continue;
                        visited.insert(std::make_pair(u,v));
                        ++degree[u];
                        fprintf(fp,"%d %d\n",u,v);
                        break;
                }
        }
        for(int i = 0;i < Q;++i){
                fprintf(fp,"%d %d",dist(generator) % 10000,9);
                for(int j = 0;j < D;++j)
                        fprintf(fp," %d",dist(generator) % M);
                fprintf(fp,"\n");
        }
        fclose(fp);
}

void dump_func(int cas,int V,int D,int E,int L,int K,int A,int B,int C,int M,int Q,int (*f)(int)){
        printf("dumping case %d\n",cas);
        std::string file_name = "sample" + std::to_string(cas) + ".in";
        FILE* fp = fopen(file_name.c_str(),"w");
        fprintf(fp,"%d %d %d %d %d %d %d %d %d %d\n",V,D,E,L,K,A,B,C,M,Q);
        for(int i = 0;i < K;++i){
                fprintf(fp,"%d\n",dist(generator) % M);
        }
        std::set<std::pair<int,int>> visited;
        std::vector<int> degree(V,0);
        for(int i = 0;i < V - 1;++i){
                fprintf(fp,"%d %d\n",i,i + 1);
                visited.insert(std::make_pair(i,i + 1));
                ++degree[i];
        }
        for(int i = V - 1;i < E;++i){
                while(1){
                        int u = dist(generator) % V;
                        int v = dist(generator) % V;
                        if(u > v)
                                std::swap(u,v);
                        if(u == v || degree[u] >= L || visited.count(std::make_pair(u,v)))
                                continue;
                        visited.insert(std::make_pair(u,v));
                        ++degree[u];
                        fprintf(fp,"%d %d\n",u,v);
                        break;
                }
        }
        for(int i = 0;i < Q - 11;++i){
                fprintf(fp,"%d %d",dist(generator) % V,3);
                for(int j = 0;j < D;++j)
                        fprintf(fp," %d",dist(generator) % M);
                fprintf(fp,"\n");
        }
        for(int i = 0;i < 10;++i){
                fprintf(fp,"%d %d",dist(generator) % V,5);
                for(int j = 0;j < D;++j)
                        fprintf(fp," %d",dist(generator) % M);
                fprintf(fp,"\n");
        }
        for(int i = 0;i < 1;++i){
                fprintf(fp,"%d %d",dist(generator) % V,10);
                for(int j = 0;j < D;++j)
                        fprintf(fp," %d",dist(generator) % M);
                fprintf(fp,"\n");
        }
        fclose(fp);
}

int main(){
        int maxV = 100000;
        int maxD = 1000;
        int maxE = 1000000;
        int maxL = 63;
        int maxQ = 10000;
        int maxM = 100;
        for(int cas = 4;cas <= 8;++cas){
                dump_func(cas,maxV,maxD,maxE,maxL,100,dist(generator),dist(generator),dist(generator),maxM,maxQ,[](int x){return dist(generator);});
        }
        dump_func_hard(9,maxV,maxD,maxE,maxL,100,dist(generator),dist(generator),dist(generator),maxM,maxQ,[](int x){return dist(generator);});
        dump_func_hard2(10,maxV,maxD,maxE,maxL,100,dist(generator),dist(generator),dist(generator),maxM,maxQ,[](int x){return dist(generator);});
        return 0;
}
#include <iostream>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include "SIMDLib/simd_arch.h"
#include "SIMDLib/VL2/vl2_matrix.h"

using namespace std;
//效率监控变量
struct timeval start_tv;
struct timeval end_tv;
//优化参数变量

/*
Input: a, b are the N*N float matrix, 0<seed<1, float
This function should initialize two matrixs with rand_float()
*/
float rand_float(float s) {
	return 4 * s*(1 - s);
	//return s;
}
void matrix_gen(float *a, float *b, int N, float seed) {
	float s = seed;
	for (int i = 0; i < N*N; i++) {
		s = rand_float(s);
		a[i] = s;
		s = rand_float(s);
		b[i] = s;
	}
}
enum optiType {
	NON_OPT,		//无优化
	MUTI_THREAD,	//多线程
	TILE,			//矩阵行读优化
	TILE_V2,		//矩阵分片
	SSE,			//SSE指令优化
	AVX,			//AVX指令优化
	GPU,			//GPU加速
};
void matrix_Mut(float *a, float *b, float *c, int N, optiType TYPE);
float matrix_Tr(float *c, int N) {
	float tmp = 0;
	for (int i = 0; i < N; i++) {
		tmp += c[i*N + i];
	}
	return tmp;
}

/* CONFIG */
// const int N_list_size = 5;
// int N_list[N_list_size] = { 512	,1024	,2048	,4096	,8192 };
const int N_list_size = 1;
int N_list[1] = {2048}; 
int turn = 0;
static optiType opt = TILE; // optimation type
static int n_thread = 8;

int main()
{
	if(opt==MUTI_THREAD) cout<<"Threads: "<<n_thread<<endl; 
	for (; turn < N_list_size; turn++) {
		//矩阵声明与变量初始化
		int N = N_list[turn]; //矩阵规模：N×N
		float *a = new float[N*N];
		float *b = new float[N*N];
		float *c = new float[N*N];
		float seed = 0.3;

		//初始化矩阵
		matrix_gen(a, b, N, seed);

		//效率监控开始
		////Linux
        start_tv.tv_usec=0;
        gettimeofday(&start_tv,NULL);

		//开始计算
		matrix_Mut(a, b, c, N, opt);

		//效率监控结束
		////Linux
        end_tv.tv_usec=0;
        gettimeofday(&end_tv,NULL);
        long long timecost = (end_tv.tv_sec-start_tv.tv_sec)*1000+(end_tv.tv_usec-start_tv.tv_usec)/1000;

		//Trace计算和结果验证
		float trace = matrix_Tr(c, N);
		float trace_val = 0.0;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				trace_val += a[i*N + j] * b[j*N + i];
			}
		}
		float per_diff = (trace - trace_val) / trace;
		// if (per_diff<0.001 || per_diff>-0.001) cout << "True" << endl;
		// else cout << "False, trace = " << trace << ", trace_val = " << trace_val << endl;
		cout << "Size: " << N << " Time: " << timecost << " Trace: "<< trace << endl;
		//释放内存
		delete a;
		delete b;
		delete c;
	}
	return 0;
}
void matrix_Mut(float *a, float *b, float *c, int N, optiType TYPE) {
	switch (TYPE)
	{
	case NON_OPT:
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				c[i*N + j] = 0.0;
				for (int k = 0; k < N; k++) {
					c[i*N + j] += a[i*N + k] * b[k*N + j];
				}
			}
		}
		break;
	case MUTI_THREAD:
		#pragma omp parallel for num_threads(n_thread)
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    c[i*N + j] = 0.0;
                    for (int k = 0; k < N; k++) {
                        c[i*N + j] += a[i*N + k] * b[k*N + j];
                    }
                }
            }
		break;
	case TILE:
		#pragma omp parallel for num_threads(n_thread)
		for(int i=0;i<N;i++){
			for(int j=0;j<N;j++){
				for(int k=0;k<N;k++){
					c[i*N+k] += a[i*N+j]*b[j*N+k];
				}
			}
		}
		break;
	case SSE:{
		/*
			// float *pA = a;
			// float *pB = b;
			// float *pC = c;
			// int idxA = 0;
			// int idxB = 0;
			// int idxC = 0;
			// for(int i=0;i<N*N/(_VF32_SIZE*_VF32_SIZE);i++){
			// 	pB = b+idxB;
			// 	idxA = idxB/N; pA = a+idxA;
			// 	idxC = idxB%N; pC = c+idxC;
			// 	for(int j=0;j<N/_VF32_SIZE;j++){
			// 		cout<<"idxA: "<<idxA/N<<','<<idxA%N<<'\t';
			// 		cout<<"idxB: "<<idxB/N<<','<<idxB%N<<'\t';
			// 		cout<<"idxC: "<<idxC/N<<','<<idxC%N<<endl;
			// 		matrixF32_madd(pC, pA, pB, N);				
			// 		idxA += N*_VF32_SIZE; pA = a+idxA;
			// 		idxC += N*_VF32_SIZE; pC = c+idxC;
			// 	}
			// 	idxB += _VF32_SIZE;
			// 	idxB = (idxB/N)*_VF32_SIZE*N + idxB%N;
			// }
		*/
		int nBox = N/_VF32_SIZE; // 当前_VF32_SIZE大小下，分块后矩阵的规模
		int iABox = 0, iBBox = 0, iCBox = 0; // 原A、B、C矩阵分块后得到的分块矩阵的索引,Box对应与原矩阵就是一个方形选框
		for(int i=0;i<nBox*nBox;i++){ // 顶层循环：固定BBox，纵向移动ABox和CBox
			iABox = iBBox/nBox+0; // 由BBox所在行决定ABox所在列
			iCBox = iBBox%nBox+0; // 由BBox所在列决定CBox所在列
			//#pragma omp parallel for num_threads(n_thread)
			for(int j=0;j<nBox;j++){
				// cout<<"iABox: "<<iABox/nBox<<','<<iABox%nBox<<'\t';
				// cout<<"iBBox: "<<iBBox/nBox<<','<<iBBox%nBox<<'\t';
				// cout<<"iCBox: "<<iCBox/nBox<<','<<iCBox%nBox<<'\t';
				// 由iBox计算Box内第一个元素在原矩阵的索引：idx_Element = x_idx + N * y_idx
				int idxA = _VF32_SIZE*(iABox/nBox)+_VF32_SIZE*N*(iABox%nBox);
				int idxB = _VF32_SIZE*(iBBox/nBox)+_VF32_SIZE*N*(iBBox%nBox);
				int idxC = _VF32_SIZE*(iCBox/nBox)+_VF32_SIZE*N*(iCBox%nBox);
				// cout<<"idxA: "<<idxA/N<<','<<idxA%N<<'\t';
				// cout<<"idxB: "<<idxB/N<<','<<idxB%N<<'\t';
				// cout<<"idxC: "<<idxC/N<<','<<idxC%N<<endl;
				matrixF32_madd(&c[idxC], &a[idxA], &b[idxB], N);
				iABox += nBox; // ABox向下移动一个Box位置
				iCBox += nBox; // CBox向下移动一个Box位置
			}
			iBBox++; // BBox移动到下一个位置
		}

		break;
	}
	case AVX:
		break;
	case GPU:
		break;
	default:
		break;
	}
}
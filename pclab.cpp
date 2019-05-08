#include <iostream>
#include <fstream>
#include <math.h>
#include <sys/time.h>
#include <string>
#include <bitset>
#include <algorithm>
#include <omp.h>
#include "SIMDLib/simd_arch.h"
#include "SIMDLib/VL2/vl2_matrix.h"

// #define MATRIX
#define SORT
#define DEBUG
#define OUT_LOG

using namespace std;
//效率监控变量
struct timeval start_tv;
struct timeval end_tv;
//优化参数变量

#ifdef MATRIX
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
        BLOCK,		//矩阵分块
        SIMD,			//SIMD指令优化
        GPU,			//GPU加速
    };
    void matrix_Mut(float *a, float *b, float *c, int N, optiType TYPE);
    void matrix_mut_simd(float *a, float *b, float *c, int N);
    float matrix_Tr(float *c, int N) {
        float tmp = 0;
        for (int i = 0; i < N; i++) {
            tmp += c[i*N + i];
        }
        return tmp;
    }
    /* CONFIG */
    // const int N_list_size = 5;
    // static int N_list[N_list_size] = { 512,1024,2048,4096,8192};
    const int N_list_size = 1;
    int N_list[1] = {8192}; 
    int nTest = 1;
    int tolTime = 0;
    int turn = 0;
    int block_size = 32;
    static optiType opt = NON_OPT; // optimation type
    static int n_thread = 8;
#endif

#ifdef SORT
    /* ==== CONFIG ==== */
    #define THREAD_NUM 4
    #define RADIX 8 // size of radix in bits, determine the length of mask
    #define N 32 // target: 1000000000
    #define SEED 0.3
    #define BUF_BIN_SIZE 16 // number of ints in per cache buf, which is no more than (cache_size/n_bins)/sizeof(int)
    #define L2_CACHE 256 // the size of L2 cache in KB
    #define DATA_DIR "./sort_data/"

    // #define R_SORT
    #define M_SORT
    // #define MT_R_SORT
    #define MT_M_SORT

    #define QSORT
    // #define UNSORTED_WBACK
    // #define SORTED_WBACK
    /* ================ */
    void sort_gen(int *d, const long long len, int seed){
        cout<<"Generating arr"<<endl;
        srand(seed);
        for(long long i=0;i<len;i++){
            d[i]=rand();
        }
    }
    int cmp1(const void *elem1, const void *elem2){
        return *(int*)elem1 - *(int*)elem2;
    }
    bool check(int *arr, const long long len, string unsorted_arr_path, string sorted_arr_path ){
        // int test_arr[N] = {0};
        #ifdef QSORT
        int *test_arr = new int [N];
        fstream FILE;
        FILE.open(unsorted_arr_path,ios::in);
        if(!FILE){
            sort_gen(test_arr, len, SEED);
            #ifdef UNSORTED_WBACK
            FILE.open(unsorted_arr_path,ios::out);
            for(long long i=0;i<N;i++){
                FILE<<test_arr[i]<<' ';
            }
            FILE.close();
            cout<<unsorted_arr_path<<" generate complete"<<endl;
            #endif
        }
        else{
            cout<<"Loading "<<unsorted_arr_path<<endl;
            for(long long i = 0;i<N;i++){
                FILE>>test_arr[i];
            }
        }
        FILE.close();
        FILE.open(sorted_arr_path,ios::in);
        if(!FILE){
            cout<<"C++ sorting ..."<<endl;
                // Efficiency monitor starts
            ////Linux
            long long timecost = 0;
            start_tv.tv_usec = 0;
            gettimeofday(&start_tv, NULL);

            // Start sorting
            qsort(test_arr, N, sizeof(test_arr[0]),cmp1);

            // Efficiency monitor ends
            ////Linux
            end_tv.tv_usec = 0;
            gettimeofday(&end_tv, NULL);
            timecost = (end_tv.tv_sec - start_tv.tv_sec) * 1000 + (end_tv.tv_usec - start_tv.tv_usec) / 1000;
            cout<<"C++ qsort() Time cost: "<<timecost<<endl;

            #ifdef SORTED_WBACK
            FILE.open(sorted_arr_path,ios::out);
            for(long long i=0;i<N;i++){
                FILE<<test_arr[i]<<' ';
            }
            cout<<sorted_arr_path<<" generate complete"<<endl;
            FILE.close();
            #endif
        }
        else{
            cout<<"Loading "<<sorted_arr_path<<endl;
            for(long long i = 0;i<N;i++){
                FILE>>test_arr[i];
            }
        }        
        FILE.close();
        cout<<"Checking ..."<<endl;
        #endif
        bool result = true;
        for(long long i=0;i<len-1;i++){
            #ifdef QSORT    
            if(arr[i]!=test_arr[i]) {
                cout<<"No."<<i<<" "<<test_arr[i]<<" vs "<<arr[i]<<endl;
                result = false;
            }
            #else
            if(arr[i]>arr[i+1]){
                cout<<"error at "<<i<<" vs "<<i+1<<" : ";
                for(int j=0;j<6;j++){
                    cout<<i-2+j<<'.'<<arr[i-2+j]<<" ";
                }
                cout<<endl;
                result = false;
            }
            #endif
        }
        // cout<<"right"<<endl;
        return result;
    }
    int *arr = new int [N];
 
    #ifdef R_SORT
        void radix_sort(int *arr, const long long offset, const long long len);
    #endif 
    #ifdef MT_R_SORT
        typedef struct rsort_thr_para{
            int *arr;
            int i_thread;
        }rsort_thr_para;
        void* thread_exec_rsort(void *p_para);
        void merge(int *arr, int *out_arr);
    #endif
    #ifdef M_SORT
        typedef struct msort_thr_para{
            int *unsorted_arr;
            int *sorted_arr;
            int len;            // total length of unsorted array
            int n_sub;          // num of sorted sub-array need to be merged
            int i_thread;
        }msort_thr_para;
        void* unit_merge(void* para); // For single thread
    #endif
    #ifdef MT_M_SORT
        int *msort_thread_assign(int *org_arr, int *tmp_arr, const int block_size, const int n_subarr_of_block, const int n_thread);
    #endif

#endif 


int main(){
    #ifdef MATRIX
        for(int iTest = 0; iTest<nTest; iTest++){
            long long timecost = 0;
            if(opt==MUTI_THREAD) cout<<"Threads: "<<n_thread<<endl; 
            for (int i=0; i < N_list_size; i++) {
                //矩阵声明与变量初始化
                int N = N_list[i]; //矩阵规模：N×N
                float *a = new float[N*N];
                float *b = new float[N*N];
                float *c = new float[N*N];
                float seed = 0.3;

                //初始化矩阵
                matrix_gen(a, b, N, seed);
                for(int i=0;i<N*N;i++){
                    c[i] = 0.0;
                }

                //效率监控开始
                ////Linux
                start_tv.tv_usec=0;
                gettimeofday(&start_tv,NULL);

                //开始计算
                matrix_Mut(a, b, c, N, opt);
                // matrix_mut_simd(a, b, c, N);

                //效率监控结束
                ////Linux
                end_tv.tv_usec=0;
                gettimeofday(&end_tv,NULL);
                timecost = (end_tv.tv_sec-start_tv.tv_sec)*1000+(end_tv.tv_usec-start_tv.tv_usec)/1000;

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
                cout << "Test #"<< i <<" | Size: " << N << " Time: " << timecost << " Trace: "<< trace << endl;
                fstream FILE;
                FILE.open("r.txt",ios::out);
                FILE << "Test #" << i << " | Size: " << N << " Time: " << timecost << " Trace: " << trace << endl;
                FILE.close();
                //释放内存
                delete a;
                delete b;
                delete c;
            }
            // cout<<"Test #"<<iTest<<" Complete"<<endl;
            tolTime += timecost;
        }
        cout << "Average Time: "<<tolTime/nTest<<endl;
    #endif

    #ifdef SORT
        long long timecost = 0;
        // int arr[N] = {0};
        arr = new int [N];
        #ifdef DEBUG
        cout<<"Enter sort_gen()"<<endl;
        #endif // DEBUG
        string unsorted_arr_path = (string)DATA_DIR + "unsorted_SEED_" + to_string(SEED) + "_" + to_string(N) + ".dat";
        string sorted_arr_path = (string)DATA_DIR + "sorted_SEED_" + to_string(SEED) + "_" + to_string(N) + ".dat";
        fstream FILE;
        FILE.open(unsorted_arr_path,ios::in);
        if(!FILE){
            //效率监控开始
            ////Linux
            long long timecost = 0;
            start_tv.tv_usec = 0;
            gettimeofday(&start_tv, NULL);

            //开始计算
            sort_gen(arr, N, SEED);

            //效率监控结束
            ////Linux
            end_tv.tv_usec = 0;
            gettimeofday(&end_tv, NULL);
            timecost = (end_tv.tv_sec - start_tv.tv_sec) * 1000 + (end_tv.tv_usec - start_tv.tv_usec) / 1000;
            cout<<"Generate Timecost = "<<timecost<<endl;

            #ifdef SORTED_WBACK
            FILE.open(unsorted_arr_path,ios::out);
            for(long long i=0;i<N;i++){
                FILE<<arr[i]<<' ';
            }
            FILE.close();
            cout<<unsorted_arr_path<<" generate complete"<<endl;
            #endif
        }
        else{
            //效率监控开始
            ////Linux
            long long timecost = 0;
            start_tv.tv_usec = 0;
            gettimeofday(&start_tv, NULL);

            //开始计算
            cout<<"Loading "<<unsorted_arr_path<<endl;
            for(long long i = 0;i<N;i++){
                FILE>>arr[i];
            }

            //效率监控结束
            ////Linux
            end_tv.tv_usec = 0;
            gettimeofday(&end_tv, NULL);
            timecost = (end_tv.tv_sec - start_tv.tv_sec) * 1000 + (end_tv.tv_usec - start_tv.tv_usec) / 1000;
            cout<<"Loading Timecost = "<<timecost<<endl;

        }

        // Efficiency monitor starts
        ////Linux
        start_tv.tv_usec = 0;
        gettimeofday(&start_tv, NULL);

        #ifdef MT_R_SORT
            rsort_thr_para *para_list = new rsort_thr_para[THREAD_NUM];
            pthread_t id_thread[THREAD_NUM], ret[THREAD_NUM], idx_thread[THREAD_NUM];
            for(int i=0;i<THREAD_NUM;i++){
                para_list[i].i_thread = i;
                para_list[i].arr = arr;
                // para.i_thread = idx_thread[i];
                ret[i] = pthread_create(&id_thread[i], NULL, thread_exec_rsort, (void*)(&para_list[i]));
                // ret[i] = pthread_create(&id_thread[i], NULL, _thread_exec, (void*)(idx_thread+i));
                if(ret[i]!=0){
                    cout<<"ERROR: thread No."<<i<<" create failed"<<endl;
                    break;
                }
                #ifdef DEBUG
                cout<<"idx_thread NO."<<i<<" is "<<idx_thread[i]<<endl;
                #endif // DEBUG
            }
            for(int i=0;i<THREAD_NUM;i++){
                pthread_join(id_thread[i],NULL);
            }
            #ifdef DEBUG
            for(int i=0;i<N;i++){
                if(i%(N/THREAD_NUM)==0) cout<<"------"<<endl;
                cout<<arr[i]<<endl;
            }
            #endif // DEBUG

            int *result_arr = new int [N];
            // qsort(arr,N,sizeof(int),cmp1);
            // radix_sort(arr,0,N);
            merge(arr, result_arr);
            arr = result_arr;
        #else
            // // Start sorting
            // radix_sort(arr, 0, N);
            // #ifdef DEBUG
            // for(int i=0;i<N;i++){
            //     if(i%(N/THREAD_NUM)==0) cout<<"------"<<endl;
            //     cout<<arr[i]<<endl;
            // }
            // #endif // DEBUG
        #endif

        #ifdef M_SORT
            #ifdef DEBUG
            cout<<"=== org arr: ==="<<endl;
            for(int i=0;i<N;i++){
                cout<<arr[i]<<endl;
            }
            #endif // DEBUG
            int *swap = new int [N];
            int *tmp_arr = swap;
            #ifdef MT_M_SORT
                int *sorted = msort_thread_assign(arr,tmp_arr,N,N,THREAD_NUM);
                if(sorted==swap){
                    tmp_arr = arr;
                }
                else tmp_arr = swap;
                #ifdef DEBUG
                    cout<<" === tmp_arr ==="<<endl;
                    for(int i=0;i<N;i++){
                        cout<<tmp_arr[i]<<endl;
                    }
                    cout<<" === sorted ===="<<endl;
                    for(int i=0;i<N;i++){
                        cout<<sorted[i]<<endl;
                }
                #endif // DEBUG
                sorted = msort_thread_assign(sorted, tmp_arr, N, THREAD_NUM, 2);
                if(sorted==swap){
                    tmp_arr = arr;
                }
                else tmp_arr = swap;
                #ifdef DEBUG
                    cout<<" === tmp_arr ==="<<endl;
                    for(int i=0;i<N;i++){
                        cout<<tmp_arr[i]<<endl;
                    }
                    cout<<" === sorted ===="<<endl;
                    for(int i=0;i<N;i++){
                        cout<<sorted[i]<<endl;
                    }
                #endif // DEBUG
                sorted = msort_thread_assign(sorted, tmp_arr, N, 2, 1);
                if(sorted==swap){
                    tmp_arr = arr;
                }
                else tmp_arr = swap;
                #ifdef DEBUG
                    cout<<" === tmp_arr ==="<<endl;
                    for(int i=0;i<N;i++){
                        cout<<tmp_arr[i]<<endl;
                    }
                    cout<<" === sorted ===="<<endl;
                    for(int i=0;i<N;i++){
                        cout<<sorted[i]<<endl;
                    }
                #endif // DEBUG
                arr = sorted;
            #else
                msort_thr_para para;
                para.len = N;
                para.unsorted_arr = arr;
                para.sorted_arr = tmp_arr;
                para.i_thread = 0;
                unit_merge(&para);
                arr = para.sorted_arr;
            #endif
            #ifdef DEBUG
            cout<<"=== after unit_merge ==="<<endl;
            for(int i=0;i<N;i++){
                cout<<arr[i]<<endl;
            }
            #endif // DEBUG
        #endif


        // Efficiency monitor ends
        ////Linux
        end_tv.tv_usec = 0;
        gettimeofday(&end_tv, NULL);
        timecost = (end_tv.tv_sec - start_tv.tv_sec) * 1000 + (end_tv.tv_usec - start_tv.tv_usec) / 1000;
        cout<<"Time cost: "<<timecost<<endl;
    #endif    

    if(check(arr,N, unsorted_arr_path,sorted_arr_path)) cout<<"ALL GREEN"<<endl;
    else cout<<"Failure"<<endl;
    return 0;
}
#ifdef MATRIX
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
        
        case BLOCK:{
            int nBox = N/block_size; // 当前block_size大小下，分块后矩阵的规模
            int iABox = 0, iBBox = 0, iCBox = 0; // 原A、B、C矩阵分块后得到的分块矩阵的索引,Box对应与原矩阵就是一个方形选框
            for(int i=0;i<nBox*nBox;i++){ // 顶层循环：固定BBox，纵向移动ABox和CBox
                iABox = iBBox/nBox+0; // 由BBox所在行决定ABox所在列
                iCBox = iBBox%nBox+0; // 由BBox所在列决定CBox所在列
                //#pragma omp parallel for num_threads(n_thread)
                for(int j=0;j<nBox;j++){
                    #ifdef _DEBUG_INFO
                        cout<<"iABox: "<<iABox/nBox<<','<<iABox%nBox<<'\t';
                        cout<<"iBBox: "<<iBBox/nBox<<','<<iBBox%nBox<<'\t';
                        cout<<"iCBox: "<<iCBox/nBox<<','<<iCBox%nBox<<'\t';
                    #endif
                    // 由iBox计算Box内第一个元素在原矩阵的索引：idx_Element = x_idx + N * y_idx
                    int idxA = block_size*(iABox/nBox)+block_size*N*(iABox%nBox);
                    int idxB = block_size*(iBBox/nBox)+block_size*N*(iBBox%nBox);
                    int idxC = block_size*(iCBox/nBox)+block_size*N*(iCBox%nBox);
                    #ifdef _DEBUG_INFO
                        cout<<"idxA: "<<idxA/N<<','<<idxA%N<<'\t';
                        cout<<"idxB: "<<idxB/N<<','<<idxB%N<<'\t';
                        cout<<"idxC: "<<idxC/N<<','<<idxC%N<<endl;
                    #endif
                        for (int i = 0; i < block_size; i++){
                            for (int j = 0; j < block_size; j++){
                                // c[idxC + i * N + j] = 0.0;
                                for (int k = 0; k < block_size; k++){
                                    c[idxC + i * N + j] += 
                                        a[idxA + i * N + k] * b[idxB + k * N + j];
                                }
                            }
                        }
                        iABox += nBox; // ABox向下移动一个Box位置
                        iCBox += nBox; // CBox向下移动一个Box位置
                }
                iBBox++; // BBox移动到下一个位置
            }

            break;
        }
        case SIMD:
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
            matrix_mut_simd(a,b,c,N);
            break;
        case GPU:
            break;
        default:
            break;
        }
    }
    void matrix_mut_simd(float *a, float *b, float *c, int N){
        int nBox = N/_VF32_SIZE; // 当前_VF32_SIZE大小下，分块后矩阵的规模
        int iABox = 0, iBBox = 0, iCBox = 0; // 原A、B、C矩阵分块后得到的分块矩阵的索引,Box对应与原矩阵就是一个方形选框
        for(int i=0;i<nBox*nBox;i++){ // 顶层循环：固定BBox，纵向移动ABox和CBox
            iABox = iBBox/nBox+0; // 由BBox所在行决定ABox所在列
            iCBox = iBBox%nBox+0; // 由BBox所在列决定CBox所在列
            //#pragma omp parallel for num_threads(n_thread)
            for(int j=0;j<nBox;j++){
                #ifdef _DEBUG_INFO
                    cout<<"iABox: "<<iABox/nBox<<','<<iABox%nBox<<'\t';
                    cout<<"iBBox: "<<iBBox/nBox<<','<<iBBox%nBox<<'\t';
                    cout<<"iCBox: "<<iCBox/nBox<<','<<iCBox%nBox<<'\t';
                #endif
                // 由iBox计算Box内第一个元素在原矩阵的索引：idx_Element = x_idx + N * y_idx
                int idxA = _VF32_SIZE*(iABox/nBox)+_VF32_SIZE*N*(iABox%nBox);
                int idxB = _VF32_SIZE*(iBBox/nBox)+_VF32_SIZE*N*(iBBox%nBox);
                int idxC = _VF32_SIZE*(iCBox/nBox)+_VF32_SIZE*N*(iCBox%nBox);
                #ifdef _DEBUG_INFO
                    cout<<"idxA: "<<idxA/N<<','<<idxA%N<<'\t';
                    cout<<"idxB: "<<idxB/N<<','<<idxB%N<<'\t';
                    cout<<"idxC: "<<idxC/N<<','<<idxC%N<<endl;
                #endif
                matrixF32_madd(&c[idxC], &a[idxA], &b[idxB], N);
                iABox += nBox; // ABox向下移动一个Box位置
                iCBox += nBox; // CBox向下移动一个Box位置
            }
            iBBox++; // BBox移动到下一个位置
        }
    }
#endif
#ifdef SORT
    #ifdef R_SORT
        void radix_sort(int *arr, const long long offset, const long long len){
            arr += offset;
            int n_bin = pow(2,RADIX);
            int l_bin = len;
            int n_seg = sizeof(int)*8 / RADIX; // number of the turns of radix sorting
            
            // init bins
            #ifdef DEBUG
            cout<<"init bins"<<endl;
            cout<<"radix = "<<RADIX<<endl;
            #endif // DEBUG
            int **arr_bin = new int *[n_bin];
            for(int i=0;i<n_bin;i++){
                arr_bin[i] = new int [l_bin];
            }
            int *count_bin = new int[n_bin]; // counter array for every bin
            // init cache buf
            int **arr_buf = new int *[n_bin];
            for(int i=0;i<n_bin;i++){
                arr_buf[i] = new int [BUF_BIN_SIZE];
            }
            int *count_buf = new int [n_bin];

            // SORT
            int mask = pow(2,RADIX)-1; // init mask for its bit length
            for(int i=0;i<n_seg;i++){ // outside loop for different segments' radix sorting
                // int* p_to_element = arr;
                cout<<"Now sorting seg No."<<i<<'\r';
                #ifdef DEBUG
                bitset<32> bin_mask(mask);
                cout<<"n_seg = "<<i<<" | mask: "<<bin_mask<<endl;
                #endif // DEBUG
                for(int j=0;j<len;j++){ // inside loop for simple SORT of every segment position
                    // get the index of target bin
                    int i_bin = arr[j]&mask; 
                    i_bin = i_bin >> i*RADIX;
                    #ifdef DEBUG
                    bitset<32> bin_element(arr[j]);
                    cout<<"At seg No."<<i<<" | element No."<<j<<" | bin No."<<i_bin<<" , counter="<<count_buf[i_bin];
                    // cout<<" | bin No."<<i_bin;
                    // if(arr[j]==1585990364) cout<<" <====== B68";
                    // else if(arr[j]==1548233367) cout<< " <====== A69";
                    cout<<endl;
                    #endif // DEBUG
                    // arr_bin[i_bin][count_bin[i_bin]] = arr[j];
                    // count_bin[i_bin]++;
                    arr_buf[i_bin][count_buf[i_bin]] = arr[j];
                    count_buf[i_bin]++;
                    if(count_buf[i_bin]==BUF_BIN_SIZE){ // if any buffer bin is full, write all its elements back to memory
                        #ifdef DEBUG
                        cout<<"Buf bin No."<<i_bin<<" is full"<<endl;
                        #endif // DEBUG
                        for(int k=0;k<BUF_BIN_SIZE;k++){
                            arr_bin[i_bin][count_bin[i_bin]] = arr_buf[i_bin][k];
                            count_bin[i_bin]++;
                        }
                        count_buf[i_bin] = 0; // reset the counter of No.i_bin buffer bin
                    }
                }
                for(int i_bin=0;i_bin<n_bin;i_bin++){ // write the rest elements in buffer bin back to memory
                    for(int k=0;k<count_buf[i_bin];k++){
                        arr_bin[i_bin][count_bin[i_bin]] = arr_buf[i_bin][k];
                        count_bin[i_bin]++;
                    }
                    count_buf[i_bin] = 0;
                }
                // **Write the elements in bins back to arr (maybe later should use another arr_bin to switch)
                int *p_tmp = arr;
                #ifdef DEBUG
                cout<<"Write back"<<endl;
                #endif // DEBUG
                for(int i_bin=0;i_bin<n_bin;i_bin++){
                    #ifdef DEBUG
                    // cout<<"Write back bin No."<<i_bin<<" total elements: "<<count_bin[i_bin]<<endl;
                    #endif // DEBUG
                    for(int k=0;k<count_bin[i_bin];k++){
                        int tmp = arr_bin[i_bin][k];
                        *p_tmp = arr_bin[i_bin][k];
                        #ifdef DEBUG
                        // if(tmp==1585990364) cout<<" B68 write back, add is arr_bin["<<i_bin<<"]["<<k<<"] "<<&arr_bin[i_bin][k]<<endl;
                        // else if(tmp==1548233367) cout<< " A69 write back, add is arr_bin["<<i_bin<<"]["<<k<<"] "<<&arr_bin[i_bin][k]<<endl;
                        #endif // DEBUG
                        p_tmp++;
                    }
                    count_bin[i_bin] = 0;
                }
                // update mask
                mask = mask << RADIX; // shift mask
                cout<<endl;
            }
        }
    #endif
    #ifdef MT_R_SORT
        void* thread_exec_rsort(void *p_para){
            // int i_thread = para->i_thread;
            rsort_thr_para para = *(rsort_thr_para*)p_para;
            int block_len = N/THREAD_NUM;
            int offset = para.i_thread*block_len;
            #ifdef DEBUG
            cout<<"Thread No."<<para.i_thread<<" offset: "<<offset<<endl;
            #endif // DEBUG
            radix_sort(para.arr, offset, block_len);
        }
        void merge(int *arr, int *out_arr){
            int block_len = N/THREAD_NUM;
            int i_element[THREAD_NUM] = {0};
            int min;
            int last_i_block;
            for(int i=0;i<N;i++){
                min = RAND_MAX;
                last_i_block = 0;
                for(int i_block=0;i_block<THREAD_NUM;i_block++){
                    if(i_element[i_block]>=block_len) continue;
                    if (min > arr[i_block * block_len + i_element[i_block]]){ // find new min
                        min = arr[i_block * block_len + i_element[i_block]];
                        last_i_block = i_block;
                    } 
                }

                #ifdef DEBUG
                cout<<"============================"<<endl;
                cout<<min<<", from block."<<last_i_block<<".["<<i_element[last_i_block]-1<<"]"<<" <<<<<<<<<<<< MIN <<"<<endl;
                for(int i_block=0;i_block<THREAD_NUM;i_block++){
                    cout<<arr[i_block * block_len + i_element[i_block]]<<", from block."<<i_block<<".["<<i_element[i_block]<<"]"<<endl;
                }
                #endif // DEBUG
                out_arr[i] = min;
                i_element[last_i_block]++;
            }
        }
    #endif
    #ifdef M_SORT
        void* unit_merge(void* vpara){ // len = M/THREAD_NUM, M = C/2e (C is the L2 Cache size)
            msort_thr_para *para = (msort_thr_para*)vpara;
            int n_subarr = para->n_sub; // num of sub-array need to be merge
            int len = para->len; // length of entire unsorted array
            #ifdef DEBUG
            cout<<"-------------len = "<<len<<endl;
            #endif
            int l_subarr = len/n_subarr; // length of sub-array
            int *arr = para->unsorted_arr;
            int *out_arr = para->sorted_arr;
            #ifdef OUT_LOG
                fstream LOG;
                LOG.open("LOG_T"+to_string(para->i_thread),ios::out);
            #endif
            #ifdef DEBUG
            cout<<"init out_arr: "<<endl;
            for(int i=0;i<len;i++){
                cout<<out_arr[i]<<endl;
            }
            #endif // DEBUG
            for(int i=0;;i++){
                #ifdef DEBUG
                    cout<<">>>>> T."<<para->i_thread <<" Merge iteration NO."<<i<<" | n_subarr = "<<n_subarr<<endl;
                    #ifdef OUT_LOG
                        LOG<<">>>>> T."<<para->i_thread <<" Merge iteration NO."<<i<<" | n_subarr = "<<n_subarr<<endl;
                    #endif
                #endif // DEBUG
                for(int i_subarr=0;i_subarr<n_subarr;i_subarr+=2){
                    #ifdef DEBUG
                        cout<<"i_subarr = "<<i_subarr<<endl;
                    #endif // DEBUG
                    int i_out = 0; // the idx of out arr
                    for(int i_in1=0, i_in2=0;;){
                        #ifdef DEBUG
                            cout<<"i_in1 = "<<i_in1<<" | in_2 = "<<i_in2<<" | i_out = "<<i_out<<endl;
                            #ifdef OUT_LOG
                            LOG<<"i_in1 = "<<i_in1<<" | in_2 = "<<i_in2<<" | i_out = "<<i_out<<endl;
                            #endif                            
                        #endif // DEBUG

                        // if any subarr is empty
                        if(!(i_in1<l_subarr)){
                            while(i_in2<l_subarr){
                                out_arr[i_subarr*l_subarr+i_out] = arr[(i_subarr+1)*l_subarr+i_in2];
                                i_in2++;
                                i_out++;
                            }
                            break;
                        }
                        else if (!(i_in2<l_subarr)) {
                            while(i_in1<l_subarr){
                                out_arr[i_subarr*l_subarr+i_out] = arr[i_subarr*l_subarr+i_in1];
                                i_in1++;
                                i_out++;
                            }
                            break;
                        }

                        if(arr[i_subarr*l_subarr+i_in1]<arr[(i_subarr+1)*l_subarr+i_in2]){ // a<b
                            out_arr[i_subarr*l_subarr+i_out] = arr[i_subarr*l_subarr+i_in1];
                            i_in1++;
                        }
                        else{
                            out_arr[i_subarr*l_subarr+i_out] = arr[(i_subarr+1)*l_subarr+i_in2];
                            i_in2++;
                        }
                        i_out++;
                    }
                    #ifdef DEBUG
                    for(int j=0;j<len;j++){
                        cout<<"arr: "<<arr[j]<<" | "<<"out: "<<out_arr[j]<<endl;
                    }
                    #ifdef OUT_LOG
                    for (int j = 0; j < len; j++){
                        LOG << "arr: " << arr[j] << " | " << "out: " << out_arr[j] << endl;
                    }
                    #endif
                    #endif // DEBUG

                }
                n_subarr /= 2;
                l_subarr *= 2;
                if(n_subarr<2){
                    para->sorted_arr = out_arr;
                    break;
                }
                int *tmp = arr; // switch two pointers for next turn of merge
                arr = out_arr;
                out_arr = tmp;
            }
            #ifdef OUT_LOG
            LOG.close();
            #endif
        }
    #endif
    #ifdef MT_M_SORT
        int *msort_thread_assign(int *arr, int *sorted_arr, const int block_size, const int n_subarr_of_block, int n_thread){
            if(n_thread>(n_subarr_of_block+1)/2) n_thread = (n_subarr_of_block+1)/2;
            pthread_t id_thread[THREAD_NUM], ret[THREAD_NUM];
            msort_thr_para *para_list = new msort_thr_para[n_thread];
            int len = block_size/n_thread; // length of the array need a thread to merge
            for(int i=0;i<n_thread;i++){ // for the foundamental merge
                para_list[i].unsorted_arr = arr+i*len;
                para_list[i].sorted_arr = sorted_arr+i*len;
                #ifdef DEBUG
                cout<<"===========================CHECK OFFSET"<<endl;
                cout<<"i*block_size = "<<i*block_size<<endl;
                #endif
                para_list[i].len = len;
                para_list[i].n_sub = n_subarr_of_block/n_thread;
                para_list[i].i_thread = i;
                #ifdef DEBUG
                cout<<"CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCHECK"<<endl;
                for(int j=0;j<para_list[i].len;j++){
                    cout<<para_list[i].unsorted_arr[j]<<endl;
                }
                #endif
            }
            for(int i=0;i<n_thread;i++){
                ret[i] = pthread_create(&id_thread[i], NULL, unit_merge, (void*)(&para_list[i]));
                // ret[i] = pthread_create(&id_thread[i], NULL, _thread_exec, (void*)(idx_thread+i));
                if(ret[i]!=0){
                    cout<<"ERROR: thread No."<<i<<" create failed"<<endl;
                    break;
                }
            }
            for(int i=0;i<n_thread;i++){
                pthread_join(id_thread[i],NULL);
            }
            
            #ifdef DEBUG
            cout<<"SORTED"<<endl;
            for(int i=0;i<block_size;i++){
                cout<<para_list[0].sorted_arr[i]<<endl;
            }
            #endif // DEBUG
            return para_list[0].sorted_arr;
        }
    #endif
#endif
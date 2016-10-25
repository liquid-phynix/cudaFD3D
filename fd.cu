#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/poll.h>
#include <sys/time.h>
#include <cuda.h>

#include <pthread.h>
#include <readline/readline.h>
#include <readline/history.h>

#include <iostream>

#include "npyutil.hpp"

#include "fd.hh"

const char usageInfo[] =
"q: quit\n"
"h: help"
"i: current frame index";

bool runSim = true;
pthread_t workerTh, controlTh;

using namespace std;

struct Simulation {
    int m_fileCounter;

    float* m_hostField;
    float* m_deviceField;
    float* m_deviceFieldPrime;
    //  float* m_deviceFieldLap;

    dim3 m_blockSize;
    dim3 m_gridSize;

    double lx, ly, lz;
    //  float dxm2, dym2, dzm2;
    float ddm2;

    int m_dims[3];
    int m_elements;
    int m_size;

    Simulation(){
        cudaSetDevice(0);
        m_fileCounter = 0;

        int bx = 8, by = 8, bz = 8;
        int gx = 16, gy = 16, gz = 16;

        lx = 1;
        ly = 1;
        lz = 1;

        ddm2 = 4*M_PI/sqrt(3) / 9;

        //    double dx = lx / (bx * gx), dy = ly / (by * gy), dz = lz / (bz * gz);
        // dxm2 = 1.0; // / (dx * dx);
        // dym2 = 1.0; // / (dy * dy);
        // dzm2 = 1.0; // / (dz * dz);

        m_blockSize = dim3(bx, by, bz);
        m_gridSize = dim3(gx, gy, gz);
        m_dims[0] = bx * gx + 6;
        m_dims[1] = by * gy + 6;
        m_dims[2] = bz * gz + 6;
        m_elements = m_dims[0] * m_dims[1] * m_dims[2];
        m_size = sizeof(float) * m_elements;

        // host
        cudaMallocHost(&m_hostField, m_size);
        memset(m_hostField, 0, m_size);

        // device
        cudaMalloc(&m_deviceField, m_size);
        cudaMalloc(&m_deviceFieldPrime, m_size);
        //    cudaMalloc(&m_deviceFieldLap, m_size);

        if((not m_hostField) or (not m_deviceField) or (not m_deviceFieldPrime)){
            cout << "error allocating device memory for fields" << endl;
            cout << m_hostField << " " << m_deviceField << " " << m_deviceFieldPrime << endl;
            exit(-1);
        }

        cudaMemcpy(m_deviceField, m_hostField, m_size, cudaMemcpyHostToDevice);
        cudaMemcpy(m_deviceFieldPrime, m_hostField, m_size, cudaMemcpyHostToDevice);
        //    cudaMemcpy(m_deviceFieldLap, m_hostField, m_size, cudaMemcpyHostToDevice);
    }
    ~Simulation(void){
        cudaFreeHost(m_hostField);
        cudaFree(m_deviceField);
        cudaFree(m_deviceFieldPrime);
        //    cudaFree(m_deviceFieldLap);
    }
    void hostToDevice(){
        cudaMemcpy(m_deviceField, m_hostField, m_size, cudaMemcpyHostToDevice);
    }
};

Simulation* g_sim = NULL;


#define LAPLACE(sm, o) \
   ( sm[x + ox + 0 + o][y + oy + 1 + o][z + oz + 1 + o] + sm[x + ox + 2 + o][y + oy + 1 + o][z + oz + 1 + o] \
   + sm[x + ox + 1 + o][y + oy + 0 + o][z + oz + 1 + o] + sm[x + ox + 1 + o][y + oy + 2 + o][z + oz + 1 + o] \
   + sm[x + ox + 1 + o][y + oy + 1 + o][z + oz + 0 + o] + sm[x + ox + 1 + o][y + oy + 1 + o][z + oz + 2 + o] \
   - 6.0f * sm[x + ox + 1 + o][y + oy + 1 + o][z + oz + 1 + o])

__global__ void kernel_fill(float* in, float* out, int sx, int sy, int sz, float ddm2, float dt, float eps){

    __shared__ float cache     [14][14][14];
    __shared__ float cache_lap [14][14][14]; // 12x12x12
    __shared__ float cache_lap2[14][14][14]; // 10x10x10
    __shared__ float cache_lap3[14][14][14]; //  8x 8x 8

    int sq14 = 14 * 14;

    int x = threadIdx.x;
    int y = threadIdx.y;
    int z = threadIdx.z;

    int bx = blockIdx.x * 8;
    int by = blockIdx.y * 8;
    int bz = blockIdx.z * 8;

    int load_idx = (z * 8 + y) * 8 + x;
    int load_y = load_idx / 14;
    int load_z = load_idx % 14;

    int base_addr = ((bz + load_z) * sy + by + load_y) * sx + bx;

    if(load_idx < sq14){
        for(int load_x = 0; load_x < 14; load_x++){
            cache[load_x][load_y][load_z] = in[base_addr + load_x];
        }
    }

    __syncthreads();
    // cache is filled now

    // ********************************************************************************
    // compute lap  
    // phase 0,0,0
    int ox = 0; int oy = 0; int oz = 0;
    cache_lap[x + ox + 1][y + oy + 1][z + oz + 1] =
        ddm2 * LAPLACE(cache, 0);
//        (cache[x + ox + 0][y + oy + 1][z + oz + 1] + cache[x + ox + 2][y + oy + 1][z + oz + 1]
//         + cache[x + ox + 1][y + oy + 0][z + oz + 1] + cache[x + ox + 1][y + oy + 2][z + oz + 1]
//         + cache[x + ox + 1][y + oy + 1][z + oz + 0] + cache[x + ox + 1][y + oy + 1][z + oz + 2]
//         - 6.0f * cache[x + ox + 1][y + oy + 1][z + oz + 1]);
    __syncthreads();
    // phase x,0,0
    ox = 8; oy = 0; oz = 0;
    if(x < 4){
        cache_lap[x + ox + 1][y + oy + 1][z + oz + 1] =
            ddm2 * 
            (cache[x + ox + 0][y + oy + 1][z + oz + 1] + cache[x + ox + 2][y + oy + 1][z + oz + 1]
             + cache[x + ox + 1][y + oy + 0][z + oz + 1] + cache[x + ox + 1][y + oy + 2][z + oz + 1]
             + cache[x + ox + 1][y + oy + 1][z + oz + 0] + cache[x + ox + 1][y + oy + 1][z + oz + 2]
             - 6.0f * cache[x + ox + 1][y + oy + 1][z + oz + 1]);
    }
    __syncthreads();
    // phase 0,y,0
    ox = 0; oy = 8; oz = 0;
    if(y < 4){
        cache_lap[x + ox + 1][y + oy + 1][z + oz + 1] =
            ddm2 * 
            (cache[x + ox + 0][y + oy + 1][z + oz + 1] + cache[x + ox + 2][y + oy + 1][z + oz + 1]
             + cache[x + ox + 1][y + oy + 0][z + oz + 1] + cache[x + ox + 1][y + oy + 2][z + oz + 1]
             + cache[x + ox + 1][y + oy + 1][z + oz + 0] + cache[x + ox + 1][y + oy + 1][z + oz + 2]
             - 6.0f * cache[x + ox + 1][y + oy + 1][z + oz + 1]);
    }
    __syncthreads();
    // phase 0,0,z
    ox = 0; oy = 0; oz = 8;
    if(z < 4){
        cache_lap[x + ox + 1][y + oy + 1][z + oz + 1] =
            ddm2 * 
            (cache[x + ox + 0][y + oy + 1][z + oz + 1] + cache[x + ox + 2][y + oy + 1][z + oz + 1]
             + cache[x + ox + 1][y + oy + 0][z + oz + 1] + cache[x + ox + 1][y + oy + 2][z + oz + 1]
             + cache[x + ox + 1][y + oy + 1][z + oz + 0] + cache[x + ox + 1][y + oy + 1][z + oz + 2]
             - 6.0f * cache[x + ox + 1][y + oy + 1][z + oz + 1]);
    }
    __syncthreads();
    // phase x,y,0
    ox = 8; oy = 8; oz = 0;
    if(x < 4 && y < 4){
        cache_lap[x + ox + 1][y + oy + 1][z + oz + 1] =
            ddm2 * 
            (cache[x + ox + 0][y + oy + 1][z + oz + 1] + cache[x + ox + 2][y + oy + 1][z + oz + 1]
             + cache[x + ox + 1][y + oy + 0][z + oz + 1] + cache[x + ox + 1][y + oy + 2][z + oz + 1]
             + cache[x + ox + 1][y + oy + 1][z + oz + 0] + cache[x + ox + 1][y + oy + 1][z + oz + 2]
             - 6.0f * cache[x + ox + 1][y + oy + 1][z + oz + 1]);
    }
    __syncthreads();
    // phase x,0,z
    ox = 8; oy = 0; oz = 8;
    if(x < 4 && z < 4){
        cache_lap[x + ox + 1][y + oy + 1][z + oz + 1] =
            ddm2 * 
            (cache[x + ox + 0][y + oy + 1][z + oz + 1] + cache[x + ox + 2][y + oy + 1][z + oz + 1]
             + cache[x + ox + 1][y + oy + 0][z + oz + 1] + cache[x + ox + 1][y + oy + 2][z + oz + 1]
             + cache[x + ox + 1][y + oy + 1][z + oz + 0] + cache[x + ox + 1][y + oy + 1][z + oz + 2]
             - 6.0f * cache[x + ox + 1][y + oy + 1][z + oz + 1]);
    }
    __syncthreads();
    // phase 0,y,z
    ox = 0; oy = 8; oz = 8;
    if(y < 4 && z < 4){
        cache_lap[x + ox + 1][y + oy + 1][z + oz + 1] =
            ddm2 * 
            (cache[x + ox + 0][y + oy + 1][z + oz + 1] + cache[x + ox + 2][y + oy + 1][z + oz + 1]
             + cache[x + ox + 1][y + oy + 0][z + oz + 1] + cache[x + ox + 1][y + oy + 2][z + oz + 1]
             + cache[x + ox + 1][y + oy + 1][z + oz + 0] + cache[x + ox + 1][y + oy + 1][z + oz + 2]
             - 6.0f * cache[x + ox + 1][y + oy + 1][z + oz + 1]);
    }
    __syncthreads();
    // phase x,y,z
    ox = 8; oy = 8; oz = 8;
    if(x < 4 && y < 4 && z < 4){
        cache_lap[x + ox + 1][y + oy + 1][z + oz + 1] =
            ddm2 * 
            (cache[x + ox + 0][y + oy + 1][z + oz + 1] + cache[x + ox + 2][y + oy + 1][z + oz + 1]
             + cache[x + ox + 1][y + oy + 0][z + oz + 1] + cache[x + ox + 1][y + oy + 2][z + oz + 1]
             + cache[x + ox + 1][y + oy + 1][z + oz + 0] + cache[x + ox + 1][y + oy + 1][z + oz + 2]
             - 6.0f * cache[x + ox + 1][y + oy + 1][z + oz + 1]);
    }
    __syncthreads();
    // ********************************************************************************


    // ********************************************************************************
    // compute lap2
    // phase 0,0,0
    ox = 0; oy = 0; oz = 0;
    cache_lap2[x + ox + 2][y + oy + 2][z + oz + 2] =
        ddm2 * LAPLACE(cache_lap, 1);
//        (cache_lap[x + ox + 1][y + oy + 2][z + oz + 2] + cache_lap[x + ox + 3][y + oy + 2][z + oz + 2]
//         + cache_lap[x + ox + 2][y + oy + 1][z + oz + 2] + cache_lap[x + ox + 2][y + oy + 3][z + oz + 2]
//         + cache_lap[x + ox + 2][y + oy + 2][z + oz + 1] + cache_lap[x + ox + 2][y + oy + 2][z + oz + 3]
//         - 6.0f * cache_lap[x + ox + 2][y + oy + 2][z + oz + 2] );
    __syncthreads();
    // phase x,0,0
    ox = 8; oy = 0; oz = 0;
    if(x < 2){
        cache_lap2[x + ox + 2][y + oy + 2][z + oz + 2] =
            ddm2 *
            (cache_lap[x + ox + 1][y + oy + 2][z + oz + 2] + cache_lap[x + ox + 3][y + oy + 2][z + oz + 2]
             + cache_lap[x + ox + 2][y + oy + 1][z + oz + 2] + cache_lap[x + ox + 2][y + oy + 3][z + oz + 2]
             + cache_lap[x + ox + 2][y + oy + 2][z + oz + 1] + cache_lap[x + ox + 2][y + oy + 2][z + oz + 3]
             - 6.0f * cache_lap[x + ox + 2][y + oy + 2][z + oz + 2] );
    }
    __syncthreads();
    // phase 0,y,0
    ox = 0; oy = 8; oz = 0;
    if(y < 2){
        cache_lap2[x + ox + 2][y + oy + 2][z + oz + 2] =
            ddm2 *
            (cache_lap[x + ox + 1][y + oy + 2][z + oz + 2] + cache_lap[x + ox + 3][y + oy + 2][z + oz + 2]
             + cache_lap[x + ox + 2][y + oy + 1][z + oz + 2] + cache_lap[x + ox + 2][y + oy + 3][z + oz + 2]
             + cache_lap[x + ox + 2][y + oy + 2][z + oz + 1] + cache_lap[x + ox + 2][y + oy + 2][z + oz + 3]
             - 6.0f * cache_lap[x + ox + 2][y + oy + 2][z + oz + 2] );
    }
    __syncthreads();
    // phase 0,0,z
    ox = 0; oy = 0; oz = 8;
    if(z < 2){
        cache_lap2[x + ox + 2][y + oy + 2][z + oz + 2] =
            ddm2 *
            (cache_lap[x + ox + 1][y + oy + 2][z + oz + 2] + cache_lap[x + ox + 3][y + oy + 2][z + oz + 2]
             + cache_lap[x + ox + 2][y + oy + 1][z + oz + 2] + cache_lap[x + ox + 2][y + oy + 3][z + oz + 2]
             + cache_lap[x + ox + 2][y + oy + 2][z + oz + 1] + cache_lap[x + ox + 2][y + oy + 2][z + oz + 3]
             - 6.0f * cache_lap[x + ox + 2][y + oy + 2][z + oz + 2] );
    }
    __syncthreads();
    // phase x,y,0
    ox = 8; oy = 8; oz = 0;
    if(x < 2 && y < 2){
        cache_lap2[x + ox + 2][y + oy + 2][z + oz + 2] =
            ddm2 *
            (cache_lap[x + ox + 1][y + oy + 2][z + oz + 2] + cache_lap[x + ox + 3][y + oy + 2][z + oz + 2]
             + cache_lap[x + ox + 2][y + oy + 1][z + oz + 2] + cache_lap[x + ox + 2][y + oy + 3][z + oz + 2]
             + cache_lap[x + ox + 2][y + oy + 2][z + oz + 1] + cache_lap[x + ox + 2][y + oy + 2][z + oz + 3]
             - 6.0f * cache_lap[x + ox + 2][y + oy + 2][z + oz + 2] );
    }
    __syncthreads();
    // phase x,0,z
    ox = 8; oy = 0; oz = 8;
    if(x < 2 && z < 2){
        cache_lap2[x + ox + 2][y + oy + 2][z + oz + 2] =
            ddm2 *
            (cache_lap[x + ox + 1][y + oy + 2][z + oz + 2] + cache_lap[x + ox + 3][y + oy + 2][z + oz + 2]
             + cache_lap[x + ox + 2][y + oy + 1][z + oz + 2] + cache_lap[x + ox + 2][y + oy + 3][z + oz + 2]
             + cache_lap[x + ox + 2][y + oy + 2][z + oz + 1] + cache_lap[x + ox + 2][y + oy + 2][z + oz + 3]
             - 6.0f * cache_lap[x + ox + 2][y + oy + 2][z + oz + 2] );
    }
    __syncthreads();
    // phase 0,y,z
    ox = 0; oy = 8; oz = 8;
    if(y < 2 && z < 2){
        cache_lap2[x + ox + 2][y + oy + 2][z + oz + 2] =
            ddm2 *
            (cache_lap[x + ox + 1][y + oy + 2][z + oz + 2] + cache_lap[x + ox + 3][y + oy + 2][z + oz + 2]
             + cache_lap[x + ox + 2][y + oy + 1][z + oz + 2] + cache_lap[x + ox + 2][y + oy + 3][z + oz + 2]
             + cache_lap[x + ox + 2][y + oy + 2][z + oz + 1] + cache_lap[x + ox + 2][y + oy + 2][z + oz + 3]
             - 6.0f * cache_lap[x + ox + 2][y + oy + 2][z + oz + 2] );
    }
    __syncthreads();
    // phase x,y,z
    ox = 8; oy = 8; oz = 8;
    if(x < 2 && y < 2 && z < 2){
        cache_lap2[x + ox + 2][y + oy + 2][z + oz + 2] =
            ddm2 *
            (cache_lap[x + ox + 1][y + oy + 2][z + oz + 2] + cache_lap[x + ox + 3][y + oy + 2][z + oz + 2]
             + cache_lap[x + ox + 2][y + oy + 1][z + oz + 2] + cache_lap[x + ox + 2][y + oy + 3][z + oz + 2]
             + cache_lap[x + ox + 2][y + oy + 2][z + oz + 1] + cache_lap[x + ox + 2][y + oy + 2][z + oz + 3]
             - 6.0f * cache_lap[x + ox + 2][y + oy + 2][z + oz + 2] );
    }
    __syncthreads();
    // ********************************************************************************

    // ********************************************************************************
    // compute lap3
    ox = 0; oy = 0; oz = 0;
    cache_lap3[x + 3][y + 3][z + 3] =
        ddm2 * LAPLACE(cache_lap2, 2);
//        ddm2 * (cache_lap2[x + 2][y + 3][z + 3] + cache_lap2[x + 4][y + 3][z + 3]
//         + cache_lap2[x + 3][y + 2][z + 3] + cache_lap2[x + 3][y + 4][z + 3]
//         + cache_lap2[x + 3][y + 3][z + 2] + cache_lap2[x + 3][y + 3][z + 4]
//         - 6.0f * cache_lap2[x + 3][y + 3][z + 3]);
    __syncthreads();
    // ********************************************************************************

    float c  = cache[x + 3][y + 3][z + 3];
    float xm = cache[x + 2][y + 3][z + 3];
    float xp = cache[x + 4][y + 3][z + 3];
    float ym = cache[x + 3][y + 2][z + 3];
    float yp = cache[x + 3][y + 4][z + 3];
    float zm = cache[x + 3][y + 3][z + 2];
    float zp = cache[x + 3][y + 3][z + 4];

    float lapPsi3 =
        ddm2 *
        (xm * xm * xm + xp * xp * xp
         + ym * ym * ym + yp * yp * yp
         + zm * zm * zm + zp * zp * zp
         - 6.0f * c * c * c);

    // ********************************************************************************
    // PFC
    out[((bz + z + 3) * sy + by + y + 3) * sx + bx + x + 3] = 
        cache[x + 3][y + 3][z + 3]
        + dt * (lapPsi3
                + (1.0f - eps) * cache_lap[x + 3][y + 3][z + 3]
                + 2.0f * cache_lap2[x + 3][y + 3][z + 3]
                + cache_lap3[x + 3][y + 3][z + 3]);
    // ********************************************************************************

    // ********************************************************************************
    // LAPLACE
    //  out[((bz + z + 3) * sy + by + y + 3) * sx + bx + x + 3] = 0.16666666666f * (cache_lap[x + 3][y + 3][z + 3] + 6.0f * cache[x + 3][y + 3][z + 3]);
    // ********************************************************************************

    // ********************************************************************************
    // ID
    // out[((bz + z + 3) * sy + by + y + 3) * sx + bx + x + 3] = cache_lap[x + 3][y + 3][z + 3];
    // ********************************************************************************

    // ********************************************************************************
    // ID
    // out[((bz + z + 3) * sy + by + y + 3) * sx + bx + x + 3] = in[((bz + z + 3) * sy + by + y + 3) * sx + bx + x + 3];
    // ********************************************************************************
}

__device__ int ifun(int sx, int sy, int sz, int x, int y, int z){ return (z * sy + y) * sx + x; }
#define I(x,y,z) ifun(sx, sy, sz, x, y, z)
__device__ int mod(float a, float b){ return ((int)a) - ((int)b) * floor(a / b); }
__device__ int wrap(int a, int b){ return mod(a - 3, b) + 3; }

__global__ void kernel_pbc(float* ar, int sx, int sy, int sz){
    int sxp = sx - 6;
    int syp = sy - 6;
    int szp = sz - 6;
    int a = threadIdx.x;
    int b = threadIdx.y;
    // threadblock: 16x16
    int x_mul_max = sx / 16;
    int y_mul_max = sy / 16;
    int z_mul_max = sz / 16;
    // x - y
    for(int x_mul = 0; x_mul <= x_mul_max; x_mul++){
        for(int y_mul = 0; y_mul <= y_mul_max; y_mul++){
            int xx = 16 * x_mul + a;
            int yy = 16 * y_mul + b;
            if(xx < sx && yy < sy){
                int xc = wrap(xx, sxp);
                int yc = wrap(yy, syp);
                ar[I(xx, yy, sz - 3)] = ar[I(xc, yc, 3)];
                ar[I(xx, yy, sz - 2)] = ar[I(xc, yc, 4)];
                ar[I(xx, yy, sz - 1)] = ar[I(xc, yc, 5)];
                ar[I(xx, yy, 2)] = ar[I(xc, yc, sz - 4)];
                ar[I(xx, yy, 1)] = ar[I(xc, yc, sz - 5)];
                ar[I(xx, yy, 0)] = ar[I(xc, yc, sz - 6)];
            }
        }
    }  
    // x - z
    for(int x_mul = 0; x_mul <= x_mul_max; x_mul++){
        for(int z_mul = 0; z_mul <= z_mul_max; z_mul++){
            int xx = 16 * x_mul + a;
            int zz = 16 * z_mul + b;
            if(xx < sx && zz < sz){
                int xc = wrap(xx, sxp);
                int zc = wrap(zz, szp);
                ar[I(xx, sy - 3, zz)] = ar[I(xc, 3, zc)];
                ar[I(xx, sy - 2, zz)] = ar[I(xc, 4, zc)];
                ar[I(xx, sy - 1, zz)] = ar[I(xc, 5, zc)];
                ar[I(xx, 2, zz)] = ar[I(xc, sy - 4, zc)];
                ar[I(xx, 1, zz)] = ar[I(xc, sy - 5, zc)];
                ar[I(xx, 0, zz)] = ar[I(xc, sy - 6, zc)];
            }
        }
    }
    // y - z
    for(int y_mul = 0; y_mul <= y_mul_max; y_mul++){
        for(int z_mul = 0; z_mul <= z_mul_max; z_mul++){
            int yy = 16 * y_mul + a;
            int zz = 16 * z_mul + b;
            if(yy < sy && zz < sz){
                int yc = wrap(yy, syp);
                int zc = wrap(zz, szp);
                ar[I(sx - 3, yy, zz)] = ar[I(3, yc, zc)];
                ar[I(sx - 2, yy, zz)] = ar[I(4, yc, zc)];
                ar[I(sx - 1, yy, zz)] = ar[I(5, yc, zc)];
                ar[I(2, yy, zz)] = ar[I(sx - 4, yc, zc)];
                ar[I(1, yy, zz)] = ar[I(sx - 5, yc, zc)];
                ar[I(0, yy, zz)] = ar[I(sx - 6, yc, zc)];
            }
        }
    }
}

__global__ void kernel_source(float* ar, int sx, int sy, int sz){
    ar[((3) * sy + sy / 2) * sx + sx / 2] = 1.0f;
}

void cumulativeTimerStop(cudaEvent_t startEvent, cudaEvent_t stopEvent, float* cumTm){
    float tm;
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&tm, startEvent, stopEvent);
    *cumTm += tm;
}

void cumulativeTimerStart(cudaEvent_t startEvent){
    cudaEventRecord(startEvent);
}

char* genFileName(char* fileName, const char* prefix, int fileCounter){
    char fileCounterString[SLEN];
    memset(fileName, 0, SLEN);
    strcat(fileName, prefix);
    sprintf(fileCounterString, "%.8d", fileCounter);
    strcat(fileName, fileCounterString);
    strcat(fileName, ".npy");
    return fileName;
}

void commandHandler(char* command){
    if(command){
        if(strcmp(command,"h") == 0){
            printf("%s\n",usageInfo);
        }else if(strcmp(command,"q") == 0){
            runSim = false;
        }else if(strcmp(command,"i")){
            cout << "i: " << g_sim->m_fileCounter << endl;
        }
        printf("\n~ %s",command);
        if (command[0]!=0)
            add_history(command);
    }
    else{
        runSim = false;
    }
}

void* controlThread(void* ptr){
    pollfd cinfd[1];
    rl_callback_handler_install ("\n$>", commandHandler);

    cinfd[0].fd = fileno(stdin);
    cinfd[0].events = POLLIN;

    rl_bind_key('\t',rl_abort);

    while(runSim){
        if(poll(cinfd, 1, 1)){
            rl_callback_read_char();
        }
    }
    rl_callback_handler_remove();
    return 0;
}

void saveFile(int fc, Simulation& sim){
    char fn[SLEN];
    genFileName(fn, "out_", fc);
    aoba::SaveArrayAsNumpy(fn, 3, sim.m_dims, sim.m_hostField);
}

void* workerThread(void* ptr){
    cudaError_t error;

    int iterations = ITERS;
    struct timeval startTime, endTime;
    Simulation sim;
    g_sim = &sim;

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    float gpuElapsedTime;

    // init field


    for(int i = 0; i < sim.m_elements; i++){
       sim.m_hostField[i] = -0.35 + (rand() / (float)RAND_MAX - 0.5) / 100.0;
    //    sim.m_hostField[i] = rand() / (float)RAND_MAX - 0.5;
    }

    sim.hostToDevice();
    // set PBC on field initially
    kernel_pbc<<<dim3(1), dim3(16, 16)>>>(sim.m_deviceField, sim.m_dims[0], sim.m_dims[1], sim.m_dims[2]);
    // set source
    kernel_source<<<dim3(1), dim3(1)>>>(sim.m_deviceField, sim.m_dims[0], sim.m_dims[1], sim.m_dims[2]);

    //saving
    saveFile(sim.m_fileCounter, sim);
    sim.m_fileCounter++;



    memset(sim.m_hostField, 0, sim.m_size);

    error = cudaThreadSynchronize();
    printf("start: %s\n", cudaGetErrorString(error));


    // main loop
    printf("initialization done\n");
    gettimeofday(&startTime, NULL);
    double time=0;
    while((iterations--) && runSim){    

    int inner_max = 10;
    double dt_expl = 0.00001;
    double nu = 0.0024;
    int inner = 0;
    while(inner < inner_max){
        inner++;
        double dt = dt_expl / (nu + 1. + (nu - 1.) * cos(M_PI/2. * (2.*inner - 1.) / inner_max));
        time += dt;

        if(sim.m_fileCounter % 1000 == 0){
            printf("%d. iteration t = %f\n", sim.m_fileCounter, time);
            // if((fileCounter % WRITEEVERY == 0) || iterations == 0){
            //   cudaMemcpy(sVars.hostFields, semaphore ? sVars.deviceFields1st : sVars.deviceFields2nd, sVars.fieldsNBytes, cudaMemcpyDeviceToHost);
            //   saveFields(sVars.hostFields, GD_X * BD_X, GD_Y * BD_Y, fileCounter);
            // }
        }
        cumulativeTimerStart(startEvent);

        // compute fieldPrime
        kernel_fill<<<sim.m_gridSize, sim.m_blockSize>>>
            (sim.m_deviceField, sim.m_deviceFieldPrime, sim.m_dims[0], sim.m_dims[1], sim.m_dims[2], sim.ddm2, dt, 0.35);

        // source
        kernel_source<<<dim3(1), dim3(1)>>>(sim.m_deviceFieldPrime, sim.m_dims[0], sim.m_dims[1], sim.m_dims[2]);

        // set PBC on fieldPrime
        kernel_pbc<<<dim3(1), dim3(16, 16)>>>
            (sim.m_deviceFieldPrime, sim.m_dims[0], sim.m_dims[1], sim.m_dims[2]);

        error = cudaThreadSynchronize();
        if(error != cudaSuccess){
            printf("error: %s\n", cudaGetErrorString(error));
            exit(-1);
        }

        // copy back field
        cudaMemcpy(sim.m_deviceField, sim.m_deviceFieldPrime, sim.m_size, cudaMemcpyDeviceToDevice);

        //saving
        if(sim.m_fileCounter % SAVE == 0){
            cudaMemcpy(sim.m_hostField, sim.m_deviceField, sim.m_size, cudaMemcpyDeviceToHost);
            saveFile(sim.m_fileCounter, sim);
        }

        cumulativeTimerStop(startEvent, stopEvent, &gpuElapsedTime);
        sim.m_fileCounter++;
    }
    }

    // timing
    cudaThreadSynchronize();
    gettimeofday(&endTime, NULL);
    printf("GPU timer: %d ms\n", (int)gpuElapsedTime);
    printf("CPU timer: %d ms\n",
            (int)(((endTime.tv_sec  - startTime.tv_sec) * 1000 
                    + (endTime.tv_usec - startTime.tv_usec)/1000.0) 
                + 0.5));

    runSim = false;

    return 0;
}

int main(){
    pthread_create(&workerTh,  NULL, workerThread,  NULL);
    pthread_create(&controlTh, NULL, controlThread, NULL);
    pthread_join(workerTh,  NULL);
    pthread_join(controlTh, NULL);
    cudaDeviceReset();
    return 0;
}

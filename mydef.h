#ifndef _MYDEF_H_
#define _MYDEF_H_
#include <cpu_bitmap.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <stdio.h>
#define pi (3.1415926535897932385)
#define INF (2e10f)
#define infinity (2e8f)
typedef float f32;
typedef double f64;
typedef char i8;
typedef int i32;
typedef long long i64;
typedef unsigned char u8;
typedef unsigned int u32;
typedef unsigned long long u64;
enum MATERIAL{LAMBERTIAN, METAL, DIELECTRIC};

static void HandleError( cudaError_t err,
    const char *file,
    int line ) {
if (err != cudaSuccess) {
printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
file, line );
exit( EXIT_FAILURE );
}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
__forceinline__ __device__ __host__
f64 degrees_to_radians(f64 degrees){
    return degrees * pi / 180.;
}
// __forceinline__ __device__ __host__
// f64 degrees_to_radians(f64 degrees){
//     return degrees * pi / 180.;
// }
// random on device
__device__ __forceinline__
f64 random_double(curandStateXORWOW_t* rand_state){
    return curand_uniform_double(rand_state);
}

__device__ __forceinline__
f64 random_double(f64 min, f64 max, curandStateXORWOW_t* rand_state){
    return min + (max-min)*curand_uniform_double(rand_state);
}

__forceinline__ __host__ __device__
f64 clamp(f64 x, f64 min, f64 max){
    if (x < min) return min;
    else if (x > max) return max;
    else return x;
}
#endif
#include <cuda.h>
#include <curand.h>
#include "mydef.h"
__global__
void kernel(float* dev_rand){
    curandStateXORWOW_t rand_state;
    u64 x = threadIdx.x + blockIdx.x * blockDim.x;
    u64 y = threadIdx.y + blockIdx.y * blockDim.y;
    u64 offset = x + y * blockDim.x * gridDim.x;
    u64 seed = offset;
    curand_init(seed, 0, 0, &rand_state);
    dev_rand[offset] = curand_uniform(&rand_state);
}
int main(){
    const int image_width = 16 * 16;
    const int image_height = 16 * 16;
    dim3 grids(image_width / 16, image_height / 16);
    dim3 threads(16,16);
    float* dev_rand;
    HANDLE_ERROR( cudaMalloc( (void**)&dev_rand, image_width*image_height*sizeof(float)));
    kernel<<<grids,threads>>>(dev_rand);
    float* host_rand;
    host_rand = (float*) malloc(image_width*image_height*sizeof(float));
    HANDLE_ERROR( cudaMemcpy(host_rand, dev_rand, image_width*image_height*sizeof(float), cudaMemcpyDeviceToHost) );
    cudaFree(dev_rand);
    for(size_t i = 0;i < image_width*image_height&&i<100;i++){
        printf("%f\n", host_rand[i]);
    }
    free(host_rand);
    return 0;
}
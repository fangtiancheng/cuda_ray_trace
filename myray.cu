#include <time.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hittable_list.h"
#include "camera.h"
#include "material.h"
#define max_depth (20)
#define debug printf("in file %s line %d\n",__FILE__,__LINE__);
__device__ __forceinline__
color ray_color(ray r, hittable_list*const dev_spheres, curandStateXORWOW_t* rand_state) {
    hit_record rec;
    vec3 mul = vec3::ones();
    i32 depth = max_depth;
    while((depth > 0)&&dev_spheres->hit(r, 0, infinity, rec)) {
        depth--;
        ray scattered;
        color attenuation;
        if(rec.mat_ptr->scatter(r, rec, attenuation, scattered, rand_state)){
            r = scattered;
            mul = cross_dot(mul,attenuation);
        }
        else{
            return color::zeros();
        }
    }
    if(depth == 0) return vec3::zeros();
    vec3 unit_direction = unit_vector(r.direction());
    f64 t = 0.5*(unit_direction.y + 1.0);
    return cross_dot(mul,((1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0)));
}

__device__ __forceinline__
void write_color(u8*const dev_ptr, const color& pixel_color, u32 samples_per_pixel, u64 offset){
    f64 r = pixel_color.x;
    f64 g = pixel_color.y;
    f64 b = pixel_color.z;
    f64 scale = 1.0/ samples_per_pixel;
    r = sqrt(scale * r);
    g = sqrt(scale * g);
    b = sqrt(scale * b);
    dev_ptr[offset*4 + 0] = (u8)(clamp(r, 0.0, 0.999) * 256);
    dev_ptr[offset*4 + 1] = (u8)(clamp(g, 0.0, 0.999) * 256);
    dev_ptr[offset*4 + 2] = (u8)(clamp(b, 0.0, 0.999) * 256);
    dev_ptr[offset*4 + 3] = (u8)255;
}

#define samples_per_pixel ((u32)200)
// Image
#define aspect_ratio (16.0 / 9.0)
#define image_width (16*16*3)
#define image_height (static_cast<int>(image_width / aspect_ratio))
// __constant__ Sphere s[SPHERES];
// Camera
#define viewport_height (2.0)
#define viewport_width (aspect_ratio * viewport_height)
#define focal_length (1.0)


#define origin (point3::zeros())
#define horizontal (vec3(viewport_width, 0, 0))
#define vertical (vec3(0, viewport_height, 0))
#define lower_left_corner (origin - horizontal/2 - vertical/2 - vec3(0, 0, focal_length))

__global__
void kernel( u8* const dev_ptr, hittable_list** dev_spheres, camera** dev_camera) {
    // map from threadIdx/BlockIdx to pixel position
    curandStateXORWOW_t rand_state;
    u64 x = threadIdx.x + blockIdx.x * blockDim.x;
    u64 y = threadIdx.y + blockIdx.y * blockDim.y;
    u64 offset = x + y * blockDim.x * gridDim.x;
    u64 seed = offset;
    curand_init(seed, 0, 0, &rand_state);
    color pixel_color = color::zeros();
    for (u32 s = 0; s < samples_per_pixel; ++s) {
        f64 u = (x + random_double(&rand_state)) / (image_width-1);
        f64 v = (y + random_double(&rand_state)) / (image_height-1);
        ray r = (*dev_camera)->get_ray(u, v);
        // pixel_color += ray_color_book(r, *dev_spheres, &rand_state, max_depth);
        pixel_color += ray_color(r, *dev_spheres, &rand_state);
    }
    write_color(dev_ptr, pixel_color, samples_per_pixel, offset);
}

// globals needed by the update routine
struct DataBlock {
    unsigned char   *dev_bitmap;
};

int main( void ) {

    // test();
    // main
    DataBlock   data;
    // // capture the start time
    // cudaEvent_t     start, stop;
    // HANDLE_ERROR( cudaEventCreate( &start ) );
    // HANDLE_ERROR( cudaEventCreate( &stop ) );
    // HANDLE_ERROR( cudaEventRecord( start, 0 ) );
    // make world
    hittable_list** dev_world = NULL;
    HANDLE_ERROR( cudaMalloc( (void**)&dev_world, sizeof(hittable_list**) ) );
    debug
    make_world<<<1, 1>>>(dev_world);
    cudaDeviceSynchronize();
    debug
    CPUBitmap bitmap( image_width, image_height, &data );
    camera** dev_camera = NULL;
    HANDLE_ERROR( cudaMalloc( (void**)&dev_camera, sizeof(camera**) ) );
    get_camera<<<1, 1>>>(dev_camera);
    cudaDeviceSynchronize();
    
    u8* dev_bitmap = NULL;
    debug
    // allocate memory on the GPU for the output bitmap
    HANDLE_ERROR( cudaMalloc( (void**)&dev_bitmap, bitmap.image_size() ) );
    // generate a bitmap from our sphere data
    dim3 grids(image_width/16,image_height/16);
    dim3 threads(16,16);
    debug
    cudaDeviceSynchronize();
    kernel<<<grids,threads>>>( dev_bitmap, dev_world, dev_camera);
    cudaDeviceSynchronize();
    debug
    // copy our bitmap back from the GPU for display
    HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(), dev_bitmap,bitmap.image_size(),cudaMemcpyDeviceToHost ) );
    
    // // get stop time, and display the timing results
    // HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    // HANDLE_ERROR( cudaEventSynchronize( stop ) );
    // float   elapsedTime;
    // HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,
    //                                     start, stop ) );
    // printf("time = %0.3f\n",elapsedTime);
    // HANDLE_ERROR( cudaEventDestroy( start ) );
    // HANDLE_ERROR( cudaEventDestroy( stop ) );

    HANDLE_ERROR( cudaFree( dev_bitmap ) );
    destroy_world<<<1, 1>>>(dev_world);
    destroy_camera<<<1, 1>>>(dev_camera);
    HANDLE_ERROR( cudaFree( dev_world ) );
    HANDLE_ERROR( cudaFree( dev_camera ) );

    // display
    bitmap.display_and_exit();
}


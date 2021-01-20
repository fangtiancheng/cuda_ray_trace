/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

#include <time.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hittable_list.h"
#include "camera.h"
#define rnd( x ) (x * rand() / RAND_MAX)
#define max_depth (50)
__device__ __forceinline__
color ray_color(ray r, sphere*const dev_spheres, curandStateXORWOW_t* rand_state) {
    hit_record rec;
    f32 mul = 1.;
    i32 depth = max_depth;
    while((depth > 0)&&world::hit(r, 0, infinity, rec, dev_spheres)) {
        depth--;
        point3 target = rec.p + random_in_hemisphere(rec.normal, rand_state);
        r = ray(rec.p, target - rec.p);
        mul *= 0.5;
        // return 0.5 * ray_color(ray(rec.p, target - rec.p), dev_spheres, rand_state);
    }
    if(depth == 0) return vec3::zeros();
    vec3 unit_direction = unit_vector(r.direction());
    f32 t = 0.5*(unit_direction.y + 1.0);
    return mul*((1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0));
}
__device__ __forceinline__
void write_color(u8*const dev_ptr, const color& pixel_color, u32 samples_per_pixel, u64 offset){
    f32 r = pixel_color.x;
    f32 g = pixel_color.y;
    f32 b = pixel_color.z;
    f32 scale = 1.0/ samples_per_pixel;
    r = sqrtf(scale * r);
    g = sqrtf(scale * g);
    b = sqrtf(scale * b);
    dev_ptr[offset*4 + 0] = (u8)(clamp(r, 0.0, 0.999) * 256);
    dev_ptr[offset*4 + 1] = (u8)(clamp(g, 0.0, 0.999) * 256);
    dev_ptr[offset*4 + 2] = (u8)(clamp(b, 0.0, 0.999) * 256);
    dev_ptr[offset*4 + 3] = (u8)255;
}

#define samples_per_pixel ((u32)100)
// Image
#define aspect_ratio (16.0 / 9.0)
#define image_width (16*16*5)
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

__global__ __forceinline__
void kernel( u8* const dev_ptr, sphere* dev_spheres, camera* dev_camera) {
    // map from threadIdx/BlockIdx to pixel position
    curandStateXORWOW_t rand_state;
    u64 x = threadIdx.x + blockIdx.x * blockDim.x;
    u64 y = threadIdx.y + blockIdx.y * blockDim.y;
    u64 offset = x + y * blockDim.x * gridDim.x;
    u64 seed = offset;
    curand_init(seed, 0, 0, &rand_state);
    color pixel_color = color::zeros();
    for (u32 s = 0; s < samples_per_pixel; ++s) {
        f32 u = (x + random_double(&rand_state)) / (image_width-1);
        f32 v = (y + random_double(&rand_state)) / (image_height-1);
        ray r = dev_camera->get_ray(u, v);
        pixel_color += ray_color(r, dev_spheres, &rand_state);
    }
    write_color(dev_ptr, pixel_color, samples_per_pixel, offset);
}

// globals needed by the update routine
struct DataBlock {
    unsigned char   *dev_bitmap;
};

int main( void ) {
    // main
    DataBlock   data;
    // capture the start time
    cudaEvent_t     start, stop;
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );
    // make world
    sphere* dev_spheres = world::hittable_list();
    CPUBitmap bitmap( image_width, image_height, &data );
    u8* dev_bitmap;

    // allocate memory on the GPU for the output bitmap
    HANDLE_ERROR( cudaMalloc( (void**)&dev_bitmap, bitmap.image_size() ) );
    // generate a bitmap from our sphere data
    dim3 grids(image_width/16,image_height/16);
    dim3 threads(16,16);
    camera* dev_camera = get_camera();
    
    kernel<<<grids,threads>>>( dev_bitmap, dev_spheres, dev_camera);
    cudaDeviceSynchronize();

    // copy our bitmap back from the GPU for display
    HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(), dev_bitmap,bitmap.image_size(),cudaMemcpyDeviceToHost ) );
    
    // get stop time, and display the timing results
    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( stop ) );
    float   elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,
                                        start, stop ) );

    HANDLE_ERROR( cudaEventDestroy( start ) );
    HANDLE_ERROR( cudaEventDestroy( stop ) );

    HANDLE_ERROR( cudaFree( dev_bitmap ) );
    HANDLE_ERROR( cudaFree( dev_spheres ) );
    HANDLE_ERROR( cudaFree( dev_camera ) );

    // display
    bitmap.display_and_exit();
}


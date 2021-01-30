#ifndef _CAMERA_H_
#define _CAMERA_H_
#include "ray.h"
class camera {
private:
    point3 origin;
    point3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
public:
    __host__ __device__ __forceinline__
    camera() {
        printf("camera()\n");
        f64 aspect_ratio = 16.0 / 9.0;
        f64 viewport_height = 2.0;
        f64 viewport_width = aspect_ratio * viewport_height;
        f64 focal_length = 1.0;

        origin = point3(0, 0, 0);
        horizontal = vec3(viewport_width, 0.0, 0.0);
        vertical = vec3(0.0, viewport_height, 0.0);
        lower_left_corner = origin - horizontal/2 - vertical/2 - vec3(0, 0, focal_length);
        printf("to exit camera()\n");
    }
    __device__
    ray get_ray(f64 u, f64 v) const {
        return ray(origin, lower_left_corner + u*horizontal + v*vertical - origin);
    }

    
};

__global__
void get_camera(camera** dev_cam){
    printf("get_camera()\n");
    *dev_cam = new camera();
    printf("to exit get_camera()\n");
    return;
    /*
    camera* dev_cam = NULL;
    HANDLE_ERROR( cudaMalloc( (void**)&dev_cam, sizeof(camera) ) );
    camera tmp_cam;
    HANDLE_ERROR( cudaMemcpy( dev_cam, &tmp_cam,sizeof(camera),cudaMemcpyHostToDevice ) );
    */
}
__global__
void destroy_camera(camera** dev_cam){
    delete *dev_cam;
    *dev_cam = NULL;
}
__global__
void test_camera(camera* dev_cam){
    dev_cam->get_ray(1.0,2.0);
}
#endif
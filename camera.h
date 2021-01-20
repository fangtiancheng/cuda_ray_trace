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
    __host__ __device__
    camera() {
        f64 aspect_ratio = 16.0 / 9.0;
        f64 viewport_height = 2.0;
        f64 viewport_width = aspect_ratio * viewport_height;
        f64 focal_length = 1.0;

        origin = point3(0, 0, 0);
        horizontal = vec3(viewport_width, 0.0, 0.0);
        vertical = vec3(0.0, viewport_height, 0.0);
        lower_left_corner = origin - horizontal/2 - vertical/2 - vec3(0, 0, focal_length);
    }
    __device__
    ray get_ray(f64 u, f64 v) const {
        return ray(origin, lower_left_corner + u*horizontal + v*vertical - origin);
    }

    
};

__host__
camera* get_camera(){
    camera* dev_cam;
    HANDLE_ERROR( cudaMalloc( (void**)&dev_cam, sizeof(camera) ) );
    camera tmp_cam;
    HANDLE_ERROR( cudaMemcpy( dev_cam, &tmp_cam,sizeof(camera),cudaMemcpyHostToDevice ) );
    return dev_cam;
}

#endif
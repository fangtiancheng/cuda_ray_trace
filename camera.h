#ifndef _CAMERA_H_
#define _CAMERA_H_
#include "ray.h"
#define _aspect_ratio (16.0 / 9.0)
class camera {
private:
    point3 origin;
    point3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    double lens_radius;
public:
    __host__ __device__ __forceinline__
    camera(
        point3 lookfrom,
        point3 lookat,
        vec3   vup,
        double vfov, // vertical field-of-view in degrees
        double aspect_ratio,
        double aperture,
        double focus_dist
    ) {
        auto theta = degrees_to_radians(vfov);
        auto h = tan(theta/2);
        auto viewport_height = 2.0 * h;
        auto viewport_width = aspect_ratio * viewport_height;

        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        origin = lookfrom;
        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;
        lower_left_corner = origin - horizontal/2 - vertical/2 - focus_dist*w;

        lens_radius = aperture / 2;
    }
    __device__
    ray get_ray(double s, double t, curandStateXORWOW_t* rand_state) const {
        vec3 rd = lens_radius * random_in_unit_disk(rand_state);
        vec3 offset = u * rd.x + v * rd.y;

        return ray(
            origin + offset,
            lower_left_corner + s*horizontal + t*vertical - origin - offset
        );
    }

    
};

__global__
void get_camera(camera** dev_cam){
    point3 lookfrom(13,2,3);
    point3 lookat(0,0,0);
    vec3 vup(0,1,0);
    auto dist_to_focus = 10.0;
    auto aperture = 0.1;
    *dev_cam = new camera(lookfrom, lookat, vup, 20, _aspect_ratio, aperture, dist_to_focus);
    printf("to exit get_camera()\n");
}
__global__
void destroy_camera(camera** dev_cam){
    delete *dev_cam;
    *dev_cam = NULL;
}

#endif
#undef _aspect_ratio
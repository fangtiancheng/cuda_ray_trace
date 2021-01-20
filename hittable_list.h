#ifndef _HITTABLE_LIST_H
#define _HITTABLE_LIST_H
#include "hittable.h"
namespace world{
#define SPHERES 4

__host__
sphere* hittable_list(){
    // make world
    sphere* dev_spheres;
    const u64 size = sizeof(sphere)*SPHERES;
    sphere* tmp_spheres = (sphere*) malloc(size);
    // make world
    tmp_spheres[0] = sphere(point3(0.,-100.5,-1.),100., material(color(0.8,0.8,0.0),LAMBERTIAN));
    tmp_spheres[1] = sphere(point3(0.,0.,-1.), 0.5, material(color(0.7,0.3,0.3),METAL,0));
    tmp_spheres[2] = sphere(point3(-1.,0.,-1.), 0.5, material(color(0.8,0.8,0.8),DIELECTRIC,1.5));
    tmp_spheres[3] = sphere(point3(1.,0.,-1.), 0.5, material(color(0.8,0.6,0.3),METAL,1.0));
    HANDLE_ERROR(cudaMalloc((void**)&dev_spheres, size));
    HANDLE_ERROR( cudaMemcpy( dev_spheres, tmp_spheres, size,cudaMemcpyHostToDevice) );
    free(tmp_spheres);
    return dev_spheres;
}
__device__
bool hit(const ray& r, f64 t_min, f64 t_max, hit_record& rec,sphere* dev_spheres){
    hit_record temp_rec;
    bool hit_anything = false;
    f64 closest_so_far = t_max;
    for (int i = 0;i < SPHERES;i++) {
        if (dev_spheres[i].hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

}// end namespace world
#endif
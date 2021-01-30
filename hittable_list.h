#ifndef _HITTABLE_LIST_H
#define _HITTABLE_LIST_H
#include "hittable.h"

#define SPHERES ((u64)4)


class hittable_list: public hittable{
public:
    hittable** objects;
public:
    __device__
    hittable_list(){
        printf("hittable_list()\n");
        // objects = (hittable**)malloc(SPHERES*sizeof(hittable*));
        objects = new hittable*[SPHERES];
        // make world
        printf("make 0\n");
        auto material_ground = new lambertian (color(0.8, 0.8, 0.0));
        auto material_center = new lambertian (color(0.7, 0.3, 0.3));
        auto material_left   = new metal (color(0.8, 0.8, 0.8), 0.);
        auto material_right  = new metal (color(0.8, 0.6, 0.2), 0.);
        objects[0] = new sphere(point3(0.,-100.5,-1.),100., material_ground);
        objects[1] = new sphere(point3(0.,0.,-1.), 0.5, material_center);
        objects[2] = new sphere(point3(-1,0.,-1.), 0.5, material_left);
        objects[3] = new sphere(point3(1.,0.,-1.), 0.5, material_right);
        printf("to exit hittable_list()\n");
    }
    __device__ __forceinline__
    virtual bool hit(const ray& r, f64 t_min, f64 t_max, hit_record& rec) const override {
        hit_record temp_rec;
        bool hit_anything = false;
        f64 closest_so_far = t_max;
        for (u64 i = 0;i < SPHERES;i++) {
            if (objects[i]->hit(r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }
    __device__
    ~hittable_list(){
        for(u64 i = 0;i < SPHERES;i++){
            delete objects[i];
        }
        // free(objects);
        delete[] objects;
        objects = NULL;
    }
};

__global__
void make_world(hittable_list** hl){
    // hl has its own space
    printf("make_world()\n");
    *hl = new hittable_list();
    printf("to exit make_world()\n");
}
__global__
void destroy_world(hittable_list** hl){
    // hl has its own space
    delete *hl;
    *hl = NULL;
}
#endif
#ifndef _HITTABLE_LIST_H
#define _HITTABLE_LIST_H
#include "hittable.h"

#define SPHERES (485)


class hittable_list: public hittable{
public:
    hittable** objects;
public:
    __device__
    hittable_list(curandStateXORWOW_t* rand_state){
        int sum = 0;
        objects = new hittable*[SPHERES];
        auto ground_material = new lambertian(color(0.5, 0.5, 0.5));
        objects[sum++] = new sphere (point3(0,-1000,0), 1000, ground_material);

        for (int a = -11; a < 11; a++) {// 22
            for (int b = -11; b < 11; b++) {// 22
                auto choose_mat = random_double(rand_state);
                point3 center(a + 0.9*random_double(rand_state), 0.2, b + 0.9*random_double(rand_state));

                if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                    material* sphere_material;

                    if (choose_mat < 0.8) {
                        // diffuse
                        auto albedo = color::random(rand_state) * color::random(rand_state);
                        sphere_material = new lambertian(albedo);
                        objects[sum++] = new sphere(center, 0.2, sphere_material);
                    } else if (choose_mat < 0.95) {
                        // metal
                        auto albedo = color::random(0.5, 1,rand_state);
                        auto fuzz = random_double(0, 0.5,rand_state);
                        sphere_material = new metal(albedo, fuzz);
                        objects[sum++] = new sphere(center, 0.2, sphere_material);
                    } else {
                        // glass
                        sphere_material = new dielectric(1.5);
                        objects[sum++] = new sphere(center, 0.2, sphere_material);
                    }
                }
            }
        }

        auto material1 = new dielectric(1.5);
        objects[sum++] = new sphere (point3(0, 1, 0), 1.0, material1);

        auto material2 = new lambertian(color(0.4, 0.2, 0.1));
        objects[sum++] = new sphere(point3(-4, 1, 0), 1.0, material2);

        auto material3 = new metal(color(0.7, 0.6, 0.5), 0.0);
        objects[sum++] = new sphere(point3(4, 1, 0), 1.0, material3);
        printf("sum = %d\n",sum);
        if(sum!=SPHERES) printf("No!!!\n");
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
    curandStateXORWOW_t rand_state;
    u64 seed = 19260817;
    curand_init(seed, 0, 0, &rand_state);
    *hl = new hittable_list(&rand_state);
    printf("to exit make_world()\n");
}
__global__
void destroy_world(hittable_list** hl){
    // hl has its own space
    delete *hl;
    *hl = NULL;
}
#endif
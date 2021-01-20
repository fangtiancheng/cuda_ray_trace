#ifndef _MATERIAL_H_
#define _MATERIAL_H_
#include "ray.h"
// #include "hittable.h"
/*
class material{
public:
    MATERIAL mat_enum;
    color albedo;
public:
    __device__ __host__ __forceinline__
    material():albedo(color::zeros()), mat_enum(METAL){}
    __device__ __host__ __forceinline__
    material(const color& c, MATERIAL mat):albedo(c),mat_enum(mat){}
    __device__
    bool lambertian_scatter(const ray& r_in,const hit_record& rec, color& attenuation, ray& scattered, curandStateXORWOW_t* rand_state){
        vec3 scatter_direction = rec.normal + random_unit_vector(rand_state);

        // Catch degenerate scatter direction
        if (scatter_direction.near_zero())
            scatter_direction = rec.normal;

        scattered = ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }
    __device__
    bool metal_scatter(const ray& r_in,const hit_record& rec, color& attenuation, ray& scattered, curandStateXORWOW_t* rand_state){
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected);
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }
    __device__ __forceinline__
    bool scatter(const ray& r_in,const hit_record& rec, color& attenuation, ray& scattered, curandStateXORWOW_t* rand_state){
        switch(mat_enum){
            case LAMBERTIAN:
                return lambertian_scatter(r_in,rec,attenuation,scattered,rand_state);
            case METAL:
                return metal_scatter(r_in,rec,attenuation,scattered,rand_state);
            default:
                return false;
        }
    }
};
*/
#endif
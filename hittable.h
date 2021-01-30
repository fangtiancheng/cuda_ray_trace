#ifndef _HITTABLE_H_
#define _HITTABLE_H_
#include "ray.h"
// #include "material.h"
class material;

struct hit_record {
    class material {
public:
    __device__
    virtual bool scatter(
        const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandStateXORWOW_t* rand_state
    ) const = 0;
};
    point3 p;
    vec3 normal;
    f64 t;
    bool front_face;
    material* mat_ptr;
    __forceinline__ __device__
    inline void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0.;
        normal = front_face ? outward_normal : -outward_normal;
    }
    // __forceinline__ __device__
    // hit_record& operator=(const hit_record& other){
    //     p = other.p;
    //     normal = other.normal;
    //     t = other.t;
    //     front_face = other.front_face;
    //     mat_ptr = other.mat_ptr;
    //     return *this;
    // }
};
#define material hit_record::material
class hittable {
public:
    __device__ __forceinline__
    virtual bool hit(const ray& r, f64 t_min, f64 t_max, hit_record& rec) const = 0;
};
#endif
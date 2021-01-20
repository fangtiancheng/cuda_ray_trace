#ifndef _HITTABLE_H_
#define _HITTABLE_H_
#include "ray.h"
#include "material.h"
struct hit_record {
    point3 p;
    vec3 normal;
    f64 t;
    bool front_face;
    material mat_ptr;
    __forceinline__ __device__
    inline void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0.;
        normal = front_face ? outward_normal : -outward_normal;
    }
    __forceinline__ __device__
    hit_record& operator=(const hit_record& other){
        p = other.p;
        normal = other.normal;
        t = other.t;
        front_face = other.front_face;
        return *this;
    }
};
#endif
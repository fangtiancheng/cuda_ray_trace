#ifndef _HITTABLE_H_
#define _HITTABLE_H_
#include "ray.h"
struct hit_record {
    point3 p;
    vec3 normal;
    f32 t;
    bool front_face;
    __forceinline__ __device__
    inline void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0.;
        normal = front_face ? outward_normal : -outward_normal;
    }
};
#endif
#ifndef _SPHERE_H_
#define _SPHERE_H_
#include "hittable.h"

class sphere {
public:
    point3 center;
    f32 radius;
public:
    __device__ __host__
    sphere() = default;
    __device__ __host__
    sphere(const point3& cen, f32 r):
    center(cen), radius(r) {}
    __device__ __host__
    sphere& operator=(const sphere& other){
        center = other.center;
        radius = other.radius;
        return *this;
    }
    __device__ __forceinline__ inline
    bool hit(const ray& r, f32 t_min, f32 t_max, hit_record& rec) const {
        vec3 oc = r.origin() - center;
        auto a = r.direction().length_squared();
        auto half_b = dot(oc, r.direction());
        auto c = oc.length_squared() - radius*radius;

        auto discriminant = half_b*half_b - a*c;
        if (discriminant < 0) return false;
        auto sqrtd = sqrtf(discriminant);

        // Find the nearest root that lies in the acceptable range.
        auto root = (-half_b - sqrtd) / a;
        if (root < t_min || t_max < root) {
            root = (-half_b + sqrtd) / a;
            if (root < t_min || t_max < root)
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);

        return true;
    }
};

#endif
#ifndef _SPHERE_H_
#define _SPHERE_H_
#include "hittable.h"
#include "material.h"
class sphere {
public:
    material mat_ptr;
    point3 center;
    f64 radius;
public:
    __device__ __host__
    sphere() = default;
    __device__ __host__
    sphere(const point3& cen, f64 r,const material& m):
    center(cen), radius(r), mat_ptr(m) {}
    // __device__ __host__
    // sphere& operator=(const sphere& other){
    //     center = other.center;
    //     radius = other.radius;
    //     mat_ptr = other.mat_ptr;
    //     return *this;
    // }
    __device__ __forceinline__ inline
    bool hit(const ray& r, f64 t_min, f64 t_max, hit_record& rec) const {
        vec3 oc = r.origin() - center;
        f64 a = r.direction().length_squared();
        f64 half_b = dot(oc, r.direction());
        f64 c = oc.length_squared() - radius*radius;

        f64 discriminant = half_b*half_b - a*c;
        if (discriminant < 0.) return false;
        f64 sqrtd = sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        f64 root = (-half_b - sqrtd) / a;
        if (root < t_min || t_max < root) {
            root = (-half_b + sqrtd) / a;
            if (root < t_min || t_max < root)
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        rec.mat_ptr = mat_ptr;

        return true;
    }
};

#endif
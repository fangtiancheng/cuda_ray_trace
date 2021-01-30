#ifndef _MATERIAL_H_
#define _MATERIAL_H_
#include "ray.h"
#include "hittable.h"
struct hit_record;
// class material {
// public:
//     __device__
//     virtual bool scatter(
//         const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandStateXORWOW_t* rand_state
//     ) const = 0;
// };
class lambertian : public material {
public:
    color albedo;
public:
    __device__
    lambertian(const color& a) : albedo(a) {}
    __device__
    virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandStateXORWOW_t* rand_state) const override {
        vec3 scatter_direction = rec.normal + random_unit_vector(rand_state);

        // Catch degenerate scatter direction
        if (scatter_direction.near_zero())
            scatter_direction = rec.normal;

        scattered = ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }
};

class metal : public material {
public:
    color albedo;
    f64 fuzz;
public:
    __device__
    metal(const color& a, f64 f) : albedo(a), fuzz(f<1?f:1) {}
    __device__
    virtual bool scatter(
        const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandStateXORWOW_t* rand_state
    ) const override {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz*random_in_unit_sphere(rand_state));
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }
};
#endif
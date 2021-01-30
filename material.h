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

class dielectric : public material {
public:
    f64 ir; // Index of Refraction
private:
    __device__
    static double reflectance(double cosine, double ref_idx) {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1-ref_idx) / (1+ref_idx);
        r0 = r0*r0;
        return r0 + (1-r0)*pow((1 - cosine),5);
    }
public:
    __device__
    dielectric(f64 index_of_refraction) : ir(index_of_refraction) {}
    __device__
    virtual bool scatter(
        const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandStateXORWOW_t* rand_state
    ) const override {
        attenuation = color(1.0, 1.0, 1.0);
            double refraction_ratio = rec.front_face ? (1.0/ir) : ir;

            vec3 unit_direction = unit_vector(r_in.direction());
            double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
            double sin_theta = sqrt(1.0 - cos_theta*cos_theta);

            bool cannot_refract = refraction_ratio * sin_theta > 1.0;
            vec3 direction;
            if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_double(rand_state))
                direction = reflect(unit_direction, rec.normal);
            else
                direction = refract(unit_direction, rec.normal, refraction_ratio);

            scattered = ray(rec.p, direction);
            return true;
    }
};

#endif
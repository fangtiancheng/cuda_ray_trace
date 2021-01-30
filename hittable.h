#ifndef _HITTABLE_H_
#define _HITTABLE_H_
#include "ray.h"
#include "material.h"

struct hit_record {
    class material{
public:
    MATERIAL mat_enum;
    color albedo;
    f64 fuzz;
    f64 ir;
public:
    __device__ __host__ __forceinline__
    material():albedo(color::zeros()), mat_enum(METAL){}
    __device__ __host__ __forceinline__
    material(const color& c, MATERIAL mat, f64 f = 0.):albedo(c),mat_enum(mat),fuzz(f<1?f:1),ir(f){}
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
    // __device__ __forceinline__
    // material& operator=(const material& other){
    //     mat_enum = other.mat_enum;
    //     albedo = other.albedo;
    //     fuzz = other.fuzz;
    //     ir = other.ir;
    //     return *this;
    // }
    __device__
    bool metal_scatter(const ray& r_in,const hit_record& rec, color& attenuation, ray& scattered, curandStateXORWOW_t* rand_state){
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz*random_in_unit_sphere(rand_state));
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }
    __device__
    bool dielectric_scatter(const ray& r_in,const hit_record& rec, color& attenuation, ray& scattered, curandStateXORWOW_t* rand_state){
        attenuation = color(1.0, 1.0, 1.0);
        f64 refraction_ratio = rec.front_face ? (1.0/ir) : ir;

        vec3 unit_direction = unit_vector(r_in.direction());
        f64 cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
        f64 sin_theta = sqrt(1.0 - cos_theta*cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0;
        vec3 direction;
        if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_double(rand_state))
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, refraction_ratio);

        scattered = ray(rec.p, direction);
        return true;
    }
    __device__ __forceinline__
    bool scatter(const ray& r_in,const hit_record& rec, color& attenuation, ray& scattered, curandStateXORWOW_t* rand_state){
        switch(mat_enum){
            case LAMBERTIAN:
                return lambertian_scatter(r_in,rec,attenuation,scattered,rand_state);
            case METAL:
                return metal_scatter(r_in,rec,attenuation,scattered,rand_state);
            case DIELECTRIC:
                return dielectric_scatter(r_in,rec,attenuation,scattered,rand_state);
            default:
                return false;
        }
    }
private:
    __device__
    static f64 reflectance(f64 cosine, f64 ref_idx) {
        // Use Schlick's approximation for reflectance.
        f64 r0 = (1-ref_idx) / (1+ref_idx);
        r0 = r0*r0;
        return r0 + (1-r0)*pow((1 - cosine),5);
    }
};
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
        mat_ptr = other.mat_ptr;
        return *this;
    }
};
#define material hit_record::material
class hittable {
public:
    __device__ __forceinline__
    virtual bool hit(const ray& r, f64 t_min, f64 t_max, hit_record& rec) const = 0;
};
#endif
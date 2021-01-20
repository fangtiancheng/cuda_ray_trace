#include "mydef.h"
#ifndef _VEC3_H_
#define _VEC3_H_

class vec3{
public:
    f64 x,y,z;
public:
    __device__ __host__
    vec3():x(0.),y(0.),z(0.){}
    __device__ __host__
    vec3(f64 a, f64 b, f64 c):
    x(a),y(b),z(c){}
    __device__ __host__
    vec3& operator=(const vec3& other){
        x = other.x;
        y = other.y;
        z = other.z;
        return *this;
    }
    __device__ __host__
    vec3 operator+(const vec3& other)const{
        return vec3(x+other.x,y+other.y,z+other.z);
    }
    __device__ __host__
    vec3& operator+=(const vec3& other){
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }
    __device__ __host__
    vec3 operator-(const vec3& other)const{
        return vec3(x-other.x,y-other.y,z-other.z);
    }
    __device__ __host__
    vec3 operator-()const{
        return vec3(-x,-y,-z);
    }
    __device__ __host__
    vec3& operator-=(const vec3& other){
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }
    __device__ __host__
    friend vec3 operator*(f64 f, const vec3& vc){
        return vec3(f*vc.x,f*vc.y,f*vc.z);
    }
    __device__ __host__
    friend f64 dot(const vec3& vc1, const vec3& vc2){
        return vc1.x*vc2.x+vc1.y*vc2.y+vc1.z*vc2.z;
    }
    __device__ __host__
    friend vec3 operator*(const vec3& vc,f64 f){
        return vec3(f*vc.x,f*vc.y,f*vc.z);
    }
    __device__ __host__
    vec3 operator*(const vec3& other) const {
        return vec3(x*other.x,y*other.y,z*other.z);
    }
    __device__ __host__
    vec3& operator*=(f64 f){
        x *= f; y *= f; z *= f;
        return *this;
    }
    __device__ __host__
    vec3 operator/(f64 f) const {
        return vec3(x/f,y/f,z/f);
    }
    __device__ __host__
    vec3& operator/=(f64 f){
        x /= f; y /= f; z /= f;
        return *this;
    }
    __device__ __host__
    f64 length_squared() const {
        return x*x + y*y + z*z;
    }
    __device__ __host__
    f64 length() const {
        return sqrt(x*x + y*y + z*z);
    }
    __forceinline__ __device__ __host__
    vec3 unit_vector() const {
        return *this / this->length();
    }
    __forceinline__ __device__ __host__
    friend vec3 unit_vector(const vec3& vc){
        return vc / vc.length();
    }
    __forceinline__ __device__ __host__
    inline static vec3 zeros(){
        return vec3(0.,0.,0.);
    }
    __forceinline__ __device__ __host__
    inline static vec3 ones(){
        return vec3(1.,1.,1.);
    }
    __forceinline__ __device__
    inline static vec3 random(curandStateXORWOW_t* rand_state){
        return vec3(curand_uniform_double(rand_state),curand_uniform_double(rand_state),curand_uniform_double(rand_state));
    }
    __forceinline__ __device__
    inline static vec3 random(f64 min, f64 max, curandStateXORWOW_t* rand_state){
        return vec3(random_double(min,max,rand_state),random_double(min,max,rand_state),random_double(min,max,rand_state));
    }
    __forceinline__ __device__ __host__
    inline bool near_zero() const {
        const f64 s = 1e-8;
        return abs(x)<s && abs(y) < s && abs(z) < s;
    }
    __forceinline__ __device__ __host__
    friend vec3 cross_dot(const vec3& a, const vec3& b){
        return vec3(a.x*b.x,a.y*b.y,a.z*b.z);
    }
};
__device__ __forceinline__
vec3 random_in_unit_sphere(curandStateXORWOW_t* rand_state){
    while(true){
        vec3 p = vec3::random(-1.,1.,rand_state);
        if(p.length_squared() >= 1.) continue;
        else return p;
    }
}
__device__ __forceinline__
vec3 random_unit_vector(curandStateXORWOW_t* rand_state) {
    return unit_vector(random_in_unit_sphere(rand_state));
}
__device__ __forceinline__
vec3 random_in_hemisphere(const vec3& normal, curandStateXORWOW_t* rand_state) {
    vec3 in_unit_sphere = random_in_unit_sphere(rand_state);
    if (dot(in_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return in_unit_sphere;
    else
        return -in_unit_sphere;
}
__device__ __forceinline__
vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2*dot(v,n)*n;
}
__device__ __forceinline__
vec3 refract(const vec3& uv, const vec3& n, f64 etai_over_etat) {
    f64 cos_theta = fmin(dot(-uv, n), 1.0);
    vec3 r_out_perp =  etai_over_etat * (uv + cos_theta*n);
    vec3 r_out_parallel = -sqrt(abs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}
typedef vec3 point3;
typedef vec3 color;

#endif
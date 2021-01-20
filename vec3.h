#include "mydef.h"
#ifndef _VEC3_H_
#define _VEC3_H_

class vec3{
public:
    f32 x,y,z;
public:
    __device__ __host__
    vec3():x(0.),y(0.),z(0.){}
    __device__ __host__
    vec3(f32 a, f32 b, f32 c):
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
    friend vec3 operator*(f32 f, const vec3& vc){
        return vec3(f*vc.x,f*vc.y,f*vc.z);
    }
    __device__ __host__
    friend f32 dot(const vec3& vc1, const vec3& vc2){
        return vc1.x*vc2.x+vc1.y*vc2.y+vc1.z*vc2.z;
    }
    __device__ __host__
    friend vec3 operator*(const vec3& vc,f32 f){
        return vec3(f*vc.x,f*vc.y,f*vc.z);
    }
    __device__ __host__
    f32 operator*(const vec3& other) const {
        return x*other.x+y*other.y+z*other.z;
    }
    __device__ __host__
    vec3& operator*=(f32 f){
        x *= f; y *= f; z *= f;
        return *this;
    }
    __device__ __host__
    vec3 operator/(f32 f) const {
        return vec3(x/f,y/f,z/f);
    }
    __device__ __host__
    vec3& operator/=(f32 f){
        x /= f; y /= f; z /= f;
        return *this;
    }
    __device__ __host__
    f32 length_squared() const {
        return x*x + y*y + z*z;
    }
    __device__ __host__
    f32 length() const {
        return sqrtf(x*x + y*y + z*z);
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
        return vec3(curand_uniform(rand_state),curand_uniform(rand_state),curand_uniform(rand_state));
    }
    __forceinline__ __device__
    inline static vec3 random(f32 min, f32 max, curandStateXORWOW_t* rand_state){
        return vec3(random_double(min,max,rand_state),random_double(min,max,rand_state),random_double(min,max,rand_state));
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
typedef vec3 point3;
typedef vec3 color;

#endif
#ifndef _RAY_H_
#define _RAY_H_
#include "vec3.h"
class ray{
public:
    point3 orig;
    vec3 dir;
public:
    __host__ __device__
    ray(){}
    __host__ __device__
    ray(const point3& origin, const vec3& direction):
    orig(origin), dir(direction){}
    __host__ __device__
    ray& operator=(const ray& other){
        orig = other.orig;
        dir = other.dir;
        return *this;
    }
    __host__ __device__
    point3 at(f32 t) const {
        return orig + t*dir;
    }
    __forceinline__ __host__ __device__
    point3 origin() const  { return orig; }
    __forceinline__ __host__ __device__
    vec3 direction() const { return dir; }
};
#endif
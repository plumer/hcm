#pragma once
#include <cassert>
#include <cmath>
#include <xmmintrin.h>
#include "misc.hpp"

namespace hcm {

class pt2;

class vec2 {
public:
	float x, y;

	vec2() : x{ 1.0f }, y{ 0.0f } {}
	vec2(float _x, float _y) : x{ _x }, y{ _y } {}
	explicit vec2(const pt2 &p);
	float & operator [] (int i) { return (&x)[i]; }
	float   operator [] (int i) const { return (&x)[i]; }

	vec2 operator + (const vec2 & rhs) const { return vec2{ x + rhs.x, y + rhs.y }; }
	vec2 operator - (const vec2 & rhs) const { return vec2{ x - rhs.x, y - rhs.y }; }
	vec2 operator * (float s) const { return vec2{ x * s, y*s }; }
	vec2 operator / (float s) const { assert(s != 0.0f); return vec2{ x / s, y / s }; }

	vec2 & operator += (const vec2 & rhs) { x += rhs.x; y += rhs.y; return *this; }
	vec2 & operator -= (const vec2 & rhs) { x -= rhs.x; y -= rhs.y; return *this; }
	vec2 & operator *= (float s) { x *= s; y *= s; return *this; }
	vec2 & operator /= (float s) { assert(s != 0.0f); x /= s; y /= s; return *this; }

	// Dot-product and cross-product
	float operator * (const vec2 & rhs) const { return x*rhs.x + y*rhs.y; }
	float operator ^ (const vec2 & rhs) const { return x*rhs.y - y*rhs.x; }
	float dot(const vec2 & rhs) const { return (*this)*rhs; }
	float cross(const vec2 & rhs) const { return (*this) ^ rhs; }

	// return vector in opposite direction
	vec2 operator - () const { return vec2{ -x, -y }; }

	float length() const { return std::sqrt(this->length_squared()); }
	float length_squared() const { return x*x + y*y; }
	void normalize() { float l = length(); assert(l > 0.0f); x /= l; y /= l; }
	vec2 normalized() const { float l = length(); assert(l > 0.0f); return vec2{ x / l, y / l }; }

	static vec2 xbase() {return vec2{1, 0};}
	static vec2 ybase() {return vec2{0, 1};}
};

static inline vec2 operator * (float s, const vec2 &v) { return v * s; }

class pt2 {
public:
	float x, y;
	pt2() : x{ 0.0f }, y{ 0.0f } {}
	pt2(float _x, float _y) : x{ _x }, y{ _y } {}
	explicit pt2(const vec2 &v) : x{ v.x }, y{ v.y } {};

	float & operator [] (int i) { return (&x)[i]; }
	float   operator [] (int i) const { return (&x)[i]; }

	// point +/+= vec, point -/-= vec
	pt2 operator + (const vec2 & rhs) const { return pt2{ x + rhs.x, y + rhs.y }; }
	pt2 operator - (const vec2 & rhs) const { return pt2{ x - rhs.x, y - rhs.y }; }

	pt2 & operator += (const vec2 & rhs) { x += rhs.x; y += rhs.y; return *this; }
	pt2 & operator -= (const vec2 & rhs) { x -= rhs.x; y -= rhs.y; return *this; }

	// point - point
	vec2 operator - (const pt2 & rhs) const { return vec2{ x - rhs.x, y - rhs.y }; }

	// static methods
	static pt2 origin() { return pt2{0, 0}; }
};

inline vec2::vec2(const pt2 &p) : x{ p.x }, y{ p.y } {}

inline pt2 operator + (const vec2 &v, const pt2 &p) {return p + v;}

class complex {
public:
	float re, im;
	complex() : re{ 1.0f }, im{ 0.0f } {}
	complex(float _re, float _im) : re{ _re }, im{ _im } {}

	complex operator+(const complex& rhs) { return complex{ re + rhs.re, im + rhs.im }; }
	complex operator-(const complex& rhs) { return complex{ re - rhs.re, im - rhs.im }; }
	complex operator*(const complex& rhs) { return complex{ re*rhs.re - im*rhs.im, re*rhs.im + im*rhs.re }; }
	complex operator/(const complex& rhs) { return (*this)*(rhs.conjugate()) / rhs.length(); }
	complex operator*(float s) { return complex{ re*s, im*s }; }
	complex operator/(float s) { return complex{ re / s, im / s }; }

	complex conjugate() const { return complex{ re, -im }; }
	float length() const {return std::sqrt(length_squared());}
	float length_squared() const { return re*re + im*im; }

	static complex rotater(float degree) {
		float radian = TO_RADIAN(degree);
		return complex{ std::cos(radian), std::sin(radian) };
	}
	vec2 operator * (const vec2 &v) { return vec2{ v.x*re - v.y*im, v.x*im + v.y*re }; }

	static complex i() {return complex{0, 1};}
};


}// namespace hcm
#pragma once
#include <cassert>
#include <cmath>
#include <xmmintrin.h>
#include "misc.hpp"

namespace hcm {

class Point2f;

class Vector2f {
public:
	float x, y;

	Vector2f() : x{ 1.0f }, y{ 0.0f } {}
	Vector2f(float _x, float _y) : x{ _x }, y{ _y } {}
	explicit Vector2f(const Point2f &p);
	float & operator [] (int i) { return (&x)[i]; }
	float   operator [] (int i) const { return (&x)[i]; }

	Vector2f operator + (const Vector2f & rhs) const { return Vector2f{ x + rhs.x, y + rhs.y }; }
	Vector2f operator - (const Vector2f & rhs) const { return Vector2f{ x - rhs.x, y - rhs.y }; }
	Vector2f operator * (float s) const { return Vector2f{ x * s, y*s }; }
	Vector2f operator / (float s) const { assert(s != 0.0f); return Vector2f{ x / s, y / s }; }

	Vector2f & operator += (const Vector2f & rhs) { x += rhs.x; y += rhs.y; return *this; }
	Vector2f & operator -= (const Vector2f & rhs) { x -= rhs.x; y -= rhs.y; return *this; }
	Vector2f & operator *= (float s) { x *= s; y *= s; return *this; }
	Vector2f & operator /= (float s) { assert(s != 0.0f); x /= s; y /= s; return *this; }

	// Dot-product and cross-product
	float operator * (const Vector2f & rhs) const { return x*rhs.x + y*rhs.y; }
	float operator ^ (const Vector2f & rhs) const { return x*rhs.y - y*rhs.x; }
	float dot(const Vector2f & rhs) const { return (*this)*rhs; }
	float cross(const Vector2f & rhs) const { return (*this) ^ rhs; }

	// return vector in opposite direction
	Vector2f operator - () const { return Vector2f{ -x, -y }; }

	float length() const { return std::sqrt(this->lengthSquared()); }
	float lengthSquared() const { return x*x + y*y; }
	void normalize() { float l = length(); assert(l > 0.0f); x /= l; y /= l; }
	Vector2f normalized() const { float l = length(); assert(l > 0.0f); return Vector2f{ x / l, y / l }; }

	static const Vector2f XBASE;
	static const Vector2f YBASE;
	static const Vector2f &XDIR;
	static const Vector2f &YDIR;
};

static inline Vector2f operator * (float s, const Vector2f &v) { return v * s; }

class Point2f {
public:
	float x, y;
	Point2f() : x{ 0.0f }, y{ 0.0f } {}
	Point2f(float _x, float _y) : x{ _x }, y{ _y } {}
	explicit Point2f(const Vector2f &v) : x{ v.x }, y{ v.y } {};

	float & operator [] (int i) { return (&x)[i]; }
	float   operator [] (int i) const { return (&x)[i]; }

	// point +/+= vec, point -/-= vec
	Point2f operator + (const Vector2f & rhs) const { return Point2f{ x + rhs.x, y + rhs.y }; }
	Point2f operator - (const Vector2f & rhs) const { return Point2f{ x - rhs.x, y - rhs.y }; }

	Point2f & operator += (const Vector2f & rhs) { x += rhs.x; y += rhs.y; return *this; }
	Point2f & operator -= (const Vector2f & rhs) { x -= rhs.x; y -= rhs.y; return *this; }

	// point - point
	Vector2f operator - (const Point2f & rhs) const { return Vector2f{ x - rhs.x, y - rhs.y }; }

	// static methods
	static Point2f origin() { return ORIGIN; }
	static const Point2f ORIGIN;
};

inline Vector2f::Vector2f(const Point2f &p) : x{ p.x }, y{ p.y } {}

inline Point2f operator + (const Vector2f &v, const Point2f &p) {return p + v;}

class Complex {
public:
	float re, im;
	Complex() : re{ 1.0f }, im{ 0.0f } {}
	Complex(float _re, float _im) : re{ _re }, im{ _im } {}

	Complex operator+(const Complex& rhs) { return Complex{ re + rhs.re, im + rhs.im }; }
	Complex operator-(const Complex& rhs) { return Complex{ re - rhs.re, im - rhs.im }; }
	Complex operator*(const Complex& rhs) { return Complex{ re*rhs.re - im*rhs.im, re*rhs.im + im*rhs.re }; }
	Complex operator/(const Complex& rhs) { return (*this)*(rhs.conjugate()) / rhs.length(); }
	Complex operator*(float s) { return Complex{ re*s, im*s }; }
	Complex operator/(float s) { return Complex{ re / s, im / s }; }

	Complex conjugate() const { return Complex{ re, -im }; }
	float length() const { return re*re + im*im; }

	static Complex rotater(float degree) {
		float radian = TO_RADIAN(degree);
		return Complex{ std::cos(radian), std::sin(radian) };
	}
	Vector2f operator * (const Vector2f &v) { return Vector2f{ v.x*re - v.y*im, v.x*im + v.y*re }; }

	static const Complex i;
};


}// namespace hcm
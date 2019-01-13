#pragma once
#include "config.hpp"
#include <cassert>
#include <cmath>

namespace hcm {

class cpt3;
class cnormal3;

class cvec3 {
public:
	float x, y, z;

	// Constructors
	cvec3() : x{ 0.0f }, y{ 0.0f }, z{ 0.0f } {}
	cvec3(float _x, float _y, float _z) : x{ _x }, y{ _y }, z{ _z } {}
	explicit cvec3(const vec3 &v) : x{ v.x }, y{ v.y }, z{ v.z } {}

	float & operator [] (int i) { return (&x)[i]; }
	float   operator [] (int i) const { return (&x)[i]; }
	// conversion
	explicit operator cpt3() const;
	explicit operator cnormal3() const;

	cvec3 & operator += (const cvec3 & rhs) {
		x += rhs.x; y += rhs.y; z += rhs.z;
		return *this;
	}
	cvec3 & operator -= (const cvec3 & rhs) {
		x -= rhs.x; y -= rhs.y; z -= rhs.z;
		return *this;

	}
	cvec3 & operator *= (float s) {
		x *= s; y *= s; z *= s;
		return *this;
	}
	cvec3 & operator /= (float s) {
		assert(s != 0.0f);
		float inverse = 1.0f / s;
		x *= inverse; y *= inverse; z *= inverse;
		return *this;
	}

	float length() const {
		return std::sqrt(length_squared());
	}
	float length_squared() const {
		return x*x + y*y + z*z;
	}
	void normalize() {
		float l = length();
		assert(l != 0.0f);
		if (l != 1.0f) {
			float inv = 1.0f / l;
			x *= inv; y *= inv; z *= inv;
		}
	}
	cvec3 normalized() const {
		float l = length();
		assert(l != 0.0f);
		if (l != 1.0f) {
			float inv = 1.0f / l;
			return cvec3{ x * inv, y * inv, z * inv };
		}
		else {
			return (*this);
		}
	}

	static cvec3 xbase() { return cvec3{1, 0, 0};}
	static cvec3 ybase() { return cvec3{0, 1, 0};}
	static cvec3 zbase() { return cvec3{0, 0, 1};}
};

class cpt3 {
public:
	float x, y, z;
	cpt3() :x{ 0.0f }, y{ 0.0f }, z{ 0.0f } {}
	cpt3(float _x, float _y, float _z) : x{ _x }, y{ _y }, z{ _z } {}
	explicit cpt3(const pt3 &p) : x{ p.x }, y{ p.y }, z{ p.z } {}

	float & operator [] (int i) { return (&x)[i]; }
	float   operator [] (int i) const { return (&x)[i]; }
	// conversion
	explicit operator cvec3() const {return cvec3{x, y, z};}

	cpt3 & operator += (const cvec3 & rhs) {
		x += rhs.x; y += rhs.y; z += rhs.z;
		return *this;
	}
	cpt3 & operator -= (const cvec3 & rhs) {
		x -= rhs.x; y -= rhs.y; z -= rhs.z;
		return *this;
	}

	// static methods
	static cpt3 origin() { return cpt3{0, 0, 0}; }
	static cpt3 all(float x) { return cpt3{x, x, x}; }
};


class cnormal3 {
public:
	float x, y, z;
	cnormal3() : x{ 0.0f }, y{ 0.0f }, z{ 1.0f } {}
	cnormal3(float x, float y, float z) :x{ x }, y{ y }, z{ z } {}

	float & operator [] (int i) { return (&x)[i]; };
	float   operator [] (int i) const { return (&x)[i]; };
	// conversion
	explicit operator cvec3 () const { return cvec3{x, y, z}; }

	// n +/+= n, n -/-= n, n */*= s, s*n, n / or /=s

	cnormal3 & operator += (const cnormal3 &rhs) {
		x += rhs.x; y += rhs.y; z += rhs.z;
		return *this;
	}
	cnormal3 & operator -= (const cnormal3 &rhs) {
		x -= rhs.x; y -= rhs.y; z -= rhs.z;
		return *this;
	}
	cnormal3 & operator *= (float s) {
		x *= s; y *= s; z *= s;
		return *this;
	}
	cnormal3 & operator /= (float s) {
		assert(s != 0.0f);
		float inv = 1.0f / s;
		x *= inv; y *= inv; z *= inv;
		return *this;
	}

	void     flip_direction() {
		x = -x; y = -y; z = -z;
	}

	// normalize, length, normalized
	float	 length() const {
		return std::sqrt(length_squared());
	}
	float	 length_squared() const {
		return x*x + y*y + z*z;
	}
	cnormal3 normalized() const {
		float l = this->length();
		assert(l != 0.0f);
		float inv = 1.0f / l;
		return cnormal3{ x*inv, y*inv, z*inv };
	}
	void	 normalize() {
		float l = this->length();
		assert(l != 0.0f);
		float inv = 1.0f / l;
		x *= inv; y *= inv; z *= inv;
	}
};

// solving forward declaration
inline cvec3::operator cpt3() const{
	return cpt3{x, y, z};
}
inline cvec3::operator cnormal3() const {
	return cnormal3{x, y, z};
}

// v+v, v-v, v*s, v/s
inline cvec3 operator + (const cvec3 & lhs, const cvec3 & rhs) {
	return cvec3{ lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z };
}
inline cvec3 operator - (const cvec3 & lhs, const cvec3 & rhs) {
	return cvec3{ lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z };
}
inline cvec3 operator * (const cvec3 & lhs, float s) {
	return cvec3{ lhs.x*s, lhs.y*s, lhs.z*s };
}
inline cvec3 operator * (float s, const cvec3 &v) { return v*s;}
inline cvec3 operator / (const cvec3 & lhs, float s) {
	assert(s != 0.0f);
	float inverse = 1.0f / s;
	return cvec3{ lhs.x*inverse, lhs.y*inverse, lhs.z*inverse };
}

// return vector in opposite direction
inline cvec3 operator - (const cvec3 &v) {
	return cvec3{ -v.x, -v.y, -v.z };
}

// point +/+= vec, point -/-= vec
inline cpt3 operator + (const cpt3 &p, const cvec3 & v) {
	return cpt3{ p.x + v.x, p.y + v.y, p.z + v.z };
}
inline cpt3 operator - (const cpt3 &p, const cvec3 & v) {
	return cpt3{ p.x - v.x, p.y - v.y, p.z - v.z };
}

// point - point
inline cvec3 operator - (const cpt3 &lhs, const cpt3 & rhs) {
	return cvec3{ lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z };
}

inline cnormal3 operator + (const cnormal3 &lhs, const cnormal3 &rhs) {
	return cnormal3{ lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z };
}
inline cnormal3 operator - (const cnormal3 &lhs, const cnormal3 &rhs) {
	return cnormal3{ lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z };
}
inline cnormal3 operator * (const cnormal3 &lhs, float s) {
	return cnormal3{ lhs.x*s, lhs.y*s, lhs.z*s };
}
inline cnormal3 operator * (float s, const cnormal3 &n) {return n*s;}
inline cnormal3 operator / (const cnormal3 &lhs, float s) {
	float inv = 1.0f / s;
	assert(s != 0.0f);
	return cnormal3{ lhs.x*inv, lhs.y*inv, lhs.z*inv };
}
inline cnormal3 operator -(const cnormal3 &n) {
	return cnormal3{ -n.x, -n.y, -n.z };
}

// Dot-product and cross-product
inline float dot(const cvec3 &lhs, const cvec3 & rhs) {
	return lhs.x*rhs.x + lhs.y*rhs.y + lhs.z*rhs.z;
}
inline cvec3 cross(const cvec3 &lhs, const cvec3 & rhs) {
	// | x | y | z |
	// | x'| y'| z'|
	// | i | j | k |
	return cvec3{ lhs.y*rhs.z - lhs.z*rhs.y, lhs.z*rhs.x - lhs.x*rhs.z, lhs.x*rhs.y - lhs.y*rhs.x };
}
inline float dot(const cvec3 &v, const cnormal3 &n) {
	return v.x*n.x + v.y*n.y + v.z*n.z;
}
inline float dot(const cnormal3 &n, const cvec3 &v) {
	return dot(v,n);
}

#ifdef GLWHEEL_MATH_GEOMETRY_USE_OPOV_DC
inline float operator * (const cvec3 &v, const cvec3 &u) {return dot(v, u);}
inline float operator * (const cvec3 &v, const cnormal3 &n) {return dot(v, n);}
inline float operator * (const cnormal3 &n, const cvec3 &v) {return dot(v, n);}
inline cvec3 operator ^ (const cvec3 &v, const cvec3 &u) {return cross(v, u);}
#endif

float dist(const cpt3& lhs, const cpt3 &rhs) {
	return (lhs - rhs).length();
}

} // namespace hcm
#pragma once
#include <cassert>
#include <cmath>

namespace hcm {

class CPoint3f;
class CNormal3f;

class CVector3f {
public:
	float x, y, z;

	// Constructors
	CVector3f() : x{ 0.0f }, y{ 0.0f }, z{ 0.0f } {}
	CVector3f(float _x, float _y, float _z) : x{ _x }, y{ _y }, z{ _z } {}
	explicit CVector3f(const CPoint3f &p);
	explicit CVector3f(const CNormal3f &n);
	explicit CVector3f(const Vector3f &v) : x{ v.x }, y{ v.y }, z{ v.z } {}


	float & operator [] (int i) { return (&x)[i]; }
	float   operator [] (int i) const { return (&x)[i]; }

	CVector3f operator + (const CVector3f & rhs) const { return CVector3f{ x + rhs.x, y + rhs.y, z + rhs.z }; }
	CVector3f operator - (const CVector3f & rhs) const { return CVector3f{ x - rhs.x, y - rhs.y, z - rhs.z }; }
	CVector3f operator * (float s) const {
		return CVector3f{ x*s, y*s, z*s };
	}
	CVector3f operator / (float s) const {
		assert(s != 0.0f);
		float inverse = 1.0f / s;
		return CVector3f{ x*inverse, y*inverse, z*inverse };
	}

	CVector3f & operator += (const CVector3f & rhs) {
		x += rhs.x; y += rhs.y; z += rhs.z;
		return *this;
	}
	CVector3f & operator -= (const CVector3f & rhs) {
		x -= rhs.x; y -= rhs.y; z -= rhs.z;
		return *this;

	}
	CVector3f & operator *= (float s) {
		x *= s; y *= s; z *= s;
		return *this;
	}
	CVector3f & operator /= (float s) {
		assert(s != 0.0f);
		float inverse = 1.0f / s;
		x *= inverse; y *= inverse; z *= inverse;
		return *this;
	}

	// Dot-product and cross-product
	float    operator * (const CVector3f & rhs) const {
		return x*rhs.x + y*rhs.y + z*rhs.z;
	}
	CVector3f operator ^ (const CVector3f & rhs) const {
		// | x | y | z |
		// | x'| y'| z'|
		// | i | j | k |
		return CVector3f{ y*rhs.z - z*rhs.y, z*rhs.x - x*rhs.z, x*rhs.y - y*rhs.x };
	}
	float    dot(const CVector3f & rhs)   const { return (*this)*rhs; }
	CVector3f cross(const CVector3f & rhs) const { return (*this) ^ rhs; }

	// return vector in opposite direction
	CVector3f operator - () const {
		return CVector3f{ -x, -y, -z };
	}

	float length() const {
		return std::sqrt(lengthSquared());
	}
	float lengthSquared() const {
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
	CVector3f normalized() const {
		float l = length();
		assert(l != 0.0f);
		if (l != 1.0f) {
			float inv = 1.0f / l;
			return CVector3f{ x * inv, y * inv, z * inv };
		}
		else {
			return (*this);
		}
	}

	static const CVector3f XBASE;
	static const CVector3f YBASE;
	static const CVector3f ZBASE;
	static const CVector3f &XDIR;
	static const CVector3f &YDIR;
	static const CVector3f &ZDIR;
};

class CPoint3f {
public:
	float x, y, z;
	CPoint3f() :x{ 0.0f }, y{ 0.0f }, z{ 0.0f } {}
	CPoint3f(float _x, float _y, float _z) : x{ _x }, y{ _y }, z{ _z } {}
	explicit CPoint3f(const CVector3f &v) :x{ v.x }, y{ v.y }, z{ v.z } {}
	explicit CPoint3f(const Point3f &p) : x{ p.x }, y{ p.y }, z{ p.z } {}

	float & operator [] (int i) { return (&x)[i]; }
	float   operator [] (int i) const { return (&x)[i]; }

	// point +/+= vec, point -/-= vec
	CPoint3f operator + (const CVector3f & rhs) const {
		return CPoint3f{ x + rhs.x, y + rhs.y, z + rhs.z };
	}
	CPoint3f operator - (const CVector3f & rhs) const {
		return CPoint3f{ x - rhs.x, y - rhs.y, z - rhs.z };

	}

	CPoint3f & operator += (const CVector3f & rhs) {
		x += rhs.x; y += rhs.y; z += rhs.z;
		return *this;
	}
	CPoint3f & operator -= (const CVector3f & rhs) {
		x -= rhs.x; y -= rhs.y; z -= rhs.z;
		return *this;
	}

	// point - point
	CVector3f operator - (const CPoint3f & rhs) const {
		return CVector3f{ x - rhs.x, y - rhs.y, z - rhs.z };
	}

	float dist(const CPoint3f &rhs) const {
		return ((*this) - rhs).length();
	}

	// static methods
	static CPoint3f origin() { return ORIGIN; }
	static CPoint3f all(float x) { return CPoint3f{x, x, x}; }
	static const CPoint3f ORIGIN;
};

inline CVector3f::CVector3f(const CPoint3f &p)
	: x{ p.x }, y{ p.y }, z{ p.z } {}

class CNormal3f {
public:
	float x, y, z;
	CNormal3f() : x{ 0.0f }, y{ 0.0f }, z{ 1.0f } {}
	CNormal3f(float x, float y, float z) :x{ x }, y{ y }, z{ z } {}
	explicit CNormal3f(const CVector3f &v) :x{ v.x }, y{ v.y }, z{ v.z } {}

	float & operator [] (int i) { return (&x)[i]; };
	float   operator [] (int i) const { return (&x)[i]; };

	// n +/+= n, n -/-= n, n */*= s, s*n, n / or /=s
	CNormal3f operator + (const CNormal3f &rhs) const {
		return CNormal3f{ x + rhs.x, y + rhs.y, z + rhs.z };
	}
	CNormal3f operator - (const CNormal3f &rhs) const {
		return CNormal3f{ x - rhs.x, y - rhs.y, z - rhs.z };
	}
	CNormal3f operator * (float s) const {
		return CNormal3f{ x*s, y*s, z*s };
	}
	CNormal3f operator / (float s) const {
		float inv = 1.0f / s;
		assert(s != 0.0f);
		return CNormal3f{ x*inv, y*inv, z*inv };
	}
	CNormal3f & operator += (const CNormal3f &rhs) {
		x += rhs.x; y += rhs.y; z += rhs.z;
		return *this;
	}
	CNormal3f & operator -= (const CNormal3f &rhs) {
		x -= rhs.x; y -= rhs.y; z -= rhs.z;
		return *this;
	}
	CNormal3f & operator *= (float s) {
		x *= s; y *= s; z *= s;
		return *this;
	}
	CNormal3f & operator /= (float s) {
		assert(s != 0.0f);
		float inv = 1.0f / s;
		x *= inv; y *= inv; z *= inv;
		return *this;
	}

	CNormal3f operator -() const {
		return CNormal3f{ -x, -y, -z };
	}
	void     flipDirection() {
		x = -x; y = -y; z = -z;
	}

	// normalize, length, normalized
	float	 length() const {
		return std::sqrt(lengthSquared());
	}
	float	 lengthSquared() const {
		return x*x + y*y + z*z;
	}
	CNormal3f normalized() const {
		float l = this->length();
		assert(l != 0.0f);
		float inv = 1.0f / l;
		return CNormal3f{ x*inv, y*inv, z*inv };
	}
	void	 normalize() {
		float l = this->length();
		assert(l != 0.0f);
		float inv = 1.0f / l;
		x *= inv; y *= inv; z *= inv;
	}

	// dot with normal, dot with vector
	float operator * (const CNormal3f &rhs) const {
		return x*rhs.x + y*rhs.y + z*rhs.z;
	}
	float operator * (const CVector3f &rhs) const {
		return x*rhs.x + y*rhs.y + z*rhs.z;
	}
};

inline CVector3f::CVector3f(const CNormal3f & n) : x(n.x), y(n.y), z(n.z) {}
} // namespace hcm
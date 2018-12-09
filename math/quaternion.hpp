#pragma once
#include "geometry.hpp"
#include "transform.hpp"
#include <xmmintrin.h>

namespace hcm {


class Quaternion
{
public:
	Quaternion() {
		simdData = _mm_set_ps(0.0f, 0.0f, 0.0f, 1.0f);
	}
	Quaternion(const Vector3f &v, float w_ = 0.0f)
		: x{v.x}, y{v.y}, z{v.z}, w{w_} {}
	Quaternion(const Point3f &p, float w_ = 1.0f)
		: x{p.x}, y{p.y}, z{p.z}, w{w_} {}

	static Quaternion rotater(const Vector3f &axis, float degree);

	// internal arithmetic operation
	Quaternion operator + (const Quaternion &rhs) const {
		Quaternion ret;
		ret.simdData = _mm_add_ps(this->simdData, rhs.simdData);
		return ret;
	}
	Quaternion operator - (const Quaternion &rhs) const {
		Quaternion ret;
		ret.simdData = _mm_sub_ps(this->simdData, rhs.simdData);
		return ret;
	}
	Quaternion operator * (const Quaternion &rhs) const {
		// cross product
		/* | y | z | x |   | z | x | y | this
		   | z | x | y | - | y | z | x | rhs
		*/
		// maybe _mm_shuffle_ps(simdData, simdData, ???)?
		__m128 thisLS = _mm_permute_ps(simdData, 0b11'00'10'01);
		__m128 thisRS = _mm_permute_ps(simdData, 0b11'01'00'10);
		__m128 rhsLS = _mm_permute_ps(rhs.simdData, 0b11'00'10'01);
		__m128 rhsRS = _mm_permute_ps(rhs.simdData, 0b11'01'00'10);
		__m128 subtractend = _mm_mul_ps(thisLS, rhsRS);
		__m128 subtractor = _mm_mul_ps(thisRS, rhsLS);
		Quaternion ret;
		ret.simdData = _mm_sub_ps(subtractend, subtractor);
		ret += (*this)*rhs.w;
		ret += rhs*w;
		__m128 dot = _mm_dp_ps(simdData, rhs.simdData, 0x7f);
		ret.w = this->w*rhs.w - _mm_cvtss_f32(dot);
		return ret;
	}
	Quaternion operator * (float s) const {
		__m128 multiplier = _mm_set_ps1(s);
		Quaternion ret;
		ret.simdData = _mm_mul_ps(simdData, multiplier);
		return ret;
	}
	Quaternion operator / (float s) const {
		assert(s != 0.0);
		__m128 inverse = _mm_set_ps1(1.0f/s);
		Quaternion ret;
		ret.simdData = _mm_mul_ps(simdData, inverse);
		return ret;
	}

	Quaternion & operator += (const Quaternion &rhs) {
		this->simdData = _mm_add_ps(this->simdData, rhs.simdData);
		return *this;
	}
	Quaternion & operator -= (const Quaternion &rhs) {
		this->simdData = _mm_sub_ps(this->simdData, rhs.simdData);
		return *this;
	}
	Quaternion & operator *= (const Quaternion &rhs) {
		// makeshift implementation
		(*this) = (*this)*rhs;
		return *this;
	}
	Quaternion & operator *= (float s) {
		__m128 multiplier = _mm_set_ps1(s);
		this->simdData = _mm_mul_ps(this->simdData, multiplier);
		return *this;
	}
	Quaternion & operator /= (float s) {
		assert(s != 0.0);
		__m128 inverse = _mm_set_ps1(1.0f / s);
		this->simdData = _mm_mul_ps(this->simdData, inverse);
		return *this;
	}

	// length, conjugate and reciprocal
	Quaternion conjugate() const {
		static const __m128i intNegator = _mm_set_epi32(0, 0x80000000, 0x80000000, 0x80000000);
		Quaternion ret;
		ret.simdData = _mm_castsi128_ps(
			_mm_xor_si128(intNegator, _mm_castps_si128(this->simdData))
		);
		return ret;
	}
	void normalize() {
		float l = this->length();
		if (l != 1.0f) {
			this->operator/=(l);
		}
	}
	Quaternion normalized() const {
		float l = this->length();
		if (l != 1.0f)
			return *this / l;
		else
			return *this;
	}
	float length() const {
		return std::sqrtf(lengthSquared());
	}
	float lengthSquared() const {
		__m128 dot = _mm_dp_ps(this->simdData, this->simdData, 0xFF);
		return _mm_cvtss_f32(dot);
	}
	Quaternion reciprocal() const {
		float l2 = lengthSquared();
		if (l2 == 1.0f) {
			return *this;
		}
		else {
			return *this / l2;
		}
	}

	// operation on other geometric objects
	Point3f rotate(const Point3f &p) {
		Quaternion q{p};
		Quaternion pPrime = (*this)*q*((*this).conjugate());
		return pPrime.toPoint3f();
	}
	
	Point3f toPoint3f() const {
		return Point3f{x, y, z};
	}
	Vector3f toVector3f() const {
		return Vector3f{x, y, z};
	}

	// put these two in the .cpp file since they won't be
	//   called too many times... especially fromRotMatrix.
	void fromRotMatrix(const Matrix3f &m);
	Matrix3f toRotMatrix() const;


	//Vector3f xyz;
	//float w;

	// memory layout:
	/*
	i[3]  i[2]  i[1]  i[0]
	+-----+-----+-----+-----+
	|  w  |  z  |  y  |  x  |
	+-----+-----+-----+-----+
	*/
	union {
		__m128 simdData;
		struct {
			float x, y, z, w;
		};
	};
};

inline Quaternion operator*(float s, const Quaternion &q) { return q * s; }

inline float dot(const Quaternion &q0, const Quaternion &q1) {
	__m128 res = _mm_dp_ps(q0.simdData, q1.simdData, 0xFF);
	return _mm_cvtss_f32(res);
}

inline Quaternion slerp(float t, const Quaternion &q0, const Quaternion &q1);

} // namespace hcm

std::ostream & operator << (std::ostream & os, const hcm::Quaternion &q);
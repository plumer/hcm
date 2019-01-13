#pragma once
#include "geometry.hpp"
#include "transform.hpp"
#include <xmmintrin.h>

namespace hcm {

class quat
{
public:
	quat() {
		simdData = _mm_set_ps(0.0f, 0.0f, 0.0f, 1.0f);
	}
	quat(const vec3 &v, float w_ = 0.0f)
		: x{v.x}, y{v.y}, z{v.z}, w{w_} {}
	quat(const pt3 &p, float w_ = 1.0f)
		: x{p.x}, y{p.y}, z{p.z}, w{w_} {}

	static quat rotater(const vec3 &axis, float degree);

	// >>>>>>>>>>>>>>>> overloaded operators >>>>>>>>>>>>>>>>

	// internal arithmetic operation
	quat operator + (const quat &rhs) const {
		quat ret;
		ret.simdData = _mm_add_ps(this->simdData, rhs.simdData);
		return ret;
	}
	quat operator - (const quat &rhs) const {
		quat ret;
		ret.simdData = _mm_sub_ps(this->simdData, rhs.simdData);
		return ret;
	}
	quat operator * (const quat &rhs) const {
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
		quat ret;
		ret.simdData = _mm_sub_ps(subtractend, subtractor);
		ret += (*this)*rhs.w;
		ret += rhs*w;
		__m128 dot = _mm_dp_ps(simdData, rhs.simdData, 0x7f);
		ret.w = this->w*rhs.w - _mm_cvtss_f32(dot);
		return ret;
	}
	quat operator * (float s) const {
		__m128 multiplier = _mm_set_ps1(s);
		quat ret;
		ret.simdData = _mm_mul_ps(simdData, multiplier);
		return ret;
	}
	quat operator / (float s) const {
		assert(s != 0.0);
		__m128 inverse = _mm_set_ps1(1.0f/s);
		quat ret;
		ret.simdData = _mm_mul_ps(simdData, inverse);
		return ret;
	}

	quat operator - () const {
		static const __m128i int_negator = _mm_set1_epi32(0x80000000);
		quat ret;
		ret.simdData = _mm_castsi128_ps(
			_mm_xor_si128(int_negator, _mm_castps_si128(this->simdData))
		);
		return ret;
	}

	// quaternion operations are pretty self-contained
	// so they are not listed as global functions
	quat & operator += (const quat &rhs) {
		this->simdData = _mm_add_ps(this->simdData, rhs.simdData);
		return *this;
	}
	quat & operator -= (const quat &rhs) {
		this->simdData = _mm_sub_ps(this->simdData, rhs.simdData);
		return *this;
	}
	quat & operator *= (const quat &rhs) {
		// makeshift implementation
		(*this) = (*this)*rhs;
		return *this;
	}
	quat & operator *= (float s) {
		__m128 multiplier = _mm_set_ps1(s);
		this->simdData = _mm_mul_ps(this->simdData, multiplier);
		return *this;
	}
	quat & operator /= (float s) {
		assert(s != 0.0);
		__m128 inverse = _mm_set_ps1(1.0f / s);
		this->simdData = _mm_mul_ps(this->simdData, inverse);
		return *this;
	}

	// <<<<<<<<<<<<<<< overloaded operators <<<<<<<<<<<<<<<<

	// length, conjugate and reciprocal
	quat conjugate() const {
		static const __m128i intNegator = _mm_set_epi32(0, 0x80000000, 0x80000000, 0x80000000);
		quat ret;
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
	quat normalized() const {
		float l = this->length();
		if (l != 1.0f)
			return *this / l;
		else
			return *this;
	}
	float length() const {
		return std::sqrtf(length_squared());
	}
	float length_squared() const {
		__m128 dot = _mm_dp_ps(this->simdData, this->simdData, 0xFF);
		return _mm_cvtss_f32(dot);
	}
	quat reciprocal() const {
		float l2 = length_squared();
		if (l2 == 1.0f) {
			return conjugate();
		}
		else {
			return conjugate() / l2;
		}
	}

	// operation on other geometric objects
	pt3 rotate(const pt3 &p) {
		quat q{p};
		quat pPrime = (*this)*q*((*this).conjugate());
		return pt3{pPrime};
	}
	
	explicit operator pt3() const {
		return pt3{x, y, z};
	}
	explicit operator vec3() const {
		return vec3{x, y, z};
	}


	bool from_mat3(const mat3 &m);
	mat3 to_mat3() const;


	//vec3 xyz;
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

inline quat operator*(float s, const quat &q) { return q * s; }

inline float dot(const quat &q0, const quat &q1) {
	__m128 res = _mm_dp_ps(q0.simdData, q1.simdData, 0xFF);
	return _mm_cvtss_f32(res);
}

inline quat quat::rotater(const vec3 &axis, float degree) {
	vec3 a = axis.normalized();
	float sinT = std::sin(TO_RADIAN(degree/2));
	float cosT = std::cos(TO_RADIAN(degree/2));
	a *= sinT;
	return quat{ a, cosT };
}

inline mat3 quat::to_mat3() const
{
	// TODO: what if the quaternion is not unit-length?
	// float x = xyz.x, y = xyz.y, z = xyz.z;
	vec3 col1{ 1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w) };
	vec3 col2{ 2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w) };
	vec3 col3{ 2 * (z*x - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y) };
	return mat3{ col1, col2, col3 };
}

inline bool quat::from_mat3(const mat3 &m) {
	mat3 id = m*(m.transposed());
	mat3 diff = id - mat3();
	if (diff.frobenius_norm() < 0.001) {
		// trace(m) = 3-2(y2 + z2 + x2 + z2 + x2 + y2) = 3-4(x^2+y^2+z^2);
		// xyz*xyz = 0.25*(3-trace)
		const vec3 &c0 = m.col(0), &c1 = m.col(1), &c2 = m.col(2);
		float vnorm = 0.5f*std::sqrt(3 - c0[0] - c1[1] - c2[2]);
		float zw = 0.25f*(c1[0] - c0[1]);
		float yw = 0.25f*(c0[2] - c2[0]);
		float xw = 0.25f*(c2[1] - c1[2]);
		vec3 xyz_{ xw, yw, zw };
		// {xw, yw, zw} = {x, y, z} * w.
		float w_ = xyz_.length() / vnorm;
		xyz_ /= w_;
		this->simdData = _mm_set_ps(w_, xyz_.z, xyz_.y, xyz_.x);
		this->normalize();
	}
	else
		return;
		//::std::cout << "not a unitary matrix\n";
}

inline quat slerp(float t, const quat & q0, const quat & q1)
{
	float cosT = dot(q0, q1)/q1.length_squared();
	if (cosT > 0.9995f) {
		// return a normalized linear combination instead.
		quat res = q0 * (1 - t) + q1 * t;
		res.normalize();
		return res;
	}
	else {
		quat qPerp = q1 - cosT * q0;
		qPerp.normalize();
		float theta = std::acos(cosT);
		return q0 * std::cos(theta*t) + qPerp * std::sin(theta*t);
	}
}

} // namespace hcm
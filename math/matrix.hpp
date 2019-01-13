#pragma once
#include "geometry.hpp"
#include "misc.hpp"
#include <xmmintrin.h>
#include <cstring>

#define COORDINATE_SYSTEM_LEFT_HANDED 0x233
#define COORDINATE_SYSTEM_RIGHT_HANDED 0x666

#ifndef COORDINATE_SYSTEM_HANDEDNESS
#define COORDINATE_SYSTEM_HANDEDNESS COORDINATE_SYSTEM_LEFT_HANDED 
#endif

namespace hcm {

class mat3 {
public:
	vec3 c0, c1, c2;

	mat3() {
		c0.x = c1.y = c2.z = 1.0f;
	}
	mat3(const vec3 &c0, const vec3 &c1, const vec3 &c2)
	: c0{c0}, c1{c1}, c2{c2} { }

	// I don't think M+N, M-N will be used a lot...
	mat3 operator + (const mat3 & rhs) const {
		return mat3{
			c0 + rhs.c0,
			c1 + rhs.c1,
			c2 + rhs.c2
		};
	}

	mat3 operator - (const mat3 & rhs) const {
		return mat3{
			c0 - rhs.c0,
			c1 - rhs.c1,
			c2 - rhs.c2
		};
	}

	mat3 operator * (const mat3 & rhs) const {
		return mat3{
			(*this)*rhs.c0,
			(*this)*rhs.c1,
			(*this)*rhs.c2
		};
	}

	// multiplication is in the order of (*this) * (rhs).

	vec3 operator * (const vec3 & v) const {
		return c0 * v.x + c1 * v.y + c2 * v.z;
	}

	mat3 & operator += (const mat3 & rhs) {
		c0 += rhs.c0;
		c1 += rhs.c1;
		c2 += rhs.c2;
		return *this;
	}

	mat3 & operator -= (const mat3 & rhs) {
		c0 -= rhs.c0;
		c1 -= rhs.c1;
		c2 -= rhs.c2;
		return *this;
	}

	mat3 & operator *= (const mat3 & rhs) {
		mat3 temp = (*this) * rhs;
		(*this) = temp;
		return *this;
	}

	// one can only get a copy of a row of matrix
	vec3   row(int x) const {
		return vec3{ c0[x], c1[x], c2[x] };
	}

	// however a reference to a column is accessible.
	vec3 & col(int x) { return (&c0)[x]; }
	const vec3 & col(int x) const { return (&c0)[x]; }

	// transpose, inverse and determinant
	void transpose() {
#define SWAP_CODE9527(a, b) \
	do {decltype(a) t = b; b = a; a = t;} while (0)
		SWAP_CODE9527(c0.y, c1.x);
		SWAP_CODE9527(c0.z, c2.x);
		SWAP_CODE9527(c1.z, c2.y);
#undef SWAP_CODE9527
	}

	mat3 transposed() const {
		return mat3{
			vec3{ c0.x, c1.x, c2.x },
			vec3{ c0.y, c1.y, c2.y },
			vec3{ c0.z, c1.z, c2.z }
		};
	}
	void invert() {
		// I copied this code from Dr. Popescu
		const vec3 a = row(0), b = row(1), c = row(2);
		vec3 _a = b ^ c; _a /= (a * _a);
		vec3 _b = c ^ a; _b /= (b * _b);
		vec3 _c = a ^ b; _c /= (c * _c);
		c0 = _a; c1 = _b; c2 = _c;
	}

	mat3 inverse() const {
		const vec3 a = row(0), b = row(1), c = row(2);
		vec3 _a = b ^ c; _a /= (a * _a);
		vec3 _b = c ^ a; _b /= (b * _b);
		vec3 _c = a ^ b; _c /= (c * _c);

		return mat3{ _a, _b, _c };
	}
	float det() const {
		return (c0 ^ c1)*c2;
	}

	float frobenius_norm() const {
		return std::sqrt(
			c0.length_squared() + c1.length_squared() + c2.length_squared());
	}


	static mat3 rotateX(float degree) {
		float radian = TO_RADIAN(degree);
		float cosAngle = std::cosf(radian);
		float sinAngle = std::sinf(radian);

		// c & -s \\ s & c
		return mat3{
			vec3::xbase(),
			vec3{ 0, cosAngle, sinAngle },
			vec3{ 0, -sinAngle, cosAngle }
		};
	}
	static mat3 rotateY(float degree) {
		float radian = TO_RADIAN(degree);
		float cosAngle = std::cosf(radian);
		float sinAngle = std::sinf(radian);

		// c & -s \\ s & c
		return mat3{
			vec3{ cosAngle, 0.0, -sinAngle },
			vec3::ybase(),
			vec3{ sinAngle, 0.0, cosAngle }
		};
	}
	static mat3 rotateZ(float degree) {
		float radian = TO_RADIAN(degree);
		float cosAngle = std::cosf(radian);
		float sinAngle = std::sinf(radian);

		// c & -s \\ s & c
		return mat3{
			vec3{ cosAngle, sinAngle, 0 },
			vec3{ -sinAngle, cosAngle, 0 },
			vec3::zbase()
		};
	}

	static mat3 rotate(const vec3 & axis, float degree) {
		vec3 a = axis.normalized();
		float sinT = std::sin(TO_RADIAN(degree));
		float cosT = std::cos(TO_RADIAN(degree));
		return mat3{
			vec3{ a.x*a.x + cosT * (1 - a.x*a.x), a.x*a.y*(1 - cosT) - sinT * a.z, a.x*a.z*(1 - cosT) + sinT * a.y },
			vec3{ a.x*a.y*(1 - cosT) + sinT * a.z, a.y*a.y + cosT * (1 - a.y*a.y), a.y*a.z*(1 - cosT) - sinT * a.x },
			vec3{ a.x*a.z*(1 - cosT) - sinT * a.y, a.y*a.z*(1 - cosT) + sinT * a.x, a.z*a.z + cosT * (1 - a.z*a.z) }
		};
	}
};

class mat4 {
public:
	vec4 c0, c1, c2, c3;
	mat4() :
		c0{ vec4::xbase() }, c1{ vec4::ybase() },
		c2{ vec4::zbase() }, c3{ vec4::wbase() } {}
	mat4(const vec4 &a, const vec4 &b, const vec4 &c, const vec4 &d) :
		c0{ a }, c1{ b }, c2{ c }, c3{ d } {
            
        }
	mat4(const vec4 _c[4]) :
		c0{ _c[0] }, c1{ _c[1] }, c2{ _c[2] }, c3{ _c[3] } {}
	explicit mat4(const mat3 &m) :
		c0{ m.c0 }, c1{ m.c1 }, c2{ m.c2 },
		c3{ vec4::wbase() } {}
	mat4(const float m[16], bool t = false) {
		memcpy(&c0, m, sizeof(float) * 16);
		if (t) transpose();
	}

	vec4 & operator[](int i) { return (&c0)[i]; }
	const vec4 & operator[](int i) const { return (&c0)[i]; }

	mat4(const mat4 &m) = default;
	mat4& operator=(const mat4 &m) = default;

	mat4 & operator+=(const mat4 &rhs) {
		c0 += rhs.c0; c1 += rhs.c1; c2 += rhs.c2; c3 += rhs.c3;
		return *this;
	}
	mat4 & operator -= (const mat4 &rhs) {
		c0 -= rhs.c0; c1 -= rhs.c1; c2 -= rhs.c2; c3 -= rhs.c3;
		return *this;
	}
	mat4 & operator *= (float s) {
		c0 *= s; c1 *= s; c2 *= s; c3 *= s;
		return *this;
	}
	mat4 & operator /= (float s) {
		float inv = 1.0f / s;
		c0 *= inv; c1 *= inv; c2 *= inv; c3 *= inv;
		return *this;
	}

	mat4 operator + (const mat4 &rhs) const { return mat4{ *this } += rhs; }
	mat4 operator - (const mat4 &rhs) const { return mat4{ *this } -= rhs; }
	mat4 operator * (float s) const { return mat4{ *this } *= s; }
	mat4 operator / (float s) const { return mat4{ *this } /= s; }

	vec4 operator * (const vec4 &v) const {
		//return c0*v.x + c1*v.y + c2*v.z + c3*v.w;
		__m128 i0 = _mm_mul_ps(_mm_set_ps1(v.x), c0.simdData);
		__m128 i1 = _mm_mul_ps(_mm_set_ps1(v.y), c1.simdData);
		__m128 i2 = _mm_mul_ps(_mm_set_ps1(v.z), c2.simdData);
		__m128 i3 = _mm_mul_ps(_mm_set_ps1(v.w), c3.simdData);

		__m128 i01 = _mm_add_ps(i0, i1);
		__m128 i23 = _mm_add_ps(i2, i3);
		return vec4{ _mm_add_ps(i01, i23) };

	}
	mat4 operator * (const mat4 &rhs) const {
		mat4 res;
		res.c0 = this->operator*(rhs.c0);
		res.c1 = this->operator*(rhs.c1);
		res.c2 = this->operator*(rhs.c2);
		res.c3 = this->operator*(rhs.c3);
		return res;
	}
	/*
	float det() const {
	__m128 x = c0.simdData, y = c1.simdData, z = c2.simdData, w = c3.simdData;
	if (w.w == 0.0f && z.w != 0.0f) {
	std::swap(z, w);
	}
	// z = z - w * z[3]/w[3].
	__m128 zOverW = _mm_set_ps1(z.w/w.w);
	z = _mm_sub_ps(z, _mm_mul_ps(zOverW), w);
	__m128 yOverW = _mm_set_ps1(y.w/w.w);
	y = _mm_sub_ps(y, _mm_mul_ps(yOverW), w);
	__m128 xOverW = _mm_set_ps1(x.w/w.w);
	x = _mm_sub_ps(x, _mm_mul_ps(xOverW), w);

	// calculate x^y.z.
	return (vec3{x}.cross(vec3{y}).dot(vec3{z});
	}
	*/
	float frobenius_norm_squared() const {
		return c0.length_squared() + c1.length_squared() + 
			c2.length_squared() + c3.length_squared();
	}
	float frobenius_norm() const {
		return std::sqrt(frobenius_norm_squared());
	}

	void transpose() {
		_MM_TRANSPOSE4_PS(c0.simdData, c1.simdData, c2.simdData, c3.simdData);
	}

	void invert() {
		// The following code is copied from Intel. see
		// 'Streaming SIMD Extensions - Inverse of 4x4 Matrix'

		// 1. Variables which will contain cofactors and, later, the lines of the 
		//    inverted matrix are declared.
		__m128 minor0, minor1, minor2, minor3;

		// 2. Variables which will contain the lines of the reference matrix and,
		//    later (after the transposition), the columns of the original matrix are declared.
		__m128 row0, row1, row2, row3;

		// 3. Temporary variables and the variable that will contain the matrix determinant.
		__m128 det, tmp1;

		// 4. Matrix transposition. row{0,1,2,3} are the rows of the matrix.
		tmp1 = _mm_loadh_pi(_mm_loadl_pi(tmp1, (__m64*)(&c0)), (__m64*)(&c1));
		row1 = _mm_loadh_pi(_mm_loadl_pi(row1, (__m64*)(&c2)), (__m64*)(&c3));

		row0 = _mm_shuffle_ps(tmp1, row1, 0x88);
		row1 = _mm_shuffle_ps(row1, tmp1, 0xDD);

		tmp1 = _mm_loadh_pi(_mm_loadl_pi(tmp1, (__m64*)(&c0[2])), (__m64*)(&c1[2]));
		row3 = _mm_loadh_pi(_mm_loadl_pi(row3, (__m64*)(&c2[2])), (__m64*)(&c3[2]));

		row2 = _mm_shuffle_ps(tmp1, row3, 0x88);
		row3 = _mm_shuffle_ps(row3, tmp1, 0xDD);

		// 5. Cofactors calculation.
		tmp1 = _mm_mul_ps(row2, row3);
		tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);

		minor0 = _mm_mul_ps(row1, tmp1);
		minor1 = _mm_mul_ps(row0, tmp1);

		tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);

		minor0 = _mm_sub_ps(_mm_mul_ps(row1, tmp1), minor0);
		minor1 = _mm_sub_ps(_mm_mul_ps(row0, tmp1), minor1);
		minor1 = _mm_shuffle_ps(minor1, minor1, 0x4E);

		// ------------------------
		tmp1 = _mm_mul_ps(row1, row2);
		tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);

		minor0 = _mm_add_ps(_mm_mul_ps(row3, tmp1), minor0);
		minor3 = _mm_mul_ps(row0, tmp1);

		tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);

		minor0 = _mm_sub_ps(minor0, _mm_mul_ps(row3, tmp1));
		minor3 = _mm_sub_ps(_mm_mul_ps(row0, tmp1), minor3);
		minor3 = _mm_shuffle_ps(minor3, minor3, 0x4E);

		// -----------------------

		tmp1 = _mm_mul_ps(_mm_shuffle_ps(row1, row1, 0x4E), row3);
		tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
		row2 = _mm_shuffle_ps(row2, row2, 0x4E);

		minor0 = _mm_add_ps(_mm_mul_ps(row2, tmp1), minor0);
		minor2 = _mm_mul_ps(row0, tmp1);

		tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);

		minor0 = _mm_sub_ps(minor0, _mm_mul_ps(row2, tmp1));
		minor2 = _mm_sub_ps(_mm_mul_ps(row0, tmp1), minor2);
		minor2 = _mm_shuffle_ps(minor2, minor2, 0x4E);
		// -----------------------

		tmp1 = _mm_mul_ps(row0, row1);
		tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);

		minor2 = _mm_add_ps(_mm_mul_ps(row3, tmp1), minor2);
		minor3 = _mm_sub_ps(_mm_mul_ps(row2, tmp1), minor3);

		tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);

		minor2 = _mm_sub_ps(_mm_mul_ps(row3, tmp1), minor2);
		minor3 = _mm_sub_ps(minor3, _mm_mul_ps(row2, tmp1));

		// ------------------------

		tmp1 = _mm_mul_ps(row0, row3);
		tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);

		minor1 = _mm_sub_ps(minor1, _mm_mul_ps(row2, tmp1));
		minor2 = _mm_add_ps(_mm_mul_ps(row1, tmp1), minor2);

		tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);

		minor1 = _mm_add_ps(_mm_mul_ps(row2, tmp1), minor1);
		minor2 = _mm_sub_ps(minor2, _mm_mul_ps(row1, tmp1));

		// ------------------------

		tmp1 = _mm_mul_ps(row0, row2);
		tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);

		minor1 = _mm_add_ps(_mm_mul_ps(row3, tmp1), minor1);
		minor3 = _mm_sub_ps(minor3, _mm_mul_ps(row1, tmp1));

		tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);

		minor1 = _mm_sub_ps(minor1, _mm_mul_ps(row3, tmp1));
		minor3 = _mm_add_ps(_mm_mul_ps(row1, tmp1), minor3);

		// 6. Evaluation of determinant and its reciprocal value.

		det = _mm_mul_ps(row0, minor0);
		det = _mm_add_ps(_mm_shuffle_ps(det, det, 0x4E), det);
		det = _mm_add_ss(_mm_shuffle_ps(det, det, 0xB1), det);
		tmp1 = _mm_rcp_ss(det);

		det = _mm_sub_ss(_mm_add_ss(tmp1, tmp1), _mm_mul_ss(det, _mm_mul_ss(tmp1, tmp1)));
		det = _mm_shuffle_ps(det, det, 0x00);

		// 7. Multiplication of cofactors by 1/det.

		minor0 = _mm_mul_ps(det, minor0);
		_mm_storel_pi((__m64*)(&c0[0]), minor0);
		_mm_storeh_pi((__m64*)(&c0[2]), minor0);

		minor1 = _mm_mul_ps(det, minor1);
		_mm_storel_pi((__m64*)(&c1[0]), minor1);
		_mm_storeh_pi((__m64*)(&c1[2]), minor1);

		minor2 = _mm_mul_ps(det, minor2);
		_mm_storel_pi((__m64*)(&c2[0]), minor2);
		_mm_storeh_pi((__m64*)(&c2[2]), minor2);

		minor3 = _mm_mul_ps(det, minor3);
		_mm_storel_pi((__m64*)(&c3[0]), minor3);
		_mm_storeh_pi((__m64*)(&c3[2]), minor3);

	}

	float is_identity() const {
		return frobenius_norm_squared() < 1e-6;
	}
	float is_identity_precise() const {
		// TODO: use _mm_test_all_zeros() (SSE4.1, smmintrin.h)
		// _mm_testc_ps is also good (avx)
		// do not use XOR since [0.f == -0.f] but [xor(0.f, -0.f) != 0]
		return frobenius_norm_squared() == 0;
	}

	// >>>>>>>>>>>>>>>>>> static builders >>>>>>>>>>>>>>>>

	static mat4 perspective(float fovy_deg, float aspect, float _near, float _far) {
		const float fovy_rad = TO_RADIAN(fovy_deg);
		const float tanHalfFovy = std::tanf(fovy_rad / 2.0f);
		mat4 res = mat4::zero();
		/*
		T const tanHalfFovy = tan(fovy / static_cast<T>(2));

		tmat4x4<T, defaultp> Result(static_cast<T>(0));
		Result[0][0] = static_cast<T>(1) / (aspect * tanHalfFovy);
		Result[1][1] = static_cast<T>(1) / (tanHalfFovy);
		Result[2][3] = - static_cast<T>(1);

		#		if GLM_DEPTH_CLIP_SPACE == GLM_DEPTH_ZERO_TO_ONE
		Result[2][2] = zFar / (zNear - zFar);
		Result[3][2] = -(zFar * zNear) / (zFar - zNear);
		#		else
		Result[2][2] = - (zFar + zNear) / (zFar - zNear);
		Result[3][2] = - (static_cast<T>(2) * zFar * zNear) / (zFar - zNear);
		#		endif
		*/
		res.c0.x = 1.0f / (aspect * tanHalfFovy);
		res.c1.y = 1.0f / tanHalfFovy;
#if COORDINATE_SYSTEM_HANDEDNESS == COORDINATE_SYSTEM_LEFT_HANDED
		res.c2.w = 1.0f;
#else
		res.c2.w = -1.0f;
#endif
		// use clip space [-1 1].
		res.c2.z = -(_far + _near) / (_far - _near);
		res.c3.z = -(_far * _near) * 2.0f / (_far - _near);
		return res;
	}
	static mat4 translate(const vec3 &t) {
		mat4 res;
		res.c3 = vec4{ t, 1.0f };
		return res;
	}

	static mat4 diag(const vec3 &diag) {
		mat4 res;
		res.c0.x = diag.x;
		res.c1.y = diag.y;
		res.c2.z = diag.z;
		return res;
	}

	static mat4 diag(float scale) {
		return mat4::diag(vec3{ scale, scale, scale });
	}

	// Conversion to float *
	explicit operator float *() { return &(c0.x); }
	explicit operator const float *() const { return &(c0.x); }

	static mat4 rotateX(float degree) {
		float radian = TO_RADIAN(degree);
		float cosAngle = std::cosf(radian);
		float sinAngle = std::sinf(radian);

		return mat4{
			vec4::xbase(),
			vec4{ 0, cosAngle, sinAngle, 0.0f },
			vec4{ 0.0f, -sinAngle, cosAngle, 0.0f },
			vec4::wbase()
		};
	}
	static mat4 rotateY(float degree) {
		float radian = TO_RADIAN(degree);
		float cosAngle = std::cosf(radian);
		float sinAngle = std::sinf(radian);

		mat4 ret;
		ret.c0.simdData = _mm_set_ps(0.0f, -sinAngle, 0.0f, cosAngle);
		ret.c2.simdData = _mm_set_ps(0.0f, cosAngle, 0.0f, sinAngle);
		return ret;
	}
	static mat4 rotateZ(float degree) {
		float radian = TO_RADIAN(degree);
		float cosAngle = std::cosf(radian);
		float sinAngle = std::sinf(radian);

		mat4 ret;
		ret.c0.simdData = _mm_set_ps(0.0f, 0.0f, sinAngle, cosAngle);
		ret.c1.simdData = _mm_set_ps(0.0f, 0.0f, cosAngle, -sinAngle);
		return ret;
	}
	static mat4 rotate(const vec3 & axis, float degree) {
		mat3 m3 = mat3::rotate(axis, degree);
		mat4 ret;
		ret.c0 = vec4{ m3.c0 };
		ret.c1 = vec4{ m3.c1 };
		ret.c2 = vec4{ m3.c2 };
		return ret;
	}

	static mat4 lookAt(const pt3 & origin, const pt3 & target, const vec3 & up) {
		return mat4::lookAt(origin, target - origin, up);
	}
	static mat4 lookAt(const pt3 & origin, const vec3 & viewDir, const vec3 & up) {
#		if COORDINATE_SYSTEM_HANDEDNESS == COORDINATE_SYSTEM_RIGHT_HANDED
		// viewDir = -z, up = y, right = x
		vec3 z = -viewDir.normalized();
		vec3 x = up ^ z;
		x.normalize();
		vec3 y = z ^ x;
		y.normalize(); // may be unnecessary.
#		else
		// if left-handed system
		// viewDir = z, up = y, right = x
		vec3 z = viewDir.normalized();
		vec3 x = up ^ z;
		x.normalize();
		vec3 y = z ^ x;
		y.normalize();
#		endif
		// Let M = Matrix{x, y, z}.
		// then p'=M(p) transforms p from object space to camera's space.
		// What _lookAt_ transform is actually an INVERSE of such transform.
		// In homogeneous coordinate,
		// Inv(M) = | R T |  where R = | x y z |
		// 			| 0 1 |  so M(p) = R(p)+T = T(R(p)).
		// So M =   | R' -R'T |
		// 			| 0    1  | 
		mat4 ret{
			vec4{ x }, vec4{ y }, vec4{ z }, vec4::wbase()
		};
		ret.transpose();
		ret.c3 = -(ret * vec4{ origin, -1.0f }); // -1.0f is used here so that it will eventually become 1.0f with the leading '-' operator.
		return ret;
	}
	static mat4 zero() {mat4 res; memset(&res, sizeof(res), 0); return res;}
};

} // namespace hcm

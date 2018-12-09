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

class Matrix3f {
public:
	Vector3f columns[3];

	Matrix3f() {
		columns[0].x = columns[1].y = columns[2].z = 1.0f;
	}
	Matrix3f(const Vector3f &c0, const Vector3f &c1, const Vector3f &c2) {
		columns[0] = c0;
		columns[1] = c1;
		columns[2] = c2;
	}

	// I don't think M+N, M-N will be used a lot...
	Matrix3f operator + (const Matrix3f & rhs) const {
		return Matrix3f{
			columns[0] + rhs.columns[0],
			columns[1] + rhs.columns[1],
			columns[2] + rhs.columns[2]
		};
	}

	Matrix3f operator - (const Matrix3f & rhs) const {
		return Matrix3f{
			columns[0] - rhs.columns[0],
			columns[1] - rhs.columns[1],
			columns[2] - rhs.columns[2]
		};
	}

	Matrix3f operator * (const Matrix3f & rhs) const {
		return Matrix3f{
			(*this)*rhs.columns[0],
			(*this)*rhs.columns[1],
			(*this)*rhs.columns[2]
		};
	}

	// multiplication is in the order of (*this) * (rhs).

	Vector3f operator * (const Vector3f & v) const {
		return columns[0] * v.x + columns[1] * v.y + columns[2] * v.z;
	}

	Matrix3f & operator += (const Matrix3f & rhs) {
		columns[0] += rhs.columns[0];
		columns[1] += rhs.columns[1];
		columns[2] += rhs.columns[2];
		return *this;
	}

	Matrix3f & operator -= (const Matrix3f & rhs) {
		columns[0] -= rhs.columns[0];
		columns[1] -= rhs.columns[1];
		columns[2] -= rhs.columns[2];
		return *this;
	}

	Matrix3f & operator *= (const Matrix3f & rhs) {
		Matrix3f temp = (*this) * rhs;
		(*this) = temp;
		return *this;
	}


	// one can only get a copy of a row of matrix
	Vector3f   row(int x) const {
		return Vector3f{ columns[0][x], columns[1][x], columns[2][x] };
	}

	// however a reference to a column is accessible.
	Vector3f & col(int x) { return columns[x]; }
	const Vector3f & col(int x) const { return columns[x]; }

	// transpose, inverse and determinant
	void transpose() {
#define SWAP_CODE9527(a, b) \
	do {decltype(a) t = b; b = a; a = t;} while (0)
		SWAP_CODE9527(columns[0].y, columns[1].x);
		SWAP_CODE9527(columns[0].z, columns[2].x);
		SWAP_CODE9527(columns[1].z, columns[2].y);
#undef SWAP_CODE9527
	}

	Matrix3f transposed() const {
		return Matrix3f{
			Vector3f{ columns[0].x, columns[1].x, columns[2].x },
			Vector3f{ columns[0].y, columns[1].y, columns[2].y },
			Vector3f{ columns[0].z, columns[1].z, columns[2].z }
		};
	}
	void invert() {
		// I copied this code from Dr. Popescu
		const Vector3f a = row(0), b = row(1), c = row(2);
		Vector3f _a = b ^ c; _a /= (a * _a);
		Vector3f _b = c ^ a; _b /= (b * _b);
		Vector3f _c = a ^ b; _c /= (c * _c);
		columns[0] = _a; columns[1] = _b; columns[2] = _c;
	}

	Matrix3f inverse() const {
		const Vector3f a = row(0), b = row(1), c = row(2);
		Vector3f _a = b ^ c; _a /= (a * _a);
		Vector3f _b = c ^ a; _b /= (b * _b);
		Vector3f _c = a ^ b; _c /= (c * _c);

		return Matrix3f{ _a, _b, _c };
	}
	float det() const {
		return (columns[0] ^ columns[1])*columns[2];
	}


	float frobeniusNorm() const {
		return std::sqrt(
			columns[0].lengthSquared() + columns[1].lengthSquared() + columns[2].lengthSquared());
	}


	static Matrix3f rotateX(float degree) {
		float radian = TO_RADIAN(degree);
		float cosAngle = std::cosf(radian);
		float sinAngle = std::sinf(radian);

		// c & -s \\ s & c
		return Matrix3f{
			Vector3f::XBASE,
			Vector3f{ 0, cosAngle, sinAngle },
			Vector3f{ 0, -sinAngle, cosAngle }
		};
	}
	static Matrix3f rotateY(float degree) {
		float radian = TO_RADIAN(degree);
		float cosAngle = std::cosf(radian);
		float sinAngle = std::sinf(radian);

		// c & -s \\ s & c
		return Matrix3f{
			Vector3f{ cosAngle, 0.0, -sinAngle },
			Vector3f::YBASE,
			Vector3f{ sinAngle, 0.0, cosAngle }
		};
	}
	static Matrix3f rotateZ(float degree) {
		float radian = TO_RADIAN(degree);
		float cosAngle = std::cosf(radian);
		float sinAngle = std::sinf(radian);

		// c & -s \\ s & c
		return Matrix3f{
			Vector3f{ cosAngle, sinAngle, 0 },
			Vector3f{ -sinAngle, cosAngle, 0 },
			Vector3f::ZBASE
		};
	}

	static Matrix3f rotate(const Vector3f & axis, float degree) {
		Vector3f a = axis.normalized();
		float sinT = std::sin(TO_RADIAN(degree));
		float cosT = std::cos(TO_RADIAN(degree));
		return Matrix3f{
			Vector3f{ a.x*a.x + cosT * (1 - a.x*a.x), a.x*a.y*(1 - cosT) - sinT * a.z, a.x*a.z*(1 - cosT) + sinT * a.y },
			Vector3f{ a.x*a.y*(1 - cosT) + sinT * a.z, a.y*a.y + cosT * (1 - a.y*a.y), a.y*a.z*(1 - cosT) - sinT * a.x },
			Vector3f{ a.x*a.z*(1 - cosT) - sinT * a.y, a.y*a.z*(1 - cosT) + sinT * a.x, a.z*a.z + cosT * (1 - a.z*a.z) }
		};
	}

	static const Matrix3f IDENTITY;
};

std::ostream & operator << (std::ostream &, const Matrix3f & m);


class Matrix4f {
public:
	Vector4f c0, c1, c2, c3;
	Matrix4f() :
		c0{ Vector4f::XBASE }, c1{ Vector4f::YBASE }, c2{ Vector4f::ZBASE }, c3{ Vector4f::WBASE } {}
	Matrix4f(const Vector4f &a, const Vector4f &b, const Vector4f &c, const Vector4f &d) :
		c0{ a }, c1{ b }, c2{ c }, c3{ d } {
            
        }
	Matrix4f(const Vector4f _c[4]) :
		c0{ _c[0] }, c1{ _c[1] }, c2{ _c[2] }, c3{ _c[3] } {}
	explicit Matrix4f(const Matrix3f &m) :
		c0{ m.columns[0] }, c1{ m.columns[1] }, c2{ m.columns[2] }, c3{ Vector4f::WBASE } {}
	Matrix4f(const float m[16], bool t = false) {
		memcpy(&c0, m, sizeof(float) * 16);
		if (t) transpose();
	}

	Vector4f & operator[](int i) { return (&c0)[i]; }
	const Vector4f & operator[](int i) const { return (&c0)[i]; }

	Matrix4f(const Matrix4f &m) = default;
	Matrix4f& operator=(const Matrix4f &m) = default;

	Matrix4f & operator+=(const Matrix4f &rhs) {
		c0 += rhs.c0; c1 += rhs.c1; c2 += rhs.c2; c3 += rhs.c3;
		return *this;
	}
	Matrix4f & operator -= (const Matrix4f &rhs) {
		c0 -= rhs.c0; c1 -= rhs.c1; c2 -= rhs.c2; c3 -= rhs.c3;
		return *this;
	}
	Matrix4f & operator *= (float s) {
		c0 *= s; c1 *= s; c2 *= s; c3 *= s;
		return *this;
	}
	Matrix4f & operator /= (float s) {
		float inv = 1.0f / s;
		c0 *= inv; c1 *= inv; c2 *= inv; c3 *= inv;
		return *this;
	}

	Matrix4f operator + (const Matrix4f &rhs) const { return Matrix4f{ *this } += rhs; }
	Matrix4f operator - (const Matrix4f &rhs) const { return Matrix4f{ *this } -= rhs; }
	Matrix4f operator * (float s) const { return Matrix4f{ *this } *= s; }
	Matrix4f operator / (float s) const { return Matrix4f{ *this } /= s; }

	Vector4f operator * (const Vector4f &v) const {
		//return c0*v.x + c1*v.y + c2*v.z + c3*v.w;
		__m128 i0 = _mm_mul_ps(_mm_set_ps1(v.x), c0.simdData);
		__m128 i1 = _mm_mul_ps(_mm_set_ps1(v.y), c1.simdData);
		__m128 i2 = _mm_mul_ps(_mm_set_ps1(v.z), c2.simdData);
		__m128 i3 = _mm_mul_ps(_mm_set_ps1(v.w), c3.simdData);

		__m128 i01 = _mm_add_ps(i0, i1);
		__m128 i23 = _mm_add_ps(i2, i3);
		return Vector4f{ _mm_add_ps(i01, i23) };

	}
	Matrix4f operator * (const Matrix4f &rhs) const {
		Matrix4f res;
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
	return (Vector3f{x}.cross(Vector3f{y}).dot(Vector3f{z});
	}
	*/
	float frobeniusNorm() const {
		return std::sqrt(
			c0.lengthSquared() + c1.lengthSquared() + c2.lengthSquared() + c3.lengthSquared()
		);
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

	static Matrix4f perspective(float fovy_deg, float aspect, float _near, float _far) {
		const float fovy_rad = TO_RADIAN(fovy_deg);
		const float tanHalfFovy = std::tanf(fovy_rad / 2.0f);
		Matrix4f res = Matrix4f::ZERO;
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
	static Matrix4f translate(const Vector3f &t) {
		Matrix4f res;
		res.c3 = Vector4f{ t, 1.0f };
		return res;
	}

	static Matrix4f diag(const Vector3f &diag) {
		Matrix4f res;
		res.c0.x = diag.x;
		res.c1.y = diag.y;
		res.c2.z = diag.z;
		return res;
	}

	static Matrix4f diag(float scale) {
		return Matrix4f::diag(Vector3f{ scale, scale, scale });
	}

	// Conversion to float *
	explicit operator float *() { return &(c0.x); }
	explicit operator const float *() const { return &(c0.x); }

	static Matrix4f rotateX(float degree) {
		float radian = TO_RADIAN(degree);
		float cosAngle = std::cosf(radian);
		float sinAngle = std::sinf(radian);

		return Matrix4f{
			Vector4f::XBASE,
			Vector4f{ 0, cosAngle, sinAngle, 0.0f },
			Vector4f{ 0.0f, -sinAngle, cosAngle, 0.0f },
			Vector4f::WBASE
		};
	}
	static Matrix4f rotateY(float degree) {
		float radian = TO_RADIAN(degree);
		float cosAngle = std::cosf(radian);
		float sinAngle = std::sinf(radian);

		Matrix4f ret = Matrix4f::IDENTITY;
		ret.c0.simdData = _mm_set_ps(0.0f, -sinAngle, 0.0f, cosAngle);
		ret.c2.simdData = _mm_set_ps(0.0f, cosAngle, 0.0f, sinAngle);
		return ret;
	}
	static Matrix4f rotateZ(float degree) {
		float radian = TO_RADIAN(degree);
		float cosAngle = std::cosf(radian);
		float sinAngle = std::sinf(radian);

		Matrix4f ret = Matrix4f::IDENTITY;
		ret.c0.simdData = _mm_set_ps(0.0f, 0.0f, sinAngle, cosAngle);
		ret.c1.simdData = _mm_set_ps(0.0f, 0.0f, cosAngle, -sinAngle);
		return ret;
	}
	static Matrix4f rotate(const Vector3f & axis, float degree) {
		Matrix3f m3 = Matrix3f::rotate(axis, degree);
		Matrix4f ret;
		ret.c0 = Vector4f{ m3.columns[0] };
		ret.c1 = Vector4f{ m3.columns[1] };
		ret.c2 = Vector4f{ m3.columns[2] };
		return ret;
	}

	static Matrix4f lookAt(const Point3f & origin, const Point3f & target, const Vector3f & up) {
		return Matrix4f::lookAt(origin, target - origin, up);
	}
	static Matrix4f lookAt(const Point3f & origin, const Vector3f & viewDir, const Vector3f & up) {
#if COORDINATE_SYSTEM_HANDEDNESS == COORDINATE_SYSTEM_RIGHT_HANDED
		// viewDir = -z, up = y, right = x
		Vector3f z = -viewDir.normalized();
		Vector3f x = up ^ z;
		x.normalize();
		Vector3f y = z ^ x;
		y.normalize(); // may be unnecessary.
#else
		// if left-handed system
		// viewDir = z, up = y, right = x
		Vector3f z = viewDir.normalized();
		Vector3f x = up ^ z;
		x.normalize();
		Vector3f y = z ^ x;
		y.normalize();
#endif
		// Let M = Matrix{x, y, z}.
		// then p'=M(p) transforms p from object space to camera's space.
		// What _lookAt_ transform is actually an INVERSE of such transform.
		// In homogeneous coordinate,
		// Inv(M) = | R T |  where R = | x y z |
		// 			| 0 1 |  so M(p) = R(p)+T = T(R(p)).
		// So M =   | R' -R'T |
		// 			| 0    1  | 
		Matrix4f ret{
			Vector4f{ x }, Vector4f{ y }, Vector4f{ z }, Vector4f::WBASE
		};
		ret.transpose();
		ret.c3 = -(ret * Vector4f{ origin, -1.0f }); // -1.0f is used here so that it will eventually become 1.0f with the leading '-' operator.
		return ret;
	}
	const static Matrix4f IDENTITY;
	const static Matrix4f ZERO;
};

} // namespace hcm

std::ostream & operator << (std::ostream &, const hcm::Matrix4f & m);

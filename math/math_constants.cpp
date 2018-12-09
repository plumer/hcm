#include <iostream>
#include <cassert>

#include "misc.hpp"
#include "geometry.hpp"
#include "geometry_compact.hpp"
#include "geometry2d.hpp"
#include "transform.hpp"
#ifdef _GLWEEL_MATH_EXTENDED_OBJECTS
#include "interaction.hpp"
#endif
#include "quaternion.hpp"


namespace hcm {

// static const for Vector
const Vector3f Vector3f::XBASE{ 1.0f, 0.0f, 0.0f };
const Vector3f Vector3f::YBASE{ 0.0f, 1.0f, 0.0f };
const Vector3f Vector3f::ZBASE{ 0.0f, 0.0f, 1.0f };
const Vector3f &Vector3f::XDIR = XBASE;
const Vector3f &Vector3f::YDIR = YBASE;
const Vector3f &Vector3f::ZDIR = ZBASE;

const CVector3f CVector3f::XBASE{ 1.0f, 0.0f, 0.0f };
const CVector3f CVector3f::YBASE{ 0.0f, 1.0f, 0.0f };
const CVector3f CVector3f::ZBASE{ 0.0f, 0.0f, 1.0f };
const CVector3f &CVector3f::XDIR = XBASE;
const CVector3f &CVector3f::YDIR = YBASE;
const CVector3f &CVector3f::ZDIR = ZBASE;

// >>>>>>>>>>>>>>>>>>>>>>>>>> Point3f >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

const Point3f Point3f::ORIGIN{ 0.0f, 0.0f, 0.0f };
const CPoint3f CPoint3f::ORIGIN{ 0.0f, 0.0f, 0.0f };

// <<<<<<<<<<<<<<<<<<<<<<<<<<< Point3f <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


// >>>>>>>>>>>>>>>>>>>>>>>>>>> Normal3f >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


// <<<<<<<<<<<<<<<<<<<<<<<<<<<<< Normal3f <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

// >>>>>>>>>>>>>>>>>>>>>>>>>> Vector2f >>>>>>>>>>>>>>>>>>>>>>>>>>

const Vector2f Vector2f::XBASE = Vector2f{ 1.0f, 0.0f };
const Vector2f Vector2f::YBASE = Vector2f{ 0.0f, 1.0f };
const Vector2f &Vector2f::XDIR = Vector2f::XBASE;
const Vector2f &Vector2f::YDIR = Vector2f::YBASE;
const Point2f Point2f::ORIGIN = Point2f{ 0.0f, 0.0f };
const Complex Complex::i{ 0.0f, 1.0f };

// <<<<<<<<<<<<<<<<<<<<<<<<<< Vector2f <<<<<<<<<<<<<<<<<<<<<<<<<<

// >> Vector4f >>>
const Vector4f Vector4f::XBASE{ 1.0f, 0.0f, 0.0f, 0.0f };
const Vector4f Vector4f::YBASE{ 0.0f, 1.0f, 0.0f, 0.0f };
const Vector4f Vector4f::ZBASE{ 0.0f, 0.0f, 1.0f, 0.0f };
const Vector4f Vector4f::WBASE{ 0.0f, 0.0f, 0.0f, 1.0f };
// << vec4f << 


// >>>>>>>>>>>>>>>>>>>>>>>>> Quaternion >>>>>>>>>>>>>>>>>>>>>>>>>

Quaternion Quaternion::rotater(const Vector3f &axis, float degree) {
	Vector3f a = axis.normalized();
	float sinT = std::sin(TO_RADIAN(degree/2));
	float cosT = std::cos(TO_RADIAN(degree/2));
	a *= sinT;
	return Quaternion{ a, cosT };
}

Matrix3f Quaternion::toRotMatrix() const
{
	// TODO: what if the quaternion is not unit-length?
	// float x = xyz.x, y = xyz.y, z = xyz.z;
	Vector3f col1{ 1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w) };
	Vector3f col2{ 2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w) };
	Vector3f col3{ 2 * (z*x - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y) };
	return Matrix3f{ col1, col2, col3 };
}

void Quaternion::fromRotMatrix(const Matrix3f &m) {
	Matrix3f id = m*(m.transposed());
	Matrix3f diff = id - Matrix3f::IDENTITY;
	if (diff.frobeniusNorm() < 0.001) {
		// trace(m) = 3-2(y2 + z2 + x2 + z2 + x2 + y2) = 3-4(x^2+y^2+z^2);
		// xyz*xyz = 0.25*(3-trace)
		const Vector3f &c0 = m.col(0), &c1 = m.col(1), &c2 = m.col(2);
		float vnorm = 0.5f*std::sqrt(3 - c0[0] - c1[1] - c2[2]);
		float zw = 0.25f*(c1[0] - c0[1]);
		float yw = 0.25f*(c0[2] - c2[0]);
		float xw = 0.25f*(c2[1] - c1[2]);
		Vector3f xyz_{ xw, yw, zw };
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

Quaternion slerp(float t, const Quaternion & q0, const Quaternion & q1)
{
	float cosT = dot(q0, q1)/q1.lengthSquared();
	if (cosT > 0.9995f) {
		// return a normalized linear combination instead.
		Quaternion res = q0 * (1 - t) + q1 * t;
		res.normalize();
		return res;
	}
	else {
		Quaternion qPerp = q1 - cosT * q0;
		qPerp.normalize();
		float theta = std::acos(cosT);
		return q0 * std::cos(theta*t) + qPerp * std::sin(theta*t);
	}
}

// <<<<<<<<<<<<<<<<<<<<<<<<< Quaternion <<<<<<<<<<<<<<<<<<<<<<<<




// >>>>>>>>>>>>>>>>>>>>>>>>>> Natrix3f class >>>>>>>>>>>>>>>>>>>>>>>>>

const Matrix3f Matrix3f::IDENTITY{ Vector3f{1.0f, 0.0f, 0.0f}, Vector3f{0.0f, 1.0f, 0.0f}, Vector3f{0.0f,0.0f, 1.0f} };

// <<<<<<<<<<<<<<<<<<<<<<<<<< Matrix3f class <<<<<<<<<<<<<<<<<<<<<<<<<


const __m128 Transform::M32_4_SIGNIFICAND_BITS = _mm_cvtepi32_ps(_mm_set1_epi32(0x7fff'ffff));
const __m128 Transform::M32_4_SIGN_BITS = _mm_cvtepi32_ps(_mm_set1_epi32(0x8000'0000));

// >>>>>>>>>>>>>>>>>>>>>>>>>> Matrix4f class >>>>>>>>>>>>>>>>>>>>>>>>>

const static float Matrix4fIdentity[16] = {
	1.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 1.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 1.0f, 0.0f,
	0.0f, 0.0f, 0.0f, 1.0f,
};
const static float Matrix4fZero[16] = { 0.0f };

const Matrix4f Matrix4f::IDENTITY{ Matrix4fIdentity };
const Matrix4f Matrix4f::ZERO{ Matrix4fZero };
const Transform Transform::IDENTITY{Matrix4f::IDENTITY, Matrix4f::IDENTITY};

// <<<<<<<<<<<<<<<<<<<<<<<<<< Matrix4f class <<<<<<<<<<<<<<<<<<<<<<<<<

#ifdef _GLWEEL_MATH_EXTENDED_OBJECTS
SurfaceInteraction Transform::operator()(const SurfaceInteraction &si) const {
	const Transform &m = *this;
	SurfaceInteraction res{
		m(si.position), m(si.pError), si.uv, m(si.wo),
		m(si.dpdu), m(si.dpdv), m(si.dndu), m(si.dndv),
		si.time, si.shape
	};
	// TODO: take care of p and pError;
	m(si.position, &res.pError);
	return res;
}
#endif

} // namespace hcm

std::ostream & operator<<(std::ostream &os, const hcm::Point3f & p)
{
	os << '[' << p.x << ' ' << p.y << ' ' << p.z << ']';
	return os;
}

std::ostream & operator<<(std::ostream &os, const hcm::Vector3f & v)
{
	os << '[' << v.x << ' ' << v.y << ' ' << v.z << ']';
	return os;
}

std::ostream & operator<<(std::ostream &os, const hcm::Normal3f & n)
{
	os << '[' << n.x << ' ' << n.y << ' ' << n.z << ']';
	return os;
}

std::ostream & operator << (std::ostream &os, const hcm::Quaternion &q) {
	std::ios::sync_with_stdio();
	printf("[%5.2fi %5.2fj %5.2fk + %5.2f]", q.x, q.y, q.z, q.w);
	return os;
}

std::ostream & operator<<(std::ostream &os, const hcm::Matrix3f & m)
{
	//const Vector3f &a = m.columns[0], &b = m.columns[1], &c = m.columns[2];
	std::ios::sync_with_stdio(true);
	printf("\n| %5.2f %5.2f %5.2f |\n", m.columns[0].x, m.columns[1].x, m.columns[2].x);
	printf("| %5.2f %5.2f %5.2f |\n", m.columns[0].y, m.columns[1].y, m.columns[2].y);
	printf("| %5.2f %5.2f %5.2f |\n", m.columns[0].z, m.columns[1].z, m.columns[2].z);
	return os;
}

std::ostream & operator<<(std::ostream &os, const hcm::Matrix4f & m)
{
    //const Vector3f &a = m.columns[0], &b = m.columns[1], &c = m.columns[2];
    std::ios::sync_with_stdio(true);
    printf("\n| %5.2f %5.2f %5.2f %5.2f |\n", m.c0.x, m.c1.x, m.c2.x, m.c3.x);
    printf("| %5.2f %5.2f %5.2f %5.2f |\n", m.c0.y, m.c1.y, m.c2.y, m.c3.y);
    printf("| %5.2f %5.2f %5.2f %5.2f |\n", m.c0.z, m.c1.z, m.c2.z, m.c3.z);
    printf("| %5.2f %5.2f %5.2f %5.2f |\n", m.c0.w, m.c1.w, m.c2.w, m.c3.w);
    return os;
}
                                                                         
            
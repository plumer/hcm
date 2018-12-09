#pragma once
#include <xmmintrin.h>
#include "matrix.hpp"
#ifdef _GLWHEEL_MATH_EXTENDED_OBJECTS
#include "ray.hpp"
#include "bbox.hpp"
class SurfaceInteraction;
#endif



namespace hcm {

class Transform
{
public:
	Transform() : m{ Matrix4f::IDENTITY }, mInv{ Matrix4f::IDENTITY } {}
	Transform(const Matrix4f &m, const Matrix4f &mInv) : m{ m }, mInv{ mInv } {
        
    }
	Transform(const Matrix4f &m) : m{ m }, mInv{ m } {
		mInv.invert();
	}

	// >>>>>>>>>>>>>> Operations on geometric objects >>>>>>>>>>>>>
	// Transform's notation work in a more mathematical way.
	// p' = T(p), v' = T(v), something like this.

	Point3f operator() (const Point3f &p) const {
		Vector4f homogeneousV{ p };
		Vector4f pPrime = m*homogeneousV;
		if (pPrime.w != 1.0f) {
			pPrime.homogenize();
		}
		return Point3f{ pPrime.x, pPrime.y, pPrime.z };
	}

	// Same as T(p), but writes the rounding error introduced in FP arithmetic
	//   into _pError_.
	Point3f operator() (const Point3f &p, Vector3f *pError) const {
		Vector4f homogeneousV{ p };
		homogeneousV = m * homogeneousV;
		if (homogeneousV.w != 1.0f) {
			homogeneousV.homogenize();
		}

		// compute rounding error introduced in FP arithmetic.
		// It's a simple matrix-vector multiplication but everything is
		//   converted to absolute value before adding up.
		__m128 _c0 = _mm_mul_ps(m.c0.simdData, _mm_set_ps1(p.x));
		__m128 _c1 = _mm_mul_ps(m.c1.simdData, _mm_set_ps1(p.y));
		__m128 _c2 = _mm_mul_ps(m.c2.simdData, _mm_set_ps1(p.z));
		__m128 _c3 = m.c3.simdData;


		_c0 = _mm_and_ps(_c0, M32_4_SIGNIFICAND_BITS);
		_c1 = _mm_and_ps(_c1, M32_4_SIGNIFICAND_BITS);
		_c2 = _mm_and_ps(_c2, M32_4_SIGNIFICAND_BITS);
		_c3 = _mm_and_ps(_c3, M32_4_SIGNIFICAND_BITS);
		_c0 = _mm_add_ps(_c0, _c1);
		_c2 = _mm_add_ps(_c2, _c3);
		_c0 = _mm_add_ps(_c0, _c2);
		pError->simdData = _mm_mul_ps(_c0, _mm_set_ps1(gamma(3)));
		pError->dummy = 0.0f;

		return Point3f{ homogeneousV.x, homogeneousV.y, homogeneousV.z };
	}

	// Same as T(p), but also computes the propagated error from _e_
	//   introduced in FP arithmetic, and writes it to _pE_.
	Point3f operator() (const Point3f &p, const Vector3f &e, Vector3f *pE) const {
		// computes the result as in T(p).
		Vector4f homogeneousV{p};
		homogeneousV = m * homogeneousV;
		if (homogeneousV.w != 1.0f) {
			homogeneousV.homogenize();
		}

		// math:
		// pTE.x = (gamma_3+1)(c0.x*dx + c1.x*dy + c2.x*dz) + 
		//          gamma_3(c0.x*p.x + c1.x*p.y + c2.x*p.z + c3.x)
		// pTE.y = (gamma_3+1)(c0.y*dx + c1.y*dy + c2.y*dz) + 
		//          gamma_3(c0.y*p.x + c1.y*p.y + c2.y*p.z + c3.y)
		// pTE.z = (gamma_3+1)(c0.z*dx + c1.z*dy + c2.z*dz) + 
		// 			gamma_3(c0.z*p.x + c1.z*p.y + c2.z*p.z + c3.z)
		// in SIMD:
		// pTE.xyz = (gamma_3+1)(c0*e.x + c1*e.y + c2*e.z) + 
		//			  gamma_3*(c0*p.x + c1*p.y + c2*p.z + c3)
		__m128 _c0 = _mm_mul_ps(m.c0.simdData, _mm_set_ps1(e.x));
		__m128 _c1 = _mm_mul_ps(m.c1.simdData, _mm_set_ps1(e.y));
		__m128 _c2 = _mm_mul_ps(m.c2.simdData, _mm_set_ps1(e.z));

		_c0 = _mm_and_ps(_c0, M32_4_SIGNIFICAND_BITS);
		_c1 = _mm_and_ps(_c1, M32_4_SIGNIFICAND_BITS);
		_c2 = _mm_and_ps(_c2, M32_4_SIGNIFICAND_BITS);

		__m128 sum = _mm_add_ps(_c0, _c1); sum = _mm_add_ps(sum, _c2);
		sum = _mm_mul_ps(sum, _mm_set_ps1(gamma(3) + 1));

		_c0 = _mm_mul_ps(m.c0.simdData, _mm_set_ps1(p.x));
		_c1 = _mm_mul_ps(m.c1.simdData, _mm_set_ps1(p.y));
		_c2 = _mm_mul_ps(m.c2.simdData, _mm_set_ps1(p.z));
		
		_c0 = _mm_and_ps(_c0, M32_4_SIGNIFICAND_BITS);
		_c1 = _mm_and_ps(_c1, M32_4_SIGNIFICAND_BITS);
		_c2 = _mm_and_ps(_c2, M32_4_SIGNIFICAND_BITS);
		__m128 _c3 = _mm_and_ps(m.c3.simdData, M32_4_SIGNIFICAND_BITS);

		// now sum _c0, _c1, _c2 and _c3 into _c0.
		_c0 = _mm_add_ps(_c0, _c1);
		_c2 = _mm_add_ps(_c2, _c3);
		_c0 = _mm_add_ps(_c0, _c2);
		_c0 = _mm_mul_ps(_c0, _mm_set_ps1(gamma(3)));

		pE->simdData = _mm_add_ps(sum, _c0);
		pE->dummy = 0.0f;

		return Point3f{ homogeneousV };
	}

	Vector3f operator() (const Vector3f &v) const {
		Vector4f vPrime = m.c0*v.x + m.c1*v.y + m.c2*v.z;
		// !! TODO: what if matrix has homogeneous component?
		return Vector3f{ vPrime };
	}

	Vector3f operator() (const Vector3f &v, Vector3f *pE) const {
		Vector4f vPrime = m.c0*v.x + m.c1*v.y + m.c2*v.z;
		
		// Calculate PF arithmetic error 
		// vPrime.x = m.c0.x*v.x + m.c1.x*v.y + m.c2*v.z
		// FP(vPrime.x) = FPadd(FPmul(m.c0.x, v.x), FPmul(m.c1.x, v.y), FPmul(m.c2.x, v.z))
		// \in FPadd(m.c0.x*v.x*(1+gamma_1) + m.c1.x*v.y*(1+gamma_1) + m.c2.x*v.z(1+gamma_1))
		// \in (m.c0.x*v.x + m.c1.x*v.y + m.c2*v.z)(1+gamma_3)
		// Therefore absError = (m.c0.x*v.x + m.c1.x*v.y + m.c2*v.z)*gamma_3.

		__m128 _c0 = _mm_mul_ps(m.c0.simdData, _mm_set_ps1(v.x));
		__m128 _c1 = _mm_mul_ps(m.c1.simdData, _mm_set_ps1(v.y));
		__m128 _c2 = _mm_mul_ps(m.c2.simdData, _mm_set_ps1(v.z));

		// Take absolute values before summing up.
		_c0 = _mm_and_ps(_c0, M32_4_SIGNIFICAND_BITS);
		_c1 = _mm_and_ps(_c1, M32_4_SIGNIFICAND_BITS);
		_c2 = _mm_and_ps(_c2, M32_4_SIGNIFICAND_BITS);
		
		// sum _c0, _c1 and _c2 to _c0.
		_c0 = _mm_add_ps(_c0, _c1);
		_c0 = _mm_add_ps(_c0, _c2);

		// multiply everything by gamma_3.
		pE->simdData = _mm_mul_ps(_c0, _mm_set_ps1(gamma(3)));
		pE->dummy = 0.0f;
		
		return Vector3f(vPrime);
	}

	Vector3f operator() (const Vector3f &v, const Vector3f &e, Vector3f *pE) const {
		// math:
		// pTE.x = (gamma_3+1)(c0.x*dx + c1.x*dy + c2.x*dz) + 
		//          gamma_3(c0.x*p.x + c1.x*p.y + c2.x*p.z)
		// pTE.y = (gamma_3+1)(c0.y*dx + c1.y*dy + c2.y*dz) + 
		//          gamma_3(c0.y*p.x + c1.y*p.y + c2.y*p.z)
		// pTE.z = (gamma_3+1)(c0.z*dx + c1.z*dy + c2.z*dz) + 
		// 			gamma_3(c0.z*p.x + c1.z*p.y + c2.z*p.z)
		// in SIMD:
		// pTE.xyz = (gamma_3+1)(c0*e.x + c1*e.y + c2*e.z) + 
		//			  gamma_3*(c0*p.x + c1*p.y + c2*p.z + c3)
		Vector4f vPrime = m.c0*v.x + m.c1*v.y + m.c2*v.z;

		// Calculate c0*e.x + c1*e.y + c2*e.z.
		__m128 _c0 = _mm_mul_ps(m.c0.simdData, _mm_set_ps1(e.x));
		__m128 _c1 = _mm_mul_ps(m.c1.simdData, _mm_set_ps1(e.y));
		__m128 _c2 = _mm_mul_ps(m.c2.simdData, _mm_set_ps1(e.z));

		// take the absolute value before summing up.
#define T_ABS(simd) (simd = _mm_and_ps(simd, M32_4_SIGNIFICAND_BITS))
		T_ABS(_c0); T_ABS(_c1); T_ABS(_c2);
		_c0 = _mm_add_ps(_c0, _c1);
		_c0 = _mm_add_ps(_c0, _c2);
		__m128 sum = _mm_mul_ps(_c0, _mm_set_ps1(gamma(3) + 1));

		_c0 = _mm_mul_ps(m.c0.simdData, _mm_set_ps1(v.x));
		_c1 = _mm_mul_ps(m.c1.simdData, _mm_set_ps1(v.y));
		_c2 = _mm_mul_ps(m.c2.simdData, _mm_set_ps1(v.z));
		
		T_ABS(_c0); T_ABS(_c1); T_ABS(_c2);
		_c0 = _mm_add_ps(_c0, _c1);
		_c0 = _mm_add_ps(_c0, _c2);
		_c0 = _mm_mul_ps(_c0, _mm_set_ps1(gamma(3)));
		pE->simdData = _mm_add_ps(_c0, sum);
		pE->dummy = 0.0f;
#undef T_ABS
		
		return Vector3f{ vPrime };

	}

	Normal3f operator() (const Normal3f &n) const {
		Matrix4f invT = mInv;
		invT.transpose();
		Vector4f v{ n.simdData };
		v = invT * v;
		return Normal3f{ v.simdData };
	}

#ifdef _GLWHEEL_MATH_EXTENDED_OBJECTS
	Ray operator() (const Ray &r) const {
		Vector3f oError;
		Point3f o = this->operator()(r.o, &oError);
		Vector3f d = this->operator()(r.d);
		
		// offset ray origin to edge of error bounds and compute tMax
		float tMax = r.tMax;
		float l2 = d.lengthSquared();
		if (l2 > 0) {
			float dt = oError * abs(d) / l2;
			o += d*dt;
			tMax -= dt;
		}
	
		Ray res{ o, d, r.medium };
		res.tMax = tMax;
		res.time = r.time;
		return res;
	}

	Ray operator() (const Ray &r, Vector3f *oError, Vector3f *dError) const {
		Point3f o = this->operator()(r.o, oError);
		Vector3f d = this->operator()(r.d, dError);
	
		// offset ray origin to edge of error bounds and compute tMax
		float tMax = r.tMax;
		float l2 = d.lengthSquared();
		if (l2 > 0) {
			float dt = *oError * abs(d) / l2;
			o += d*dt;
			tMax -= dt;
		}
	
		Ray res{ o, d, r.medium };
		res.tMax = tMax;
		res.time = r.time;
		return res;
	}

	Bounds3f operator() (const Bounds3f &bbox, bool useFoolProofMethod = false) const {
		if (useFoolProofMethod == false) {
			Bounds3f newB{ Point3f{this->m.c3} };
			for (int i = 0; i < 2; i++)
			{
				for (int j = 0; j < 2; ++j) {
					float a = m[j][i] * bbox.pMin[j];
					float b = m[j][i] * bbox.pMax[j];
					newB.pMin[i] += (a<b?a:b);
					newB.pMax[i] += (a>b?a:b);
				}
			}
			return newB;
		}
		else {
			Bounds3f newB;
			for (int i = 0; i < 8; ++i) {
				newB = uni(newB, bbox.corner(i));
			}
			return newB;
		}
	}

	SurfaceInteraction operator() (const SurfaceInteraction &si) const;
#endif

	Transform operator() (const Transform &rhs) const {
		Transform res;
		res.m = m*rhs.m;
		res.mInv = rhs.mInv*mInv;
		res.isRigidBody = isRigidBody & rhs.isRigidBody;
		return res;
	}
	// <<<<<<<<<<<<<< Operations on geometric objects <<<<<<<<<<<<<

	Transform inverse() const {
		return Transform(mInv, m);
	}
	void invert() {
		std::swap(m, mInv);
	}

	// >>>>>>>>>>> static methods that create transforms >>>>>>>>>>
	static Transform translate(const Vector3f &t) {
		Transform res{ Matrix4f::translate(t), Matrix4f::translate(-t) };
		res.isRigidBody = true;
		return res;
	}

	static Transform rotate(const Vector3f &axis, float degree) {
		Matrix3f rotateM = Matrix3f::rotate(axis, degree);
		Transform res{ Matrix4f(rotateM), Matrix4f(rotateM) };
		res.mInv.transpose();
		res.isRigidBody = true;
		return res;
	}

	static Transform scale(float s) {
		assert(s != 0.0f);
		Transform res;
		if (s != 1.0f) {
			res.m.c0.x = res.m.c1.y = res.m.c2.z = s;
			res.mInv.c0.x = res.mInv.c1.y = res.mInv.c2.z = 1.0f / s;
		}
		res.isRigidBody = false;
		return res;
	}

	static Transform scale(const Vector3f &scaleV) {
		assert(scaleV.x != 0.0f);
		assert(scaleV.y != 0.0f);
		assert(scaleV.z != 0.0f);
		Vector3f inverseScaleV{ 1.0f / scaleV.x, 1.0f / scaleV.y, 1.0f / scaleV.z };
		Transform res{ Matrix4f::diag(scaleV), Matrix4f::diag(inverseScaleV) };
		res.isRigidBody = false;
		return res;
	}

	// <<<<<<<<<<< static methods that create transforms <<<<<<<<<<

	Matrix4f m, mInv;
	mutable bool isRigidBody;
	mutable bool isAffine;
    
    // >>>>>>>>>>> Identity >>>>>>>>>>>>
    static const Transform IDENTITY;
    // <<<<<<<<<<< Identity <<<<<<<<<<<<
	
private:
	// This mask has bit pattern 0x7FFF'FFFF on every of its 4 components.
	// A bit-wise AND operation with this mask to another __m128 value
	//   would return the absolute value of the other operand.
	static const __m128 M32_4_SIGNIFICAND_BITS;

	static const __m128 M32_4_SIGN_BITS;
};

}// namespace hcm
#pragma once
#include <xmmintrin.h>
#include "matrix.hpp"

namespace hcm {

class Transform
{
public:
	Transform() : m{ mat4() }, mInv{ mat4() } {}
	Transform(const mat4 &m, const mat4 &mInv) : m{ m }, mInv{ mInv } {
        
    }
	Transform(const mat4 &m) : m{ m }, mInv{ m } {
		mInv.invert();
	}

	// >>>>>>>>>>>>>> Operations on geometric objects >>>>>>>>>>>>>
	// Transform's notation work in a more mathematical way.
	// p' = T(p), v' = T(v), something like this.

	pt3 operator() (const pt3 &p) const {
		vec4 homogeneousV{ p };
		vec4 pPrime = m*homogeneousV;
		if (pPrime.w != 1.0f) {
			pPrime.homogenize();
		}
		return pt3{ pPrime.x, pPrime.y, pPrime.z };
	}

	// Same as T(p), but writes the rounding error introduced in FP arithmetic
	//   into _pError_.
	pt3 operator() (const pt3 &p, vec3 *pError) const {
		vec4 homogeneousV{ p };
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
		
		static __m128 abs_mask = _mm_cvtepi32_ps(_mm_set1_epi32(0x7FFFFFFF));
		// take absolute values of _c0, _c1, _c2, _c3.
		_c0 = _mm_and_ps(_c0, abs_mask);
		_c1 = _mm_and_ps(_c1, abs_mask);
		_c2 = _mm_and_ps(_c2, abs_mask);
		_c3 = _mm_and_ps(_c3, abs_mask);
		_c0 = _mm_add_ps(_c0, _c1);
		_c2 = _mm_add_ps(_c2, _c3);
		_c0 = _mm_add_ps(_c0, _c2);
		pError->simdData = _mm_mul_ps(_c0, _mm_set_ps1(gamma(3)));
		pError->dummy = 0.0f;

		return pt3{ homogeneousV.x, homogeneousV.y, homogeneousV.z };
	}

	// Same as T(p), but also computes the propagated error from _e_
	//   introduced in FP arithmetic, and writes it to _pE_.
	pt3 operator() (const pt3 &p, const vec3 &e, vec3 *pE) const {
		// computes the result as in T(p).
		vec4 homogeneousV{p};
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

		static __m128 abs_mask = _mm_cvtepi32_ps(_mm_set1_epi32(0x7FFFFFFF));
		_c0 = _mm_and_ps(_c0, abs_mask);
		_c1 = _mm_and_ps(_c1, abs_mask);
		_c2 = _mm_and_ps(_c2, abs_mask);

		__m128 sum = _mm_add_ps(_c0, _c1); sum = _mm_add_ps(sum, _c2);
		sum = _mm_mul_ps(sum, _mm_set_ps1(gamma(3) + 1));

		_c0 = _mm_mul_ps(m.c0.simdData, _mm_set_ps1(p.x));
		_c1 = _mm_mul_ps(m.c1.simdData, _mm_set_ps1(p.y));
		_c2 = _mm_mul_ps(m.c2.simdData, _mm_set_ps1(p.z));
		
		_c0 = _mm_and_ps(_c0, abs_mask);
		_c1 = _mm_and_ps(_c1, abs_mask);
		_c2 = _mm_and_ps(_c2, abs_mask);
		__m128 _c3 = _mm_and_ps(m.c3.simdData, abs_mask);

		// now sum _c0, _c1, _c2 and _c3 into _c0.
		_c0 = _mm_add_ps(_c0, _c1);
		_c2 = _mm_add_ps(_c2, _c3);
		_c0 = _mm_add_ps(_c0, _c2);
		_c0 = _mm_mul_ps(_c0, _mm_set_ps1(gamma(3)));

		pE->simdData = _mm_add_ps(sum, _c0);
		pE->dummy = 0.0f;

		return pt3{ homogeneousV };
	}

	vec3 operator() (const vec3 &v) const {
		vec4 vPrime = m.c0*v.x + m.c1*v.y + m.c2*v.z;
		// !! TODO: what if matrix has homogeneous component?
		return vec3{ vPrime };
	}

	vec3 operator() (const vec3 &v, vec3 *pE) const {
		vec4 vPrime = m.c0*v.x + m.c1*v.y + m.c2*v.z;
		
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
		static __m128 abs_mask = _mm_cvtepi32_ps(_mm_set1_epi32(0x7FFFFFFF));
		_c0 = _mm_and_ps(_c0, abs_mask);
		_c1 = _mm_and_ps(_c1, abs_mask);
		_c2 = _mm_and_ps(_c2, abs_mask);
		
		// sum _c0, _c1 and _c2 to _c0.
		_c0 = _mm_add_ps(_c0, _c1);
		_c0 = _mm_add_ps(_c0, _c2);

		// multiply everything by gamma_3.
		pE->simdData = _mm_mul_ps(_c0, _mm_set_ps1(gamma(3)));
		pE->dummy = 0.0f;
		
		return vec3(vPrime);
	}

	vec3 operator() (const vec3 &v, const vec3 &e, vec3 *pE) const {
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
		vec4 vPrime = m.c0*v.x + m.c1*v.y + m.c2*v.z;

		// Calculate c0*e.x + c1*e.y + c2*e.z.
		__m128 _c0 = _mm_mul_ps(m.c0.simdData, _mm_set_ps1(e.x));
		__m128 _c1 = _mm_mul_ps(m.c1.simdData, _mm_set_ps1(e.y));
		__m128 _c2 = _mm_mul_ps(m.c2.simdData, _mm_set_ps1(e.z));

		// take the absolute value before summing up.
		static __m128 abs_mask = _mm_cvtepi32_ps(_mm_set1_epi32(0x7FFFFFFF));
#define T_ABS(simd) (simd = _mm_and_ps(simd, abs_mask))
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
		
		return vec3{ vPrime };

	}

	normal3 operator() (const normal3 &n) const {
		mat4 invT = mInv;
		invT.transpose();
		vec4 v{ n.simdData };
		v = invT * v;
		return normal3{ v.simdData };
	}

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
	static Transform translate(const vec3 &t) {
		Transform res{ mat4::translate(t), mat4::translate(-t) };
		res.isRigidBody = true;
		return res;
	}

	static Transform rotate(const vec3 &axis, float degree) {
		mat3 rotateM = mat3::rotate(axis, degree);
		Transform res{ mat4(rotateM), mat4(rotateM) };
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

	static Transform scale(const vec3 &scaleV) {
		assert(scaleV.x != 0.0f);
		assert(scaleV.y != 0.0f);
		assert(scaleV.z != 0.0f);
		vec3 inverseScaleV{ 1.0f / scaleV.x, 1.0f / scaleV.y, 1.0f / scaleV.z };
		Transform res{ mat4::diag(scaleV), mat4::diag(inverseScaleV) };
		res.isRigidBody = false;
		return res;
	}

	// <<<<<<<<<<< static methods that create transforms <<<<<<<<<<

	mat4 m, mInv;
	mutable bool isRigidBody;
	mutable bool isAffine;
	
private:
};

}// namespace hcm
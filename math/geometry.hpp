#ifndef _GLWHEEL_MATH_GEOMETRY
#define _GLWHEEL_MATH_GEOMETRY
#include <cassert>
#include <cmath>
#include <cinttypes>
#include <xmmintrin.h>
#include <smmintrin.h> // SSE 4.1, _mm_dp_ps
#include <immintrin.h>

#include "config.hpp"

#ifdef GM_COMPACT_3D_OBJECT
#include "geometry_compact.hpp"
#else

#ifdef __APPLE__
#define SIMD_ARRAY_ACCESS(sd, index) (reinterpret_cast<float *>(&sd)[index])
#define SIMD_ARRAY_READ(sd, index) (reinterpret_cast<const float *>(&sd)[index])
#endif
#ifdef _WIN32
#define SIMD_ARRAY_ACCESS(sd, index) ((sd).m128_f32[index])
#define SIMD_ARRAY_READ(sd, index) ((sd).m128_f32[index])
#endif

#define PBRT_MIN(a, b) (((a) > (b)) ? (b) : (a))
#define PBRT_MAX(a, b) (((a) > (b)) ? (a) : (b))

namespace hcm {

// +--------------------------------------------------------+
// |     class definition: vec3, vec4, pt3 and normal3      |
// +--------------------------------------------------------+

class pt3;
class normal3;

class vec3 {
public:
	union {
		// memory layout:
		/*
		 i[3]  i[2]  i[1]  i[0]
		+-----+-----+-----+-----+
		|dummy|  z  |  y  |  x  |
		+-----+-----+-----+-----+
		 */
		struct {
			float x, y, z;
			float dummy;
		};
		__m128 simdData;
	};

	// Constructors
	vec3() { simdData = _mm_setzero_ps(); }
	explicit vec3(const __m128 & _simd) { simdData = _simd; dummy = 0.0f; }
	vec3(float _x, float _y, float _z) { simdData = _mm_set_ps(0.0f, _z, _y, _x); }

	// Conversion to other types
	explicit operator pt3() const;
	explicit operator normal3() const;

	// element access, dangerous
	explicit operator float *() { return &x; }
	explicit operator const float *() const { return &x; }
	float & operator [] (int i)       { return SIMD_ARRAY_ACCESS(simdData, i); }
	float   operator [] (int i) const { return SIMD_ARRAY_READ(simdData, i); }

	// >>>>>>>>>>>>> arithmetic operations >>>>>>>>>>>>>>

	vec3 & operator += (const vec3 & rhs) {
		this->simdData = _mm_add_ps(this->simdData, rhs.simdData);
		return *this;
	}
	vec3 & operator -= (const vec3 & rhs) {
		simdData = _mm_sub_ps(simdData, rhs.simdData);
		return *this;

	}
	vec3 & operator *= (float s) {
		__m128 multiplier = _mm_set_ps1(s);
		simdData = _mm_mul_ps(simdData, multiplier);
		return *this;
	}
	vec3 & operator /= (float s) {
		__m128 inverse = _mm_set_ps1(1.0f / s);
		simdData = _mm_mul_ps(simdData, inverse);
		return *this;
	}
	// <<<<<<<<<<<<< arithmetic operations <<<<<<<<<<<<<

	void flip_direction() {
		static const __m128i intNegator = _mm_set1_epi32(0x80000000);
		this->simdData = _mm_castsi128_ps(
			_mm_xor_si128(intNegator, _mm_castps_si128(this->simdData))
		);
	}

	// 2-norm and normalization
	float length() const {
		return std::sqrt(length_squared());
	}
	float length_squared() const {
		__m128 res = _mm_dp_ps(simdData, simdData, 0x71);
		return _mm_cvtss_f32(res);
	}
	void normalize() {
		//float length2 = length_squared();
		__m128 invLength = _mm_set_ps1(1.0f / length());
		//__m128 lsquared = _mm_dp_ps(simdData, simdData, 0x7f);
		//__m128 invLength = _mm_invsqrt_ps(lsquared);
		simdData = _mm_mul_ps(simdData, invLength);
	}
	vec3 normalized() const {
		__m128 invLength = _mm_set_ps1(1.0f / length());
		return vec3{ _mm_mul_ps(simdData, invLength) };
	}

	static vec3 xbase() { return vec3{1.0f, 0.0f, 0.0f}; }
	static vec3 ybase() { return vec3{0.0f, 1.0f, 0.0f}; }
	static vec3 zbase() { return vec3{0.0f, 0.0f, 1.0f}; }
	// static const vec3 &XDIR;
	// static const vec3 &YDIR;
	// static const vec3 &ZDIR;
	// static const vec3 xbase;
	// static const vec3 ybase;
	// static const vec3 zbase;

	// static builders
	static vec3 normalize(float x, float y, float z) {
		vec3 res{ x, y, z };
		res.normalize();
		return res;
	}
};

// const vec3 vec3::xbase{1.0f, 0.0f, 0.0f};
// const vec3 vec3::ybase{0.0f, 1.0f, 0.0f};
// const vec3 vec3::zbase{0.0f, 0.0f, 1.0f};

// class pt3

class pt3 {
public:
	union {
		/* memory layout:
		 i[3]  i[2]  i[1]  i[0]
		+-----+-----+-----+-----+
		|dummy|  z  |  y  |  x  |
		+-----+-----+-----+-----+
		*/
		struct {
			float x, y, z;
			float dummy;
		};
		__m128 simdData;
	};

	// default constructor; construct from coordinates
	pt3() { simdData = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f); }
	pt3(float _x, float _y, float _z) { simdData = _mm_set_ps(1.0f, _z, _y, _x); }
	pt3(const __m128 & _simd) { simdData = _simd; }

	// conversion to vec3
	explicit operator vec3() const { return vec3{x, y, z}; }

	// element access, dangerous
	explicit operator float * () {return &x;}
	explicit operator const float * () const {return &x;}
	float & operator [] (int i)       { return SIMD_ARRAY_ACCESS(simdData, i); }
	float   operator [] (int i) const { return SIMD_ARRAY_READ(simdData, i); }

	// point +/+= vec, point -/-= vec
	pt3 & operator += (const vec3 & rhs) {
		simdData = _mm_add_ps(simdData, rhs.simdData);
		return *this;
	}
	pt3 & operator -= (const vec3 & rhs) {
		simdData = _mm_sub_ps(simdData, rhs.simdData);
		return *this;
	}

// scale a point
#ifdef GLWHEEL_MATH_GEOMETRY_USE_PXF

	pt3 & operator *= (float scale) {
		simdData = _mm_mul_ps(simdData, _mm_set1_ps(scale));
		return *this; 
	}
#endif

	// static methods
	static pt3 origin() { return pt3{0.0f, 0.0f, 0.0f}; }
	static pt3 all(float x) { return pt3{ _mm_set_ps(1.0f, x, x, x) }; }
};


class normal3 {
public:
	union {
		/* memory layout:
		i[3]  i[2]  i[1]  i[0]
		+-----+-----+-----+-----+
		|dummy|  z  |  y  |  x  |
		+-----+-----+-----+-----+
		*/
		struct {
			float x, y, z;
			float dummy;
		};
		__m128 simdData;
	};

	// constructor: default = zdir
	normal3() {
		simdData = _mm_set_ps(0.0f, 1.0f, 0.0f, 0.0f);
	}
	normal3(float x, float y, float z) {
		simdData = _mm_set_ps(0.0f, z, y, x);
	}

	explicit normal3(const __m128 & s) : simdData{ s } {};

	// element access: dangerous
	float & operator [] (int i)       { return SIMD_ARRAY_ACCESS(simdData, i); };
	float   operator [] (int i) const { return SIMD_ARRAY_READ(simdData, i); };
	
	// conversion to vec3
	explicit operator vec3() const {return vec3{simdData};}
	
	// n += n, n -= n, n *= s, n /=s
	normal3 & operator += (const normal3 &rhs) {
		this->simdData = _mm_add_ps(this->simdData, rhs.simdData);
		return *this;
	}
	normal3 & operator -= (const normal3 &rhs) {
		this->simdData = _mm_sub_ps(this->simdData, rhs.simdData);
		return *this;
	}
	normal3 & operator *= (float s) {
		__m128 multiplier = _mm_set_ps1(s);
		this->simdData = _mm_mul_ps(this->simdData, multiplier);
		return *this;
	}
	normal3 & operator /= (float s) {
		assert(s != 0.0f);
		__m128 inverse = _mm_set_ps1(1.0f / s);
		this->simdData = _mm_mul_ps(this->simdData, inverse);
		return *this;
	}

	// comparison
	bool operator == (const normal3 &rhs) const {
		return (x == rhs.x && y == rhs.y && z == rhs.z);
	}

	bool operator != (const normal3 &rhs) const {
		return !((*this) == rhs);
	}

	void     flip_direction() {
		static const __m128i intNegator = _mm_set1_epi32(0x80000000);
		this->simdData = _mm_castsi128_ps(
			_mm_xor_si128(intNegator, _mm_castps_si128(this->simdData))
		);
	}

	// normalize, length, normalized
	float	 length() const {
		return std::sqrtf(length_squared());
	}
	float	 length_squared() const {
		__m128 res = _mm_dp_s(lhs.simdData, rhs.simdData, 0x7F);
		return _mm_cvtss_f32(res);
	}
	normal3  normalized() const {
		normal3 res;
		res.normalize();
		return res;
	}
	void	 normalize() {
		float l = this->length();
		assert(l != 0.0f);
		if (l != 1.0f) {
			this->operator/=(l);
		}
	}
}; // normal3

// >>>>>>>>>>>>> Solve forward declaration here >>>>>>>>>>>>>

inline vec3::operator pt3() const { return pt3{simdData}; }
inline vec3::operator normal3() const {return normal3{simdData};}

#undef SIMD_ARRAY_ACCESS
#undef SIMD_ARRAY_READ

#endif // GM_COMPACT_3D_OBJECT

// <<<<<<<<<<<<< Solve forward declaration above <<<<<<<<<<<<

class vec4 {
public: 
	union {
		struct {
			float x, y, z, w;
		};
		__m128 simdData;
	};
	
	// >>>>>>>>>>> Constructors >>>>>>>>>>>
	vec4() {
		simdData = _mm_setzero_ps();
		assert((((uint64_t)(this)) & 0xF) == 0);
	}
	explicit vec4(const __m128 & _intrin) {
		simdData = _intrin;
	}
	
	vec4(const vec4 & copy) { simdData = copy.simdData; }
	
	vec4(float _x, float _y, float _z, float _w) {
		simdData = _mm_set_ps(_w, _z, _y, _x);
	}
	
	explicit vec4(const vec3 &v3, float _w = 0.0f) {
		simdData = _mm_set_ps(_w, v3.z, v3.y, v3.x);
	}

	explicit vec4(const pt3 &p, float _w = 1.0f) {
		simdData = _mm_set_ps(_w, p.z, p.y, p.x);
	}
	// <<<<<<<<<<< Constructors <<<<<<<<<<<

	float & operator [] (int i) { return (&x)[i]; }
	float   operator [] (int i) const { return (&x)[i]; }

	// conversion to Vec3, Pt3 and Norm3
	explicit operator vec3 () const { vec3 res{ simdData }; res.dummy = 0.0f; return res; }
	explicit operator pt3  () const {
		pt3 res{ simdData };
		if (res.dummy != 1.0f && res.dummy != 0.0f)
			res.simdData = _mm_div_ps(res.simdData, _mm_set_ps1(res.dummy));
		else
			res.dummy = 1.0f;
		return res;
	}
	explicit operator normal3 () const { normal3 res{ simdData }; res.dummy = 0.0f; return res; }

	// >>>>>>>>>>> Arithmetic >>>>>>>>>>>>>


	vec4 & operator += (const vec4 & rhs) {
		//	x += rhs.x; y += rhs.y; z += rhs.z; w += rhs.w;
		simdData = _mm_add_ps(simdData, rhs.simdData);
		return *this;
	}
	vec4 & operator -= (const vec4 & rhs) {
		//x -= rhs.x; y -= rhs.y; z -= rhs.z; w -= rhs.w;
		simdData = _mm_sub_ps(simdData, rhs.simdData);
		return *this;
	}
	vec4 & operator *= (float s) {
		__m128 multiplier = _mm_set_ps1(s);
		simdData = _mm_mul_ps(simdData, multiplier);
		return *this;
	}
	vec4 & operator /= (float s) {
		assert(s != 0.0f);
		__m128 inverse = _mm_set_ps1(1.0f / s);
		simdData = _mm_mul_ps(simdData, inverse);
		return *this;
	}

	// unary
	vec4 operator - () const {
		static const __m128i intNegator = _mm_set1_epi32(0x80000000);
		vec4 res;
		res.simdData = _mm_castsi128_ps(
			_mm_xor_si128(intNegator, _mm_castps_si128(this->simdData))
		);
		return res;
	}

	// <<<<<<<<<< Arithmetic <<<<<<<<<<<<<

	void homogenize() {
		if (w == 1.0f || w == 0.0f) return;
		else this->operator*=(1.0f / w);
	}
	vec4 homogenized() const {
		vec4 res = *this;
		res.homogenize();
		return res;
	}

	float length_squared() const {
		__m128 dot = _mm_dp_ps(simdData, simdData, 0xff);
		return _mm_cvtss_f32(dot);
	}

	static vec4 xbase() { return vec4{1.0f, 0.0f, 0.0f, 0.0f}; }
	static vec4 ybase() { return vec4{0.0f, 1.0f, 0.0f, 0.0f}; }
	static vec4 zbase() { return vec4{0.0f, 0.0f, 1.0f, 0.0f}; }
	static vec4 wbase() { return vec4{0.0f, 0.0f, 0.0f, 1.0f}; }
};

// +--------------------------------------------------------+
// |                     global functions                   |
// +--------------------------------------------------------+

inline float dot(const vec3 &v, const vec3 &u) { 
	__m128 res = _mm_dp_ps(v.simdData, u.simdData, 0x7f);
	return _mm_cvtss_f32(res);
}
inline float dot(const vec3 &v, const normal3 &n) { return dot(v, vec3{n}); }
inline float dot(const normal3 &n, const vec3 &v) { return dot(vec3{n}, v); }
inline vec3 cross(const vec3 &v, const vec3 &u) {
	/* | y | z | x |   | z | x | y | this
		| z | x | y | - | y | z | x | rhs
	*/
	// maybe _mm_shuffle_ps(simdData, simdData, ???)?
	__m128 lhsLS = _mm_permute_ps(v.simdData, 0b11001001);
	__m128 lhsRS = _mm_permute_ps(v.simdData, 0b11010010);
	__m128 rhsLS = _mm_permute_ps(u.simdData, 0b11001001);
	__m128 rhsRS = _mm_permute_ps(u.simdData, 0b11010010);
	__m128 subtractend = _mm_mul_ps(lhsLS, rhsRS);
	__m128 subtractor  = _mm_mul_ps(lhsRS, rhsLS);
	return vec3{ _mm_sub_ps(subtractend, subtractor) };

}

inline vec3 abs(const vec3 &v) {
	return vec3{ std::abs(v.x), std::abs(v.y), std::abs(v.z) };
	__m128i intBits = _mm_cvttps_epi32(v.simdData);
	__m128i iMask = _mm_set1_epi32(0x7fff'ffff);
	__m128i absIntBits = _mm_and_si128(intBits, iMask);
	__m128 res = _mm_cvtepi32_ps(absIntBits);
	return vec3{ res };

	//static const __m128 mask = _mm_cvtepi32_ps(_mm_set1_epi32(0x7fff'ffff));
	//return vec3{_mm_and_ps(v.simdData, mask)};
}

inline float dist(const pt3 &p, const pt3 &q) { return (p-q).length(); }
inline float absDot(const vec3 &v, const vec3 &u) { return std::fabs(v*u); }
inline float absDot(const vec3 &v, const normal3 &n) { return std::fabs(n*v); }
inline float absDot(const normal3 &n, const vec3 &v) { return std::fabs(n*v); }

inline float max(const vec3 &v) {
	return PBRT_MAX(v.x, PBRT_MAX(v.y, v.z));
}
inline float min(const vec3 &v) {
	return PBRT_MIN(v.x, PBRT_MIN(v.y, v.z));
}
inline uint8_t argmax(const vec3 &v) {
	uint8_t res = 0;
	if (v[res] < v[1]) res = 1;
	if (v[res] < v[2]) res = 2;
	return res;
}
inline uint8_t argmin(const vec3 &v) {
	uint8_t res = 0;
	if (v[res] > v[1]) res = 1;
	if (v[res] > v[2]) res = 2;
	return res;
}

inline pt3 permute(const pt3 &p, int ix, int iy, int iz) { return pt3{ p[ix], p[iy], p[iz] }; }
inline vec3 permute(const vec3 &v, int ix, int iy, int iz) { return vec3{ v[ix], v[iy], v[iz] }; }
inline void coordSystem(const vec3 &v0, vec3 *v1, vec3 *v2) {
	if (v0.length_squared() == 0) {
		*v1 = vec3::xbase(); *v2 = vec3::ybase();
	} else {
		int dim = argmax(abs(v0));
		// TODO
	}
}

inline pt3 lerp(const pt3 &p, const pt3 &q, float t) {
	return p + (q - p)*t;
}


// +--------------------------------------------------------+
// |              global operator overloading               |
// +--------------------------------------------------------+

// v+v, v-v, v*s, v/s
inline vec3 operator + (const vec3 & lhs, const vec3 & rhs) { 
	return vec3{ _mm_add_ps(lhs.simdData, rhs.simdData) };
}
inline vec3 operator - (const vec3 & lhs, const vec3 & rhs) {
	return vec3{ _mm_sub_ps(lhs.simdData, rhs.simdData) };
}
inline vec3 operator * (const vec3 & lhs, float s) {
	__m128 multiplier = _mm_set_ps1(s);
	return vec3{ _mm_mul_ps(lhs.simdData, multiplier) };
}
// s*v, the other order of operands
inline vec3 operator*(float s, const vec3 & v) {
	return v * s;
}
inline vec3 operator / (const vec3 & lhs, float s) {
	assert(s != 0.0f);
	__m128 multiplier = _mm_set_ps1(1.0f / s);
	return vec3{ _mm_mul_ps(lhs.simdData, multiplier) };
}
// unary -v: return vector in opposite direction
inline vec3 operator - (const vec3 &v) {
	static const __m128i intNegator = _mm_set1_epi32(0x80000000);
	vec3 ret;
	ret.simdData = _mm_castsi128_ps(
		_mm_xor_si128(intNegator, _mm_castps_si128(v.simdData))
	);
	return ret;
}

// p+v, p-v, p-p -> v
inline pt3 operator + (const pt3 &p, const vec3 & rhs) {
	pt3 res;
	res.simdData = _mm_add_ps(p.simdData, rhs.simdData);
	return res;
}
// commutative operation: p+v and v+p
inline pt3 operator + (const vec3 & v, const pt3 & p) {
	return p + v;
}
inline pt3 operator - (const pt3 &p, const vec3 & rhs) {
	pt3 res;
	res.simdData = _mm_sub_ps(p.simdData, rhs.simdData);
	return res;
}
// point - point
inline vec3 operator - (const pt3 &lhs, const pt3 & rhs) {
	vec3 res;
	res.simdData = _mm_sub_ps(lhs.simdData, rhs.simdData);
	return res;
}

// n+n, n-n, n*s, n/s
inline normal3 operator + (const normal3 &lhs, const normal3 &rhs) {
	normal3 res;
	res.simdData = _mm_add_ps(lhs.simdData, rhs.simdData);
	return res;
}
inline normal3 operator - (const normal3 &lhs, const normal3 &rhs) {
	normal3 res;
	res.simdData = _mm_sub_ps(lhs.simdData, rhs.simdData); 
	return res;
}
inline normal3 operator * (const normal3 &lhs, float s) {
	__m128 multiplier = _mm_set_ps1(s);
	normal3 res;
	res.simdData = _mm_mul_ps(multiplier, lhs.simdData);
	return res;
}
inline normal3 operator / (const normal3 &lhs, float s) {
	assert(s != 0.0f);
	__m128 inverse = _mm_set_ps1(1.0f/s);
	normal3 res;
	res.simdData = _mm_mul_ps(inverse, lhs.simdData);
	return res;
}

// unary -n: return normal in opposite direction
normal3 operator -(const normal3 &n) {
	static const __m128i intNegator = _mm_set1_epi32(0x80000000);
	normal3 res;
	res.simdData = _mm_castsi128_ps(
		_mm_xor_si128(intNegator, _mm_castps_si128(n.simdData))
	);
	return res;
}

// Dot-product and cross-product
#ifdef GLWHEEL_MATH_GEOMETRY_USE_OPOV_DC
inline float operator * (const vec3 & v, const vec3 & u) {
	return dot(v, u);
}

float operator * (const normal3 &n, const vec3 &v) {
	return dot(n, v);
}
inline float operator * (const vec3 &v, const normal3 &n) {
	return dot(v, n);
}

inline vec3 operator ^ (const vec3 & lhs, const vec3 & rhs) {
	return cross(lhs, rhs);
}
#endif // GLWHEEL_MATH_GEOMETRY_USE_OPOV_DC

// scale a point
#ifdef GLWHEEL_MATH_GEOMETRY_USE_PXF
	pt3 operator * (const pt3 &p, float scale) {
		pt3 res;
		res.simdData = _mm_mul_ps(p.simdData, _mm_set1_ps(scale));
		return res;
	}
#endif

vec4 operator + (const vec4 &lhs, const vec4 &rhs) {
	return vec4{ _mm_add_ps(lhs.simdData, rhs.simdData) };
}
vec4 operator - (const vec4 &lhs, const vec4 &rhs) {
	return vec4{ _mm_sub_ps(lhs.simdData, rhs.simdData) };
}
vec4 operator * (const vec4 &lhs, float s) {
	__m128 multiplier = _mm_set_ps1(s);
	return vec4{ _mm_mul_ps(lhs.simdData, multiplier) };
}
vec4 operator / (const vec4 &lhs, float s) {
	assert(s != 0.0f);
	__m128 inverse = _mm_set_ps1(1.0f / s);
	return vec4{ _mm_mul_ps(lhs.simdData, inverse) };
}

// TODO: use align here.




inline vec3 reflect(const vec3 &normal, const vec3 &incident) {
	// 1. remove from _incident_ the component parallel to _normal_.
	auto parallel = (incident*normal) *normal/normal.length_squared();
#ifndef NDEBUG
	auto perpendicular = incident - parallel;
	float check = perpendicular * normal;
	assert(std::abs(check) < 1e-6);
#endif
	return incident - 2 * parallel;
}

} // namespace hcm


#endif //_GLWHEEL_MATH_GEOMETRY
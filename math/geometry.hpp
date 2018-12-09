#ifndef _GLWHEEL_MATH_GEOMETRY
#define _GLWHEEL_MATH_GEOMETRY
#include <iosfwd>
#include <cassert>
#include <cmath>
#include <cinttypes>
#include <xmmintrin.h>
#include <smmintrin.h> // SSE 4.1, _mm_dp_ps
#include <immintrin.h>

#ifdef GM_COMPACT_3D_OBJECT
#include "geometryCompact3D.hpp"
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

class Point3f;
class Normal3f;

class Vector3f {
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
	Vector3f() { simdData = _mm_setzero_ps(); }
	explicit Vector3f(const __m128 & _simd) { simdData = _simd; dummy = 0.0f; }
	Vector3f(float _x, float _y, float _z) { simdData = _mm_set_ps(0.0f, _z, _y, _x); }

	// Conversion to other types
	explicit Vector3f(const Point3f &p);
	explicit Vector3f(const Normal3f &n);

	// element access, dangerous
	explicit operator float *() { return &x; }
	explicit operator const float *() const { return &x; }
	float & operator [] (int i)       { return SIMD_ARRAY_ACCESS(simdData, i); }
	float   operator [] (int i) const { return SIMD_ARRAY_READ(simdData, i); }

	// >>>>>>>>>>>>> arithmetic operations >>>>>>>>>>>>>>
	Vector3f operator + (const Vector3f & rhs) const { return Vector3f{ _mm_add_ps(simdData, rhs.simdData) }; }
	Vector3f operator - (const Vector3f & rhs) const { return Vector3f{ _mm_sub_ps(simdData, rhs.simdData) }; }
	Vector3f operator * (float s) const {
		__m128 multiplier = _mm_set_ps1(s);
		return Vector3f{ _mm_mul_ps(simdData, multiplier) };
	}
	Vector3f operator / (float s) const {
		assert(s != 0.0f);
		__m128 multiplier = _mm_set_ps1(1.0f / s);
		return Vector3f{ _mm_mul_ps(simdData, multiplier) };
	}

	Vector3f & operator += (const Vector3f & rhs) {
		this->simdData = _mm_add_ps(this->simdData, rhs.simdData);
		return *this;
	}
	Vector3f & operator -= (const Vector3f & rhs) {
		simdData = _mm_sub_ps(simdData, rhs.simdData);
		return *this;

	}
	Vector3f & operator *= (float s) {
		__m128 multiplier = _mm_set_ps1(s);
		simdData = _mm_mul_ps(simdData, multiplier);
		return *this;
	}
	Vector3f & operator /= (float s) {
		__m128 inverse = _mm_set_ps1(1.0f / s);
		simdData = _mm_mul_ps(simdData, inverse);
		return *this;
	}
	// <<<<<<<<<<<<< arithmetic operations <<<<<<<<<<<<<

	// Dot-product and cross-product
	float    operator * (const Vector3f & rhs) const {
		__m128 res;
		// using 7 (0b0111) as a mask ignores dummy
		res = _mm_dp_ps(simdData, rhs.simdData, 0x71);
		return _mm_cvtss_f32(res);
	}
	Vector3f operator ^ (const Vector3f & rhs) const {
		/* | y | z | x |   | z | x | y | this
		   | z | x | y | - | y | z | x | rhs
		*/
		// maybe _mm_shuffle_ps(simdData, simdData, ???)?
		__m128 thisLS = _mm_permute_ps(simdData, 0b11'00'10'01);
		__m128 thisRS = _mm_permute_ps(simdData, 0b11'01'00'10);
		__m128 rhsLS  = _mm_permute_ps(rhs.simdData, 0b11'00'10'01);
		__m128 rhsRS  = _mm_permute_ps(rhs.simdData, 0b11'01'00'10);
		__m128 subtractend = _mm_mul_ps(thisLS, rhsRS);
		__m128 subtractor  = _mm_mul_ps(thisRS, rhsLS);
		return Vector3f{ _mm_sub_ps(subtractend, subtractor) };
	}
	float    dot(const Vector3f & rhs)   const { return (*this)*rhs; }
	Vector3f cross(const Vector3f & rhs) const { return (*this) ^ rhs; }

	// return vector in opposite direction
	Vector3f operator - () const {
		static const __m128i intNegator = _mm_set1_epi32(0x80000000);
		Vector3f ret;
		ret.simdData = _mm_castsi128_ps(
			_mm_xor_si128(intNegator, _mm_castps_si128(this->simdData))
		);
		return ret;
	}
	void     flipDirection() {
		static const __m128i intNegator = _mm_set1_epi32(0x80000000);
		this->simdData = _mm_castsi128_ps(
			_mm_xor_si128(intNegator, _mm_castps_si128(this->simdData))
		);
	}

	// 2-norm and normalization
	float length() const {
		return std::sqrt(lengthSquared());
	}
	float lengthSquared() const {
		__m128 res = _mm_dp_ps(simdData, simdData, 0x71);
		return _mm_cvtss_f32(res);
	}
	void normalize() {
		//float length2 = lengthSquared();
		__m128 invLength = _mm_set_ps1(1.0f / length());
		simdData = _mm_mul_ps(simdData, invLength);
	}
	Vector3f normalized() const {
		__m128 invLength = _mm_set_ps1(1.0f / length());
		return Vector3f{ _mm_mul_ps(simdData, invLength) };
	}

	static const Vector3f XBASE;
	static const Vector3f YBASE;
	static const Vector3f ZBASE;
	static const Vector3f &XDIR;
	static const Vector3f &YDIR;
	static const Vector3f &ZDIR;

	// static builders
	static Vector3f normalize(float x, float y, float z) {
		Vector3f res{ x, y, z };
		res.normalize();
		return res;
	}
};

// class Point3f

class Point3f {
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
	Point3f() {simdData = _mm_setzero_ps(); }
	Point3f(float _x, float _y, float _z) { simdData = _mm_set_ps(1.0f, _z, _y, _x); }
	Point3f(const __m128 & _simd) { simdData = _simd; }

	// conversion from vector3f
	explicit Point3f(const Vector3f &v) { simdData = _mm_set_ps(1.0f, v.z, v.y, v.x); }

	// element access, dangerous
	explicit operator float * () {return &x;}
	explicit operator const float * () const {return &x;}
	float & operator [] (int i)       { return SIMD_ARRAY_ACCESS(simdData, i); }
	float   operator [] (int i) const { return SIMD_ARRAY_READ(simdData, i); }

	// point +/+= vec, point -/-= vec
	Point3f operator + (const Vector3f & rhs) const {
		Point3f res;
		res.simdData = _mm_add_ps(simdData, rhs.simdData);
		return res;
	}
	Point3f operator - (const Vector3f & rhs) const {
		Point3f res;
		res.simdData = _mm_sub_ps(simdData, rhs.simdData);
		return res;
	}

	Point3f & operator += (const Vector3f & rhs) {
		simdData = _mm_add_ps(simdData, rhs.simdData);
		return *this;
	}
	Point3f & operator -= (const Vector3f & rhs) {
		simdData = _mm_sub_ps(simdData, rhs.simdData);
		return *this;
	}

	// point - point
	Vector3f operator - (const Point3f & rhs) const {
		Vector3f res;
		res.simdData = _mm_sub_ps(simdData, rhs.simdData);
		return res;
	}

	float dist(const Point3f &rhs) const {
		__m128 diff, l;
		diff = _mm_sub_ps(simdData, rhs.simdData);
		l = _mm_dp_ps(diff, diff, 0x71);
		return _mm_cvtss_f32(l);
	}

	// static methods
	static Point3f origin() { return ORIGIN; }
	static Point3f all(float x) { return Point3f{ _mm_set_ps(1.0f, x, x, x) }; }
	static const Point3f ORIGIN;
};


class Normal3f {
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
	Normal3f() {
		simdData = _mm_set_ps(0.0f, 1.0f, 0.0f, 0.0f);
	}
	Normal3f(float x, float y, float z) {
		simdData = _mm_set_ps(0.0f, z, y, x);
	}
	explicit Normal3f(const Vector3f &v) {
		simdData = v.simdData;
	}
	explicit Normal3f(const __m128 & s) : simdData{ s } {};

	float & operator [] (int i)       { return SIMD_ARRAY_ACCESS(simdData, i); };
	float   operator [] (int i) const { return SIMD_ARRAY_READ(simdData, i); };
	
	// n +/+= n, n -/-= n, n */*= s, s*n, n / or /=s
	Normal3f operator + (const Normal3f &rhs) const {
		Normal3f res;
		res.simdData = _mm_add_ps(simdData, rhs.simdData);
		return res;
	}
	Normal3f operator - (const Normal3f &rhs) const {
		Normal3f res;
		res.simdData = _mm_sub_ps(simdData, rhs.simdData); 
		return res;
	}
	Normal3f operator * (float s) const {
		__m128 multiplier = _mm_set_ps1(s);
		Normal3f res;
		res.simdData = _mm_mul_ps(multiplier, simdData);
		return res;
	}
	Normal3f operator / (float s) const {
		assert(s != 0.0f);
		__m128 inverse = _mm_set_ps1(1.0f/s);
		Normal3f res;
		res.simdData = _mm_mul_ps(inverse, simdData);
		return res;
	}
	Normal3f & operator += (const Normal3f &rhs) {
		this->simdData = _mm_add_ps(this->simdData, rhs.simdData);
		return *this;
	}
	Normal3f & operator -= (const Normal3f &rhs) {
		this->simdData = _mm_sub_ps(this->simdData, rhs.simdData);
		return *this;
	}
	Normal3f & operator *= (float s) {
		__m128 multiplier = _mm_set_ps1(s);
		this->simdData = _mm_mul_ps(this->simdData, multiplier);
		return *this;
	}
	Normal3f & operator /= (float s) {
		assert(s != 0.0f);
		__m128 inverse = _mm_set_ps1(1.0f / s);
		this->simdData = _mm_mul_ps(this->simdData, inverse);
		return *this;
	}

	Normal3f operator -() const {
		static const __m128i intNegator = _mm_set1_epi32(0x80000000);
		Normal3f res;
		res.simdData = _mm_castsi128_ps(
			_mm_xor_si128(intNegator, _mm_castps_si128(this->simdData))
		);
		return res;
	}

	// comparison
	bool operator == (const Normal3f &rhs) const {
		return (x == rhs.x && y == rhs.y && z == rhs.z);
	}

	bool operator != (const Normal3f &rhs) const {
		return !((*this) == rhs);
	}

	void     flipDirection() {
		static const __m128i intNegator = _mm_set1_epi32(0x80000000);
		this->simdData = _mm_castsi128_ps(
			_mm_xor_si128(intNegator, _mm_castps_si128(this->simdData))
		);
	}

	// dot with normal, dot with vector
	float operator * (const Normal3f &rhs) const {
		__m128 res = _mm_dp_ps(simdData, rhs.simdData, 0xFF);
		return _mm_cvtss_f32(res);
	}
	float operator * (const Vector3f &rhs) const {
		__m128 res = _mm_dp_ps(simdData, rhs.simdData, 0xFF);
		return _mm_cvtss_f32(res);
	}

	// normalize, length, normalized
	float	 length() const {
		return std::sqrtf(lengthSquared());
	}
	float	 lengthSquared() const {
		return (*this)*(*this);
	}
	Normal3f normalized() const {
		float l = this->length();
		assert(l != 0.0f);
		if (l != 1.0f)
			return (*this) / l;
		else
			return (*this);
	}
	void	 normalize() {
		float l = this->length();
		assert(l != 0.0f);
		if (l != 1.0f) {
			this->operator/=(l);
		}
	}
};

// >>>>>>>>>>>>> Solve forward declaration here >>>>>>>>>>>>>
inline Vector3f::Vector3f(const Point3f &p) {
	simdData = _mm_set_ps(0.0f, p.z, p.y, p.x);
}
inline Vector3f::Vector3f(const Normal3f &n) {
	simdData = n.simdData;
}

#undef SIMD_ARRAY_ACCESS
#undef SIMD_ARRAY_READ

#endif // GM_COMPACT_3D_OBJECT

inline Vector3f operator*(float s, const Vector3f & v) {
	return v * s;
}
inline Point3f operator + (const Vector3f & v, const Point3f & p) {
	return p + v;
}
inline float operator * (const Vector3f &v, const Normal3f &n) {
	return n*v;
}
// <<<<<<<<<<<<< Solve forward declaration above <<<<<<<<<<<<

inline Vector3f abs(const Vector3f &v) {
	return Vector3f{ std::abs(v.x), std::abs(v.y), std::abs(v.z) };
	__m128i intBits = _mm_cvttps_epi32(v.simdData);
	__m128i iMask = _mm_set1_epi32(0x7fff'7fff);
	__m128i absIntBits = _mm_and_si128(intBits, iMask);
	__m128 res = _mm_cvtepi32_ps(absIntBits);
	return Vector3f{ res };

	//static const __m128 mask = _mm_cvtepi32_ps(_mm_set1_epi32(0x7fff'ffff));
	//return Vector3f{_mm_and_ps(v.simdData, mask)};
}
inline float dot(const Vector3f &v, const Vector3f &u) { return v*u; }
inline float dot(const Vector3f &v, const Normal3f &n) { return n*v; }
inline float dot(const Normal3f &n, const Vector3f &v) { return n*v; }
inline float dot(const Normal3f &n, const Normal3f &m) { return n*m; }
inline Vector3f cross(const Vector3f &v, const Vector3f &u) { return v^u; }
inline float dist(const Point3f &p, const Point3f &q) { return p.dist(q); }
inline float absDot(const Vector3f &v, const Vector3f &u) { return std::fabs(v*u); }
inline float absDot(const Vector3f &v, const Normal3f &n) { return std::fabs(n*v); }
inline float absDot(const Normal3f &n, const Vector3f &v) { return std::fabs(n*v); }
inline float absDot(const Normal3f &n, const Normal3f &m) { return std::fabs(n*m); }
inline float max(const Vector3f &v) {
	return PBRT_MAX(v.x, PBRT_MAX(v.y, v.z));
}
inline float min(const Vector3f &v) {
	return PBRT_MIN(v.x, PBRT_MIN(v.y, v.z));
}
inline uint8_t argmax(const Vector3f &v) {
	uint8_t res = 0;
	if (v[res] < v[1]) res = 1;
	if (v[res] < v[2]) res = 2;
	return res;
}
inline uint8_t argmin(const Vector3f &v) {
	uint8_t res = 0;
	if (v[res] > v[1]) res = 1;
	if (v[res] > v[2]) res = 2;
	return res;
}

inline Point3f permute(const Point3f &p, int ix, int iy, int iz) { return Point3f{ p[ix], p[iy], p[iz] }; }
inline Vector3f permute(const Vector3f &v, int ix, int iy, int iz) { return Vector3f{ v[ix], v[iy], v[iz] }; }
inline void coordSystem(const Vector3f &v0, Vector3f *v1, Vector3f *v2) {
	if (v0.length() == 0) {
		*v1 = Vector3f::XBASE; *v2 = Vector3f::YBASE;
	} else {
		int dim = argmax(abs(v0));

	}
}

inline Point3f lerp(const Point3f &p, const Point3f &q, float t) {
	return p + (q - p)*t;
}


// TODO: use align here.
class Vector4f {
public: 
	union {
		struct {
			float x, y, z, w;
		};
		__m128 simdData;
	};
	
	// >>>>>>>>>>> Constructors >>>>>>>>>>>
	Vector4f() {
		simdData = _mm_setzero_ps();
		assert((((uint64_t)(this)) & 0xF) == 0);
	}
	explicit Vector4f(const __m128 & _intrin) {
		simdData = _intrin;
	}
	
	Vector4f(const Vector4f & copy) { simdData = copy.simdData; }
	
	Vector4f(float _x, float _y, float _z, float _w) {
		simdData = _mm_set_ps(_w, _z, _y, _x);
	}
	
	explicit Vector4f(const Vector3f &v3, float _w = 0.0f) {
		simdData = _mm_set_ps(_w, v3.z, v3.y, v3.x);
	}

	explicit Vector4f(const Point3f &p, float _w = 1.0f) {
		simdData = _mm_set_ps(_w, p.z, p.y, p.x);
	}
	
	// <<<<<<<<<<< Constructors <<<<<<<<<<<

	float & operator [] (int i) { return (&x)[i]; }
	float   operator [] (int i) const { return (&x)[i]; }

	// conversion to Vec3, Pt3 and Norm3
	explicit operator Vector3f () const { Vector3f res{ simdData }; res.dummy = 0.0f; return res; }
	explicit operator Point3f  () const {
		Point3f res{ simdData };
		if (res.dummy != 1.0f && res.dummy != 0.0f)
			res.simdData = _mm_div_ps(res.simdData, _mm_set_ps1(res.dummy));
		else
			res.dummy = 1.0f;
		return res;
	}
	explicit operator Normal3f () const { Normal3f res{ simdData }; res.dummy = 0.0f; return res; }

	// >>>>>>>>>>> Arithmetic >>>>>>>>>>>>>
	Vector4f operator + (const Vector4f &rhs) const {
		return Vector4f{ _mm_add_ps(simdData, rhs.simdData) };
	}
	Vector4f operator - (const Vector4f &rhs) const {
		return Vector4f{ _mm_sub_ps(simdData, rhs.simdData) };
	}
	Vector4f operator * (float s) const {
		__m128 multiplier = _mm_set_ps1(s);
		return Vector4f{ _mm_mul_ps(simdData, multiplier) };
	}
	Vector4f operator / (float s) const {
		assert(s != 0.0f);
		__m128 inverse = _mm_set_ps1(1.0f / s);
		return Vector4f{ _mm_mul_ps(simdData, inverse) };
	}

	Vector4f & operator += (const Vector4f & rhs) {
		//	x += rhs.x; y += rhs.y; z += rhs.z; w += rhs.w;
		simdData = _mm_add_ps(simdData, rhs.simdData);
		return *this;
	}
	Vector4f & operator -= (const Vector4f & rhs) {
		//x -= rhs.x; y -= rhs.y; z -= rhs.z; w -= rhs.w;
		simdData = _mm_sub_ps(simdData, rhs.simdData);
		return *this;
	}
	Vector4f & operator *= (float s) {
		__m128 multiplier = _mm_set_ps1(s);
		simdData = _mm_mul_ps(simdData, multiplier);
		return *this;
	}
	Vector4f & operator /= (float s) {
		assert(s != 0.0f);
		__m128 inverse = _mm_set_ps1(1.0f / s);
		simdData = _mm_mul_ps(simdData, inverse);
		return *this;
	}

	// unary
	Vector4f operator - () const {
		static const __m128i intNegator = _mm_set1_epi32(0x80000000);
		Vector4f res;
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
	Vector4f homogenized() const {
		if (w == 1.0f || w == 0.0f) return *this;
		else return this->operator*(1.0f / w);
	}

	float lengthSquared() const {
		__m128 dot = _mm_dp_ps(simdData, simdData, 0xff);
		return _mm_cvtss_f32(dot);
	}

	const static Vector4f XBASE;
	const static Vector4f YBASE;
	const static Vector4f ZBASE;
	const static Vector4f WBASE;
};



inline Vector3f reflect(const Vector3f &normal, const Vector3f &incident) {
	// 1. remove from _incident_ the component parallel to _normal_.
	auto parallel = (incident*normal) *normal/normal.lengthSquared();
	auto perpendicular = incident - parallel;
	float check = perpendicular * normal;
	assert(std::abs(check) < 1e-6);
	return incident - 2 * perpendicular;
}

} // namespace hcm


std::ostream & operator << (std::ostream &, const hcm::Point3f & p);
std::ostream & operator << (std::ostream &, const hcm::Vector3f & v);
std::ostream & operator << (std::ostream &, const hcm::Normal3f & n);

#endif //_GLWHEEL_MATH_GEOMETRY
#ifndef _GLWHEEL_MATH_MISC_H
#define _GLWHEEL_MATH_MISC_H

#include <cstdint>
#include <cmath> // _isinf_
#include <limits>
// #include <cstring> // _memcpy_, replaced by _reinterpret_cast_

namespace hcm {

// On windows platform, the macro `PI` is not defined in cmath.
// I know that there's a macro whose definition will trigger the definition of `PI`
//   but I really preferred the global constant approach..

static constexpr float PI = 3.14159265358979f;
static constexpr double PI_D = 3.1415926535897932384626;
static constexpr float MAX_FLOAT = std::numeric_limits<float>::max();
static constexpr float INFTY = std::numeric_limits<float>::infinity();

static inline constexpr float TO_RADIAN(float degree) {
	return degree * PI / 180.0f;
}

static inline constexpr float TO_DEGREE(float radian) {
	return radian * 180.0f / PI;
}


static inline float bitsToFloat(uint32_t ui) {
	return * reinterpret_cast<float *>(&ui);
	// float f;
	// memcpy(&f, &ui, sizeof(uint32_t));
	// return f;
}

static inline uint32_t floatToBits(float f) {
	return * reinterpret_cast<uint32_t *>(&f);
	// uint32_t ui;
	// memcpy(&ui, &f, sizeof(uint32_t));
	// return ui;
}

static inline double bitsToDouble(uint64_t uil) {
	return * reinterpret_cast<double *>(&uil);
	// double d;
	// memcpy(&d, &uil, sizeof(double));
	// return d;
}

static inline uint64_t doubleToBits(double d) {
	return * reinterpret_cast<uint64_t *>(&d);
	// uint64_t uil;
	// memcpy(&uil,&d, sizeof(double));
	// return uil;
}

static inline float nextFloatUp(float f) {
	// The IEEE 754 standard specifies that the float number with bit string
	//   x 1111 1111 0000 0000 0000 0000 0000 000
	//   is infinity.
	// If x == 0, then it's a positive infinity.
	if (f == INFTY) return f;
	//if (std::isinf(f) && f > 0.f) return f;
	else {
		if (f == -0.0f) f = 0.0f;
		uint32_t ui = floatToBits(f);
		if (f >= 0) ++ui;
		else		--ui;
		return bitsToFloat(ui);
	}
}

static inline float nextFloatDown(float f) {
	if (f == -INFTY) return f;
	//if (std::isinf(f) && f < 0.f) return f;
	else {
		if (f == 0.0f) f = -0.0f;
		uint32_t ui = floatToBits(f);
		if (f > 0)  --ui;
		else		++ui;
		return bitsToFloat(ui);
	}
}

static constexpr float MACHINE_EPSILON = std::numeric_limits<float>::epsilon() * 0.5f;

static constexpr float gamma(int n) {
	return n*MACHINE_EPSILON / (1 - n*MACHINE_EPSILON);
}

// template <typename T>
// static constexpr T max(T x, T y) {
// 	return (x < y) ? y : x;
// }

// template <typename T>
// static constexpr T min(T x, T y) {
// 	return (x < y) ? x : y;
// }


// Solves the quadratic equation ax^2+bx+c=0.
// If the discriminant is less than 0, false will return.
// It is guaranteed that x0 < x1 if function returns true. (<- maybe not a good idea)
// Don't know yet what to do if a == 0.	
static inline bool solveQuadratic(float a, float b, float c, float *x0, float *x1) {
	double discriminant = double(b)*double(b) - 4 * double(a)*double(c);
	if (discriminant < 0) return false;
	else {
		// x = (b \pm sqrt(delta)) / (2*a).
		*x0 = static_cast<float>((-b + sqrt(discriminant)) / (2 * a));
		*x1 = static_cast<float>((-b - sqrt(discriminant)) / (2 * a));
		if (*x0 > *x1) {
			float tmp = *x0;
			*x0 = *x1;
			*x1 = tmp;
		}
		return true;
	}
}

static inline float clamp(float v, float min, float max) {
	return v > max ? max : (v < min ? min : v);
}

} // namespace hcm

#endif // _GLWHEEL_HCM_H

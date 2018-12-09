#ifndef _GLWHEEL_MATH_EFLOAT_H
#define _GLWHEEL_MATH_EFLOAT_H
#include <cmath>
#include "misc.hpp"

namespace hcm {

class EFloat {

	float v;
	float err; // absolute error.
#ifndef NDEBUG
	double d;
#endif

public:
	EFloat() :v{ 0.0f }, err{ 0.0f }
#ifndef NDEBUG
		, d{ 0.0 }
#endif
	{}

	EFloat(float _v, float _e = 0.0f) : v{ _v }, err{ _e }
#ifndef NDEBUG
		, d{ _v }
#endif
	{}

	EFloat operator+(const EFloat &rhs) const {
		EFloat res{ v + rhs.v };
#ifndef NDEBUG
		res.d = d + rhs.d;
#endif
		res.err = ::std::abs(v + rhs.v) * gamma(1) + (err + rhs.err)*(1 + gamma(1));
		return res;
	}

	EFloat operator-(const EFloat &rhs) const {
		EFloat res{ v - rhs.v };
#ifndef NDEBUG
		res.d = d - rhs.d;
#endif
		res.err = ::std::abs(v - rhs.v)*gamma(1) + (err + rhs.err)*(1 + gamma(1));
		return res;
	}

	EFloat operator*(const EFloat &rhs) const {
		EFloat res{ v*rhs.v };
#ifndef NDEBUG
		res.d = d * rhs.d;
#endif
		res.err = ::std::abs(v*rhs.v)*gamma(1) +
			(::std::abs(v)*rhs.err + ::std::abs(rhs.v)*err + err*rhs.err)*(1 + gamma(1));
		return res;
	}

	EFloat operator/(const EFloat &rhs) const {
		EFloat res{ v / rhs.v };
#ifndef NDEBUG
		res.d = d / rhs.d;
#endif
		float tmp = rhs.err / ::std::abs(rhs.v);
		res.err = (err + (::std::abs(v) + err)*(gamma(1) + tmp + 2 * tmp*tmp))
			/ (::std::abs(rhs.v) - rhs.err);
		return res;
	}

	EFloat sqrt() const {
		EFloat res;
#ifndef NDEBUG
		res.d = ::std::sqrt(d);
#endif
		res.v = ::std::sqrt(v);
		if (::std::isnan(res.v)) {
			res.err = NAN;
		}
		else {
			res.err = gamma(1)*res.v + (err*(1 + gamma(1))) / (2 * ::std::sqrt(v - err));
		}
		return res;
	}

	EFloat operator-() const {
		EFloat res{ -v, err };
#ifndef NDEBUG
		res.d = -d;
#endif
		return res;
	}

	float absoluteError() const { return err; }
	explicit operator float() const { return v; }

	// The upper bound and lower bound are designed to be conservative
	float upperBound() const { return nextFloatUp(v + err); }
	float lowerBound() const { return nextFloatDown(v - err); }

#ifndef NDEBUG
	float relativeError() const { return (v - d) / d; }
	float preciseValue() const { return d; }
	explicit operator double() const { return d; }
#endif
};

static inline EFloat operator+(float f, const EFloat & ef) { return EFloat{ f, 0.0f } + ef; }
static inline EFloat operator+(const EFloat & ef, float f) { return ef + EFloat{ f, 0.0f }; }
static inline EFloat operator-(float f, const EFloat & ef) { return EFloat{ f, 0.0f } - ef; }
static inline EFloat operator-(const EFloat & ef, float f) { return ef - EFloat{ f, 0.0f }; }
static inline EFloat operator*(float f, const EFloat & ef) { return EFloat{ f, 0.0f } * ef; }
static inline EFloat operator*(const EFloat & ef, float f) { return ef * EFloat{ f, 0.0f }; }
static inline EFloat operator/(float f, const EFloat & ef) { return EFloat{ f, 0.0f } / ef; }
static inline EFloat operator/(const EFloat & ef, float f) { return ef / EFloat{ f, 0.0f }; }



inline bool solveQuadratic(const EFloat &a, const EFloat &b, const EFloat &c,
	EFloat *x0, EFloat *x1) {

	EFloat discriminant = b * b - 4 * a*c;
	if (float(discriminant) < 0) {
		return false;
	}
	else {
		EFloat tmp = discriminant.sqrt();
		*x0 = (-b + tmp) / (2 * a);
		*x1 = (-b - tmp) / (2 * a);
		if (float(*x0) > float(*x1)) {
			EFloat tmp = *x0;
			*x0 = *x1;
			*x1 = tmp;
		}
		return true;
	}
}

namespace std {
	static inline EFloat sqrt(const EFloat &ef) { return ef.sqrt(); }
}

} // namespace hcm

#endif // _GLWHEEL_MATH_EFLOAT
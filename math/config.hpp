#ifndef _GLWHEEL_MATH_CONFIG
#define _GLWHEEL_MATH_CONFIG

// The following functions are provided:
//   dot(v, v), dot(v, n), dot(n, v), dot(n, n)
//   cross(v, v)
// There are also operator-overloaded versions that can be helpful but confusing
// Comment the following MACRO to disable operator-overloaded versions
#define GLWHEEL_MATH_GEOMETRY_USE_OPOV_DC

// Define the following macro to enable <pt3 * float> arithmetic
// #define GLWHEEL_MATH_GEOMETRY_USE_PXF

#endif // _GLWHEEL_MATH_CONFIG
#include <iostream>

#include "misc.hpp"
#include "geometry.hpp"
#include "geometry_compact.hpp"
#include "geometry2d.hpp"
#include "transform.hpp"
#include "quaternion.hpp"


std::ostream & operator<<(std::ostream &os, const hcm::pt3 & p)
{
	os << '[' << p.x << ' ' << p.y << ' ' << p.z << ']';
	return os;
}

std::ostream & operator<<(std::ostream &os, const hcm::vec3 & v)
{
	os << '[' << v.x << ' ' << v.y << ' ' << v.z << ']';
	return os;
}

std::ostream & operator<<(std::ostream &os, const hcm::normal3 & n)
{
	os << '[' << n.x << ' ' << n.y << ' ' << n.z << ']';
	return os;
}

std::ostream & operator << (std::ostream &os, const hcm::quat &q) {
	std::ios::sync_with_stdio();
	printf("[%5.2fi %5.2fj %5.2fk + %5.2f]", q.x, q.y, q.z, q.w);
	return os;
}

std::ostream & operator<<(std::ostream &os, const hcm::mat3 & m)
{
	//const vec3 &a = m.columns[0], &b = m.columns[1], &c = m.columns[2];
	std::ios::sync_with_stdio(true);
	printf("\n| %5.2f %5.2f %5.2f |\n", m.c0.x, m.c1.x, m.c2.x);
	printf("| %5.2f %5.2f %5.2f |\n", m.c0.y, m.c1.y, m.c2.y);
	printf("| %5.2f %5.2f %5.2f |\n", m.c0.z, m.c1.z, m.c2.z);
	return os;
}

std::ostream & operator<<(std::ostream &os, const hcm::mat4 & m)
{
    //const vec3 &a = m.columns[0], &b = m.columns[1], &c = m.columns[2];
    std::ios::sync_with_stdio(true);
    printf("\n| %5.2f %5.2f %5.2f %5.2f |\n", m.c0.x, m.c1.x, m.c2.x, m.c3.x);
    printf("| %5.2f %5.2f %5.2f %5.2f |\n", m.c0.y, m.c1.y, m.c2.y, m.c3.y);
    printf("| %5.2f %5.2f %5.2f %5.2f |\n", m.c0.z, m.c1.z, m.c2.z, m.c3.z);
    printf("| %5.2f %5.2f %5.2f %5.2f |\n", m.c0.w, m.c1.w, m.c2.w, m.c3.w);
    return os;
}
                                                                         
            
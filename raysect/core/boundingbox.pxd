# cython: language_level=3

# Copyright (c) 2014-2025, Dr Alex Meakins, Raysect Project
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright notice,
#        this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#
#     3. Neither the name of the Raysect Project nor the names of its
#        contributors may be used to endorse or promote products derived from
#        this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from raysect.core.boundingsphere cimport BoundingSphere3D
from raysect.core.math cimport Point3D, Point2D
from raysect.core.ray cimport Ray
from libc.math cimport INFINITY
cimport cython


cdef class BoundingBox3D:

    cdef Point3D lower
    cdef Point3D upper

    cdef Point3D get_centre(self)
    cpdef bint hit(self, Ray ray)
    cpdef tuple full_intersection(self, Ray ray)
    cdef bint intersect(self, Ray ray, double *front_intersection, double *back_intersection)
    cdef void _slab(self, double origin, double direction, double lower, double upper, double *front_intersection, double *back_intersection) nogil
    cpdef bint contains(self, Point3D point)
    cpdef object union(self, BoundingBox3D box)
    cpdef object extend(self, Point3D point, double padding=*)
    cpdef double surface_area(self)
    cpdef double volume(self)
    cpdef list vertices(self)
    cpdef double extent(self, int axis) except -1
    cpdef int largest_axis(self)
    cpdef double largest_extent(self)
    cpdef object pad(self, double padding)
    cpdef object pad_axis(self, int axis, double padding)
    cpdef BoundingSphere3D enclosing_sphere(self)


@cython.cdivision(True)
cdef inline void _slab_inline(double origin, double direction, double lower, double upper, double *front_intersection, double *back_intersection) noexcept nogil:

    cdef double reciprocal, tmin, tmax

    if direction != 0.0:

        reciprocal = 1.0 / direction

        if direction > 0.0:
            tmin = (lower - origin) * reciprocal
            tmax = (upper - origin) * reciprocal
        else:
            tmin = (upper - origin) * reciprocal
            tmax = (lower - origin) * reciprocal

    else:

        if origin < lower:
            tmin = -INFINITY
            tmax = -INFINITY
        elif origin > upper:
            tmin = INFINITY
            tmax = INFINITY
        else:
            tmin = -INFINITY
            tmax = INFINITY

    if tmin > front_intersection[0]:
        front_intersection[0] = tmin

    if tmax < back_intersection[0]:
        back_intersection[0] = tmax


cdef inline bint _bbox_intersect_inline(BoundingBox3D box, Ray ray, double *front_intersection, double *back_intersection) noexcept nogil:

    front_intersection[0] = -INFINITY
    back_intersection[0] = INFINITY

    _slab_inline(ray.origin.x, ray.direction.x, box.lower.x, box.upper.x, front_intersection, back_intersection)
    _slab_inline(ray.origin.y, ray.direction.y, box.lower.y, box.upper.y, front_intersection, back_intersection)
    _slab_inline(ray.origin.z, ray.direction.z, box.lower.z, box.upper.z, front_intersection, back_intersection)

    if front_intersection[0] > back_intersection[0]:
        return False

    if (front_intersection[0] < 0.0) and (back_intersection[0] < 0.0):
        return False
    return True


cdef inline bint _bbox_hit_inline(BoundingBox3D box, Ray ray) noexcept nogil:
    cdef double front_intersection, back_intersection
    return _bbox_intersect_inline(box, ray, &front_intersection, &back_intersection)


cdef inline bint _bbox_contains_inline(BoundingBox3D box, Point3D point) noexcept nogil:
    if (point.x < box.lower.x) or (point.x > box.upper.x):
        return False
    if (point.y < box.lower.y) or (point.y > box.upper.y):
        return False
    if (point.z < box.lower.z) or (point.z > box.upper.z):
        return False
    return True


cdef inline BoundingBox3D new_boundingbox3d(Point3D lower, Point3D upper):
    """
    BoundingBox3D factory function.

    Creates a new BoundingBox3D object with less overhead than the equivalent
    Python call. This function is callable from cython only.
    """

    cdef BoundingBox3D v
    v = BoundingBox3D.__new__(BoundingBox3D)
    v.lower = lower
    v.upper = upper
    return v


cdef class BoundingBox2D:

    cdef Point2D lower
    cdef Point2D upper

    cpdef bint contains(self, Point2D point)
    cpdef object union(self, BoundingBox2D box)
    cpdef object extend(self, Point2D point, double padding=*)
    cpdef double surface_area(self)
    cpdef list vertices(self)
    cpdef double extent(self, int axis) except -1
    cpdef int largest_axis(self)
    cpdef double largest_extent(self)
    cpdef object pad(self, double padding)
    cpdef object pad_axis(self, int axis, double padding)


cdef inline BoundingBox2D new_boundingbox2d(Point2D lower, Point2D upper):
    """
    BoundingBox2D factory function.

    Creates a new BoundingBox2D object with less overhead than the equivalent
    Python call. This function is callable from cython only.
    """

    cdef BoundingBox2D b
    b = BoundingBox2D.__new__(BoundingBox2D)
    b.lower = lower
    b.upper = upper
    return b

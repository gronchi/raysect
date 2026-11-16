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

from raysect.core cimport new_point3d, new_vector3d, Normal3D, new_normal3d, AffineMatrix3D, Material, new_intersection, BoundingBox3D
from raysect.core.math.affinematrix cimport _transform_ray_inline
from libc.math cimport fabs, INFINITY
cimport cython

# bounding box is padded by a small amount to avoid numerical accuracy issues
cdef const double BOX_PADDING = 1e-9

# additional ray distance to avoid re-hitting the same surface point
cdef const double EPSILON = 1e-9

cdef enum:

    # slab face enumeration
    NO_FACE = -1
    LOWER_FACE = 0
    UPPER_FACE = 1

    # axis enumeration
    NO_AXIS = -1
    X_AXIS = 0
    Y_AXIS = 1
    Z_AXIS = 2


@cython.cdivision(True)
cdef inline void _box_slab_inline(int axis, double origin, double direction, double lower, double upper, double *near_intersection, double *far_intersection, int *near_face, int *far_face, int *near_axis, int *far_axis) noexcept nogil:

    cdef:
        double reciprocal, tmin, tmax
        int fmin, fmax

    if direction != 0.0:

        # calculate intersections with slab planes
        reciprocal = 1.0 / direction

        if direction > 0:

            # calculate length along ray path of intersections
            tmin = (lower - origin) * reciprocal
            tmax = (upper - origin) * reciprocal

            fmin = LOWER_FACE
            fmax = UPPER_FACE

        else:

            # calculate length along ray path of intersections
            tmin = (upper - origin) * reciprocal
            tmax = (lower - origin) * reciprocal

            fmin = UPPER_FACE
            fmax = LOWER_FACE

    else:

        # ray is not propagating along this axis so limits are infinite
        if origin < lower:

            tmin = -INFINITY
            tmax = -INFINITY

        elif origin > upper:

            tmin = INFINITY
            tmax = INFINITY

        else:

            tmin = -INFINITY
            tmax = INFINITY

        fmin = NO_FACE
        fmax = NO_FACE

    # calculate slab intersection overlap, store closest dimension and intersected face
    if tmin > near_intersection[0]:
        near_intersection[0] = tmin
        near_face[0] = fmin
        near_axis[0] = axis

    if tmax < far_intersection[0]:
        far_intersection[0] = tmax
        far_face[0] = fmax
        far_axis[0] = axis


cdef inline double _box_interior_offset_inline(double hit_point, double lower, double upper) noexcept nogil:
    if fabs(hit_point - lower) < EPSILON:
        return EPSILON
    elif fabs(hit_point - upper) < EPSILON:
        return -EPSILON
    return 0.0


cdef inline Intersection _generate_box_intersection_inline(
    Box self, Ray ray,
    double ox, double oy, double oz,
    double dx, double dy, double dz,
    double ray_distance, int face, int axis
):
    cdef Point3D hit_point, inside_point, outside_point
    cdef Normal3D normal
    cdef double hx, hy, hz, nx = 0.0, ny = 0.0, nz = 0.0
    cdef bint exiting

    hx = ox + ray_distance * dx
    hy = oy + ray_distance * dy
    hz = oz + ray_distance * dz
    hit_point = new_point3d(hx, hy, hz)

    if axis == X_AXIS:
        nx = -1.0 if face == LOWER_FACE else 1.0
    elif axis == Y_AXIS:
        ny = -1.0 if face == LOWER_FACE else 1.0
    elif axis == Z_AXIS:
        nz = -1.0 if face == LOWER_FACE else 1.0
    normal = new_normal3d(nx, ny, nz)

    inside_point = new_point3d(hx + _box_interior_offset_inline(hx, self._lower.x, self._upper.x),
                               hy + _box_interior_offset_inline(hy, self._lower.y, self._upper.y),
                               hz + _box_interior_offset_inline(hz, self._lower.z, self._upper.z))

    outside_point = new_point3d(hx + EPSILON * nx,
                                hy + EPSILON * ny,
                                hz + EPSILON * nz)

    exiting = (dx * nx + dy * ny + dz * nz) >= 0.0

    return new_intersection(ray, ray_distance, self, hit_point, inside_point, outside_point,
                            normal, exiting, self._root_transform_inverse, self._root_transform)


cdef class Box(Primitive):
    """
    A box primitive.

    The box is defined by lower and upper points in the local co-ordinate
    system.

    :param Point3D lower: Lower point of the box (default = Point3D(-0.5, -0.5, -0.5)).
    :param Point3D upper: Upper point of the box (default = Point3D(0.5, 0.5, 0.5)).
    :param Node parent: Scene-graph parent node or None (default = None).
    :param AffineMatrix3D transform: An AffineMatrix3D defining the local co-ordinate system relative to the scene-graph parent (default = identity matrix).
    :param Material material: A Material object defining the box's material (default = None).
    :param str name: A string specifying a user-friendly name for the box (default = "").

    .. code-block:: pycon

        >>> from raysect.core import Point3D, translate
        >>> from raysect.primitive import Box
        >>> from raysect.optical import World
        >>> from raysect.optical.material import UniformSurfaceEmitter
        >>> from raysect.optical.library.spectra.colours import red
        >>>
        >>> world = World()
        >>>
        >>> cube = Box(Point3D(0,0,0), Point3D(1,1,1), parent=world, transform=translate(0, 1, 0),
                       material=UniformSurfaceEmitter(red), name="red cube")
    """

    def __init__(self, Point3D lower=None, Point3D upper=None, object parent=None, AffineMatrix3D transform=None, Material material=None, str name=None):

        super().__init__(parent, transform, material, name)

        if lower is not None and upper is not None:

            if lower.x > upper.x or lower.y > upper.y or lower.z > upper.z:
                raise ValueError("The lower point coordinates must be less than or equal to the upper point coordinates.")
            self._lower = lower
            self._upper = upper

        elif lower is None and upper is None:

            # default to unit box centred on the axis
            self._lower = new_point3d(-0.5, -0.5, -0.5)
            self._upper = new_point3d(0.5, 0.5, 0.5)

        else:
            raise ValueError("Lower and upper points must both be defined.")

        # initialise next intersection caching and control attributes
        self._further_intersection = False
        self._next_t = 0.0
        self._cached_origin = None
        self._cached_direction = None
        self._cached_ray = None
        self._cached_face = NO_FACE
        self._cached_axis = NO_AXIS

    @property
    def lower(self):
        """
        Lower 3D coordinate of the box in primitive's local coordinates.

        :rtype: Point3D
        """

        return self._lower

    @lower.setter
    def lower(self, Point3D value not None):
        if value.x > self._upper.x or value.y > self._upper.y or value.z > self._upper.z:
            raise ValueError("The lower point coordinates must be less than or equal to the upper point coordinates.")
        self._lower = value

        # the next intersection cache has been invalidated by the geometry change
        self._further_intersection = False

        # any geometry caching in the root node is now invalid, inform root
        self.notify_geometry_change()

    @property
    def upper(self):
        """
        Upper 3D coordinate of the box in primitive's local coordinates.

        :rtype: Point3D
        """

        return self._upper

    @upper.setter
    def upper(self, Point3D value not None):
        if self._lower.x > value.x or self._lower.y > value.y or self._lower.z > value.z:
            raise ValueError("The upper point coordinates must be greater than or equal to the lower point coordinates.")
        self._upper = value

        # the next intersection cache has been invalidated by the geometry change
        self._further_intersection = False

        # any geometry caching in the root node is now invalid, inform root
        self.notify_geometry_change()

    cpdef Intersection hit(self, Ray ray):

        cdef:
            double ox, oy, oz, dx, dy, dz
            double near_intersection, far_intersection, closest_intersection
            int near_face, far_face, closest_face
            int near_axis, far_axis, closest_axis

        # invalidate next intersection cache
        self._further_intersection = False

        # convert ray origin and direction to local space in C registers
        _transform_ray_inline(
            ray.origin.x, ray.origin.y, ray.origin.z,
            ray.direction.x, ray.direction.y, ray.direction.z,
            self._root_transform_inverse,
            &ox, &oy, &oz, &dx, &dy, &dz
        )

        # set initial ray-slab intersection search range
        near_intersection = -INFINITY
        far_intersection = INFINITY

        # initially there are no face intersections, set to invalid values to aid debugging
        near_face = NO_FACE
        near_axis = NO_AXIS
        far_face = NO_FACE
        far_axis = NO_AXIS

        # evaluate ray-slab intersection for x, y and z dimensions and update the intersection positions
        _box_slab_inline(X_AXIS, ox, dx, self._lower.x, self._upper.x, &near_intersection, &far_intersection, &near_face, &far_face, &near_axis, &far_axis)
        _box_slab_inline(Y_AXIS, oy, dy, self._lower.y, self._upper.y, &near_intersection, &far_intersection, &near_face, &far_face, &near_axis, &far_axis)
        _box_slab_inline(Z_AXIS, oz, dz, self._lower.z, self._upper.z, &near_intersection, &far_intersection, &near_face, &far_face, &near_axis, &far_axis)

        # does ray intersect box?
        if near_intersection > far_intersection:
            return None

        # are there any intersections inside the ray search range?
        if near_intersection > ray.max_distance or far_intersection < 0.0:
            return None

        # identify closest intersection
        if near_intersection >= 0.0:

            closest_intersection = near_intersection
            closest_face = near_face
            closest_axis = near_axis

            if far_intersection <= ray.max_distance:

                self._further_intersection = True
                self._next_t = far_intersection
                if self._cached_origin is None:
                    self._cached_origin = new_point3d(ox, oy, oz)
                    self._cached_direction = new_vector3d(dx, dy, dz)
                else:
                    self._cached_origin.x = ox
                    self._cached_origin.y = oy
                    self._cached_origin.z = oz
                    self._cached_direction.x = dx
                    self._cached_direction.y = dy
                    self._cached_direction.z = dz
                self._cached_ray = ray
                self._cached_face = far_face
                self._cached_axis = far_axis

        elif far_intersection <= ray.max_distance:

            closest_intersection = far_intersection
            closest_face = far_face
            closest_axis = far_axis

        else:
            return None

        return _generate_box_intersection_inline(self, ray, ox, oy, oz, dx, dy, dz, closest_intersection, closest_face, closest_axis)

    cpdef Intersection next_intersection(self):

        if not self._further_intersection:
            return None

        # this is the 2nd and therefore last intersection
        self._further_intersection = False

        return _generate_box_intersection_inline(
            self, self._cached_ray,
            self._cached_origin.x, self._cached_origin.y, self._cached_origin.z,
            self._cached_direction.x, self._cached_direction.y, self._cached_direction.z,
            self._next_t, self._cached_face, self._cached_axis
        )

    cdef void _slab(self, int axis, double origin, double direction, double lower, double upper, double *near_intersection, double *far_intersection, int *near_face, int *far_face, int *near_axis, int *far_axis) nogil:
        _box_slab_inline(axis, origin, direction, lower, upper, near_intersection, far_intersection, near_face, far_face, near_axis, far_axis)

    cdef Intersection _generate_intersection(self, Ray ray, Point3D origin, Vector3D direction, double ray_distance, int face, int axis):
        return _generate_box_intersection_inline(self, ray, origin.x, origin.y, origin.z, direction.x, direction.y, direction.z, ray_distance, face, axis)

    cdef double _interior_offset(self, double hit_point, double lower, double upper) nogil:
        """
        Calculates an interior offset that ensures the inside_point is away from the primitive surface.

        The calculation breaks down for boxes dimensions close to the minimum offset epsilon (~1e-9).
        Users should not be creating objects that approach raysect numerical accuracy limits.
        """
        return _box_interior_offset_inline(hit_point, lower, upper)

    @cython.cdivision(True)
    cpdef bint contains(self, Point3D point) except -1:

        cdef AffineMatrix3D m = self._root_transform_inverse
        cdef double px = point.x, py = point.y, pz = point.z
        cdef double x, y, z, w

        w = m.m[3][0] * px + m.m[3][1] * py + m.m[3][2] * pz + m.m[3][3]
        if w == 1.0:
            x = m.m[0][0] * px + m.m[0][1] * py + m.m[0][2] * pz + m.m[0][3]
            if (x < self._lower.x) or (x > self._upper.x):
                return False
            y = m.m[1][0] * px + m.m[1][1] * py + m.m[1][2] * pz + m.m[1][3]
            if (y < self._lower.y) or (y > self._upper.y):
                return False
            z = m.m[2][0] * px + m.m[2][1] * py + m.m[2][2] * pz + m.m[2][3]
            return (self._lower.z <= z <= self._upper.z)

        if w == 0.0:
            raise ZeroDivisionError("Bad matrix transform, 4th element of homogeneous coordinate is zero.")
        w = 1.0 / w
        x = (m.m[0][0] * px + m.m[0][1] * py + m.m[0][2] * pz + m.m[0][3]) * w
        if (x < self._lower.x) or (x > self._upper.x):
            return False
        y = (m.m[1][0] * px + m.m[1][1] * py + m.m[1][2] * pz + m.m[1][3]) * w
        if (y < self._lower.y) or (y > self._upper.y):
            return False
        z = (m.m[2][0] * px + m.m[2][1] * py + m.m[2][2] * pz + m.m[2][3]) * w
        return (self._lower.z <= z <= self._upper.z)

    cpdef BoundingBox3D bounding_box(self):

        cdef:
            list points
            Point3D point
            BoundingBox3D box

        # generate box vertices
        points = [
            self._lower,
            new_point3d(self._lower.x, self._lower.y, self._upper.z),
            new_point3d(self._lower.x, self._upper.y, self._lower.z),
            new_point3d(self._lower.x, self._upper.y, self._upper.z),
            new_point3d(self._upper.x, self._lower.y, self._lower.z),
            new_point3d(self._upper.x, self._lower.y, self._upper.z),
            new_point3d(self._upper.x, self._upper.y, self._lower.z),
            self._upper
            ]

        # convert points to world space and build an enclosing world space bounding box
        # a small degree of padding is added to avoid potential numerical accuracy issues
        box = BoundingBox3D()
        for point in points:
            box.extend(point.transform(self.to_root()), BOX_PADDING)

        return box

    cpdef object instance(self, object parent=None, AffineMatrix3D transform=None, Material material=None, str name=None):
        return Box(self._lower.copy(), self._upper.copy(), parent, transform, material, name)
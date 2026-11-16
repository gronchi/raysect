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

from raysect.optical cimport new_point3d
from raysect.core.math.affinematrix cimport _mat4_is_identity
from libc.math cimport floor
cimport cython


cdef class VolumeIntegrator:
    """
    Base class for integrators in InhomogeneousVolumeEmitter materials.

    The deriving class must implement the integrate() method.
    """

    cpdef Spectrum integrate(self, Spectrum spectrum, World world, Ray ray, Primitive primitive,
                             InhomogeneousVolumeEmitter material, Point3D start_point, Point3D end_point,
                             AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):
        """
        Performs a customised integration of the emission through a volume emitter.

        This is a virtual method and must be implemented in a sub class.

        :param Spectrum spectrum: Spectrum measured so far along ray path. Add your emission
          to this spectrum, don't override it.
        :param World world: The world scene-graph.
        :param Ray ray: The ray being traced.
        :param Primitive primitive: The geometric primitive to which this material belongs
          (i.e. a cylinder or a mesh).
        :param InhomogeneousVolumeEmitter material: The material whose emission needs to be
          integrated.
        :param Point3D start_point: The start point for integration in world space.
        :param Point3D end_point: The end point for integration in world space.
        :param AffineMatrix3D world_to_primitive: Affine matrix defining the coordinate
          transform from world space to the primitive's local space.
        :param AffineMatrix3D primitive_to_world: Affine matrix defining the coordinate
          transform from the primitive's local space to world space.
        """

        raise NotImplementedError("Virtual method integrate() has not been implemented.")


cdef class NumericalIntegrator(VolumeIntegrator):
    """
    A basic implementation of the trapezium integration scheme for volume emitters.

    :param float step: The step size for numerical integration in metres.
    :param int min_samples: The minimum number of samples to use over integration
      range (default=5).
    """

    def __init__(self, float step, int min_samples=5):
        self._step = step
        self._min_samples = min_samples

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, double value):
        if value <= 0:
            raise ValueError("Numerical integration step size can not be less than or equal to zero")
        self._step = value

    @property
    def min_samples(self):
        return self._min_samples

    @min_samples.setter
    def min_samples(self, int value):
        if value < 2:
            raise ValueError("At least two samples are required to perform the numerical integration.")
        self._min_samples = value

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef Spectrum integrate(self, Spectrum spectrum, World world, Ray ray, Primitive primitive,
                             InhomogeneousVolumeEmitter material, Point3D start_point, Point3D end_point,
                             AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        cdef:
            Point3D start, end, sample_point
            Vector3D integration_direction, ray_direction
            double length, step, t, c, sx, sy, sz, dx, dy, dz
            Spectrum emission, emission_previous, temp
            int intervals, interval, index
            int bins
            double *spec_ptr
            double *em_ptr
            double *em_prev_ptr

        # convert start and end points to local space
        if _mat4_is_identity(world_to_primitive):
            start = start_point
            end = end_point
        else:
            start = start_point.transform(world_to_primitive)
            end = end_point.transform(world_to_primitive)

        # obtain local space ray direction and integration length
        integration_direction = start.vector_to(end)
        length = integration_direction.get_length()

        # nothing to contribute?
        if length == 0.0:
            return spectrum

        integration_direction = integration_direction.normalise()
        ray_direction = integration_direction.neg()

        # calculate number of complete intervals (samples - 1)
        intervals = max(self._min_samples - 1, <int> floor(length / self._step))

        # adjust (increase) step size to absorb any remainder and maintain equal interval spacing
        step = length / intervals

        # create working buffers
        emission = ray.new_spectrum()
        emission_previous = ray.new_spectrum()

        bins = spectrum.bins
        # sample point and sanity check as bounds checking is disabled
        emission_previous = material.emission_function(start, ray_direction, emission_previous, world, ray, primitive, world_to_primitive, primitive_to_world)
        if emission_previous.samples.ndim != 1 or emission_previous.samples.shape[0] != bins:
            raise ValueError("Spectrum returned by emission function has the wrong number of samples.")

        # numerical integration
        c = 0.5 * step
        sx = start.x
        sy = start.y
        sz = start.z
        dx = integration_direction.x
        dy = integration_direction.y
        dz = integration_direction.z
        sample_point = new_point3d(sx, sy, sz)
        spec_ptr = &spectrum.samples_mv[0]

        for interval in range(0, intervals):

            # calculate location of sample point at the top of the interval
            t = (interval + 1) * step
            sample_point.x = sx + t * dx
            sample_point.y = sy + t * dy
            sample_point.z = sz + t * dz

            # sample point and sanity check as bounds checking is disabled
            emission = material.emission_function(sample_point, ray_direction, emission, world, ray, primitive, world_to_primitive, primitive_to_world)
            if emission.samples.ndim != 1 or emission.samples.shape[0] != bins:
                raise ValueError("Spectrum returned by emission function has the wrong number of samples.")

            # trapezium rule integration + fused buffer clear in a single pass
            em_ptr = &emission.samples_mv[0]
            em_prev_ptr = &emission_previous.samples_mv[0]
            for index in range(bins):
                spec_ptr[index] += c * (em_ptr[index] + em_prev_ptr[index])
                em_prev_ptr[index] = 0.0

            # swap buffers (emission_previous was already cleared in the loop above)
            temp = emission_previous
            emission_previous = emission
            emission = temp

        return spectrum

    cdef int _check_dimensions(self, Spectrum spectrum, int bins) except -1:
        if spectrum.samples.ndim != 1 or spectrum.samples.shape[0] != bins:
            raise ValueError("Spectrum returned by emission function has the wrong number of samples.")


cdef class InhomogeneousVolumeEmitter(NullSurface):
    """
    Base class for inhomogeneous volume emitters.

    The integration technique can be changed by the user, but defaults to
    a basic numerical integration scheme.

    The deriving class must implement the emission_function() method.

    :param VolumeIntegrator integrator: Integration object, defaults to
      NumericalIntegrator(step=0.01, min_samples=5).
    """

    def __init__(self, VolumeIntegrator integrator=None):
        super().__init__()
        self.integrator = integrator or NumericalIntegrator(step=0.01, min_samples=5)
        self.importance = 1.0

    cpdef Spectrum evaluate_volume(self, Spectrum spectrum, World world,
                                   Ray ray, Primitive primitive,
                                   Point3D start_point, Point3D end_point,
                                   AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        # pass to volume integrator class
        return self.integrator.integrate(spectrum, world, ray, primitive, self, start_point, end_point,
                                         world_to_primitive, primitive_to_world)

    cpdef Spectrum emission_function(self, Point3D point, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):
        """
        The emission function for the material at a given sample point.

        This is a virtual method and must be implemented in a sub class.

        :param Point3D point: Requested sample point in local coordinates.
        :param Vector3D direction: The emission direction in local coordinates.
        :param Spectrum spectrum: Spectrum measured so far along ray path. Add your emission
          to this spectrum, don't override it.
        :param World world: The world scene-graph.
        :param Ray ray: The ray being traced.
        :param Primitive primitive: The geometric primitive to which this material belongs
          (i.e. a cylinder or a mesh).
        :param AffineMatrix3D world_to_primitive: Affine matrix defining the coordinate
          transform from world space to the primitive's local space.
        :param AffineMatrix3D primitive_to_world: Affine matrix defining the coordinate
          transform from the primitive's local space to world space.
        """

        raise NotImplementedError("Virtual method emission_function() has not been implemented.")


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

from raysect.optical.observer.sampler2d import FullFrameSampler2D
from raysect.optical.observer.pipeline import RGBPipeline2D
from raysect.optical.observer.sampler2d import RGBAdaptiveSampler2D

from raysect.core cimport Point3D, new_point3d, Vector3D, new_vector3d, RectangleSampler3D
from raysect.core.math.random cimport uniform
from raysect.optical cimport Ray, new_ray
from libc.math cimport M_PI as pi, tan, sqrt
from raysect.optical.observer.base cimport Observer2D
cimport cython


cdef class PinholeCamera(Observer2D):
    """
    An observer that models an idealised pinhole camera.

    A simple camera that launches rays from the observer's origin point over a
    specified field of view.

    :param tuple pixels: A tuple of pixel dimensions for the camera (default=(720, 480)).
    :param float fov: The field of view of the camera in degrees (default=45 degrees).
    :param float sensitivity: The sensitivity of each pixel (default=1.0)
    :param FrameSampler2D frame_sampler: The frame sampling strategy, defaults to adaptive
      sampling (i.e. extra samples for noisier pixels).
    :param list pipelines: The list of pipelines that will process the spectrum measured
      at each pixel by the camera (default=RGBPipeline2D()).
    :param kwargs: **kwargs and properties from Observer2D and _ObserverBase.

    .. code-block:: pycon

        >>> from raysect.core import translate
        >>> from raysect.optical import World
        >>> from raysect.optical.observer import PinholeCamera, PowerPipeline2D
        >>>
        >>> power = PowerPipeline2D(display_unsaturated_fraction=0.96, name="Unfiltered")
        >>>
        >>> camera = PinholeCamera((512, 512), parent=world, pipelines=[power])
        >>> camera.transform = translate(0, 0, -3.3)
        >>> camera.pixel_samples = 250
        >>> camera.spectral_bins = 15
        >>>
        >>> camera.observe()
    """

    def __init__(self, pixels, fov=None, sensitivity=None, frame_sampler=None, pipelines=None, parent=None, transform=None, name=None):

        # defaults to an adaptively sampled RGB pipeline
        if not pipelines and not frame_sampler:
            rgb = RGBPipeline2D()
            pipelines = [rgb]
            frame_sampler = RGBAdaptiveSampler2D(rgb)
        else:
            pipelines = pipelines or [RGBPipeline2D()]
            frame_sampler = frame_sampler or FullFrameSampler2D()

        self._fov = 45  # initial value to prevent undefined behaviour when setting pixels for the first time before fov is set

        super().__init__(pixels, frame_sampler, pipelines, parent=parent, transform=transform, name=name)

        # note that the fov property triggers a call to _update_image_geometry()
        self.fov = fov or 45
        self.sensitivity = sensitivity or 1.0

    @property
    def fov(self):
        """
        The field of view of the camera in degrees.

        :rtype: float
        """
        return self._fov

    @fov.setter
    def fov(self, value):
        if value <= 0 or value >= 180:
            raise ValueError("The field-of-view angle must lie in the range (0, 180).")
        self._fov = value
        self._update_image_geometry()

    @property
    def pixels(self):
        """
        Tuple describing the pixel dimensions for this observer (nx, ny), i.e. (512, 512).

        :rtype: tuple
        """
        return self._pixels

    @pixels.setter
    def pixels(self, value):
        pixels = tuple(value)
        if len(pixels) != 2:
            raise ValueError("Pixels must be a 2 element tuple defining the x and y resolution.")
        x, y = pixels
        if x <= 0:
            raise ValueError("Number of x pixels must be greater than 0.")
        if y <= 0:
            raise ValueError("Number of y pixels must be greater than 0.")
        self._pixels = pixels
        self._update_image_geometry()

    @property
    def sensitivity(self):
        """
        The sensitivity applied to each pixel.

        If sensitivity=1.0 all spectral units are in radiance.

        :rtype: float
        """
        return self._sensitivity

    @sensitivity.setter
    def sensitivity(self, value):
        if value <= 0:
            raise ValueError("Sensitivity must be greater than zero.")
        self._sensitivity = value

    cdef object _update_image_geometry(self):

        max_pixels = max(self.pixels)

        if max_pixels > 1:

            # get width of image plane at a distance of 1m from aperture
            image_max_width = 2 * tan(pi / 180 * 0.5 * self._fov)

            # set pixel step size in image plane
            self.image_delta = image_max_width / max_pixels

            self.image_start_x = 0.5 * self.pixels[0] * self.image_delta
            self.image_start_y = 0.5 * self.pixels[1] * self.image_delta

            # rebuild point generator
            self.point_sampler = RectangleSampler3D(self.image_delta, self.image_delta)

        else:
            raise RuntimeError("Number of Pinhole camera Pixels must be > 1.")

    @cython.cdivision(True)
    cpdef list _generate_rays(self, int x, int y, Ray template, int ray_count):

        cdef:
            double pixel_x, pixel_y, width, height, width_offset, height_offset
            double dx, dy, dz, inv_len
            list rays
            Point3D origin
            Vector3D direction
            Ray ray
            int i

        # generate pixel centre offsets
        pixel_x = self.image_start_x - self.image_delta * (x + 0.5)
        pixel_y = self.image_start_y - self.image_delta * (y + 0.5)
        width = self.point_sampler.width
        height = self.point_sampler.height
        width_offset = 0.5 * width
        height_offset = 0.5 * height
        origin = new_point3d(0.0, 0.0, 0.0)

        # assemble rays directly without intermediate Point3D list or unnormalised Vector3D allocations
        rays = []
        for i in range(ray_count):

            dx = (uniform() * width - width_offset) + pixel_x
            dy = (uniform() * height - height_offset) + pixel_y
            inv_len = 1.0 / sqrt(dx * dx + dy * dy + 1.0)
            dz = inv_len
            direction = new_vector3d(dx * inv_len, dy * inv_len, dz)

            ray = new_ray(
                origin, direction,
                template._min_wavelength, template._max_wavelength, template._bins,
                template.max_distance,
                template._extinction_prob, template._extinction_min_depth, template._max_depth,
                template.importance_sampling,
                template._important_path_weight
            )

            # non-physical camera, samples radiance directly
            # projected area weight is normal.incident which simplifies
            # to incident.z here as the normal is (0, 0 ,1)
            rays.append((ray, dz))

        return rays

    cpdef double _pixel_sensitivity(self, int x, int y):
        return self._sensitivity

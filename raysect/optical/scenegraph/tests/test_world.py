# Copyright (c) 2014-2026, Raysect Project

import pickle
import unittest

from raysect.core import Node, Point3D, Vector3D, translate
from raysect.optical import ConstantSF, World
from raysect.optical.material import UniformSurfaceEmitter
from raysect.optical.scenegraph.world import ImportanceError
from raysect.primitive import Sphere


class TestWorldImportance(unittest.TestCase):

    def setUp(self):
        self.world = World()
        self.light = Sphere(
            0.5, parent=self.world, transform=translate(0, 0, 3),
            material=UniformSurfaceEmitter(ConstantSF(1))
        )
        self.origin = Point3D()
        self.forward = Vector3D(0, 0, 1)
        self.right = Vector3D(1, 0, 0)

    def test_prebuilt_world_pickle_preserves_importance(self):
        self.world.build_accelerator()
        self.world.build_importance()
        expected_pdf = self.world.important_direction_pdf(self.origin, self.forward)

        restored = pickle.loads(pickle.dumps(self.world))

        self.assertTrue(restored.has_important_primitives())
        self.assertAlmostEqual(
            restored.important_direction_pdf(self.origin, self.forward), expected_pdf
        )
        for _ in range(10):
            direction = restored.important_direction_sample(self.origin)
            self.assertAlmostEqual(direction.length, 1.0)
            self.assertGreater(restored.important_direction_pdf(self.origin, direction), 0.0)

    def test_prebuilt_empty_world_pickle_preserves_empty_importance(self):
        world = World()
        world.build_importance()
        restored = pickle.loads(pickle.dumps(world))

        self.assertFalse(restored.has_important_primitives())
        self.assertEqual(restored.important_direction_pdf(self.origin, self.forward), 0.0)
        with self.assertRaises(ImportanceError):
            restored.important_direction_sample(self.origin)

    def test_moving_light_updates_importance(self):
        self.world.build_importance()
        self.light.transform = translate(3, 0, 0)

        self.assertEqual(self.world.important_direction_pdf(self.origin, self.forward), 0.0)
        self.assertGreater(self.world.important_direction_pdf(self.origin, self.right), 0.0)

    def test_moving_parent_updates_importance(self):
        parent = Node(parent=self.world)
        self.light.parent = parent
        self.world.build_importance()
        parent.transform = translate(3, 0, -3)

        self.assertEqual(self.world.important_direction_pdf(self.origin, self.forward), 0.0)
        self.assertGreater(self.world.important_direction_pdf(self.origin, self.right), 0.0)

    def test_resizing_light_updates_importance(self):
        direction = Vector3D(1, 0, 3).normalise()
        self.assertEqual(self.world.important_direction_pdf(self.origin, direction), 0.0)

        self.light.radius = 1.5

        self.assertGreater(self.world.important_direction_pdf(self.origin, direction), 0.0)

    def test_removing_light_updates_importance(self):
        self.world.build_importance()
        self.light.parent = None

        self.assertFalse(self.world.has_important_primitives())
        self.assertEqual(self.world.important_direction_pdf(self.origin, self.forward), 0.0)

    def test_reparenting_light_updates_both_worlds(self):
        other_world = World()
        self.world.build_importance()
        other_world.build_importance()

        self.light.parent = other_world

        self.assertFalse(self.world.has_important_primitives())
        self.assertTrue(other_world.has_important_primitives())
        self.assertGreater(other_world.important_direction_pdf(self.origin, self.forward), 0.0)

    def test_material_change_still_updates_importance(self):
        self.world.build_importance()
        self.light.material.importance = 0.0

        self.assertFalse(self.world.has_important_primitives())

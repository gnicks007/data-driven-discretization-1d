# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for equations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest  # pylint: disable=g-bad-import-order
import numpy as np
import tensorflow as tf

from pde_superresolution import equations  # pylint: disable=g-bad-import-order


class GridTest(absltest.TestCase):

  def test_grid(self):
    grid = equations.Grid(3, resample_factor=2, period=60)

    self.assertEqual(grid.solution_num_points, 3)
    np.testing.assert_equal(grid.solution_x, [0, 20, 40])
    np.testing.assert_equal(grid.solution_dx, 20)

    self.assertEqual(grid.reference_num_points, 6)
    np.testing.assert_equal(grid.reference_x, [0, 10, 20, 30, 40, 50])
    np.testing.assert_equal(grid.reference_dx, 10)


class EquationsTest(absltest.TestCase):

  def test_staggered_first_derivative_consistency(self):
    # numpy and tensorflow should give the same result
    y = np.random.RandomState(0).randn(10)
    np_result = equations.staggered_first_derivative(y, dx=1.0)
    with tf.Graph().as_default():
      with tf.compat.v1.Session:
        tf_result = equations.staggered_first_derivative(
            tf.constant(y), dx=1.0).eval()
    np.testing.assert_allclose(np_result, tf_result)


if __name__ == '__main__':
  absltest.main()

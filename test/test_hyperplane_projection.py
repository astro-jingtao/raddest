

import jax
import jax.numpy as jnp
import pytest

from raddest.reg.constructor import project_to_hyperplane

jax.config.update("jax_enable_x64", True)


class TestProjectToHyperplane:

    def test_valid_projection(self):
        # Test with a valid projection
        y = jnp.array([1., 2., 3.])
        A = jnp.array([[1., 1., 1.]])
        b = jnp.array([5.])
        projection = project_to_hyperplane(y, A, b)
        assert jnp.allclose(A.dot(projection), b)

        # Test with a more complex scenario
        y = jnp.array([4., 5., 6.])
        A = jnp.array([[1., 2., 3.], [4., 5., 6.]])
        b = jnp.array([7., 8.])
        projection = project_to_hyperplane(y, A, b)
        assert jnp.allclose(A.dot(projection), b)

        # Test with a 2D hyperplane
        y = jnp.array([1., 1.])
        A = jnp.array([[1., 1.]])
        b = jnp.array([1.])
        projection = project_to_hyperplane(y, A, b)
        assert jnp.allclose(A.dot(projection), b)

        # Test with a different A matrix
        y = jnp.array([1., 2.])
        A = jnp.array([[1., 0.], [0., 1.]])
        b = jnp.array([1., 1.])
        projection = project_to_hyperplane(y, A, b)
        assert jnp.allclose(A.dot(projection), b)

    def test_projection_onto_same_point(self):
        # Test when the point is already on the hyperplane
        y = jnp.array([1., 2., 2.])
        A = jnp.array([[1., 1., 1.]])
        b = jnp.array([5.])
        projection = project_to_hyperplane(y, A, b)
        assert jnp.allclose(projection, y)

    def test_b_is_zero(self):
        # Test when b is zero
        y = jnp.array([1., 2., 3.])
        A = jnp.array([[1., 1., 1.]])
        b = jnp.array([0.])
        projection = project_to_hyperplane(y, A, b)
        assert jnp.allclose(A.dot(projection), b)

    def test_A_is_identity(self):
        # Test when A is the identity matrix
        y = jnp.array([1., 2., 3.])
        A = jnp.eye(3)
        b = jnp.array([1., 2., 3.])
        projection = project_to_hyperplane(y, A, b)
        assert jnp.allclose(A.dot(projection), b)

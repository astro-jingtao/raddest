import jax
import jax.numpy as jnp

from raddest.reg.constructor import project_to_hyperplane

jax.config.update("jax_enable_x64", True)


class TestProjectToHyperplane:
    def test_valid_projection(self):
        # Test with a valid projection
        y = jnp.array([1.0, 2.0, 3.0])
        A = jnp.array([[1.0, 1.0, 1.0]])
        b = jnp.array([5.0])
        projection = project_to_hyperplane(y, A, b)
        assert jnp.allclose(A.dot(projection), b)

        # Test with a more complex scenario
        y = jnp.array([4.0, 5.0, 6.0])
        A = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = jnp.array([7.0, 8.0])
        projection = project_to_hyperplane(y, A, b)
        assert jnp.allclose(A.dot(projection), b)

        # Test with a 2D hyperplane
        y = jnp.array([1.0, 1.0])
        A = jnp.array([[1.0, 1.0]])
        b = jnp.array([1.0])
        projection = project_to_hyperplane(y, A, b)
        assert jnp.allclose(A.dot(projection), b)

        # Test with a different A matrix
        y = jnp.array([1.0, 2.0])
        A = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        b = jnp.array([1.0, 1.0])
        projection = project_to_hyperplane(y, A, b)
        assert jnp.allclose(A.dot(projection), b)

    def test_projection_onto_same_point(self):
        # Test when the point is already on the hyperplane
        y = jnp.array([1.0, 2.0, 2.0])
        A = jnp.array([[1.0, 1.0, 1.0]])
        b = jnp.array([5.0])
        projection = project_to_hyperplane(y, A, b)
        assert jnp.allclose(projection, y)

    def test_b_is_zero(self):
        # Test when b is zero
        y = jnp.array([1.0, 2.0, 3.0])
        A = jnp.array([[1.0, 1.0, 1.0]])
        b = jnp.array([0.0])
        projection = project_to_hyperplane(y, A, b)
        assert jnp.allclose(A.dot(projection), b)

    def test_A_is_identity(self):
        # Test when A is the identity matrix
        y = jnp.array([1.0, 2.0, 3.0])
        A = jnp.eye(3)
        b = jnp.array([1.0, 2.0, 3.0])
        projection = project_to_hyperplane(y, A, b)
        assert jnp.allclose(A.dot(projection), b)

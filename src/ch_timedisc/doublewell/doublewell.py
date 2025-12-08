class DoubleWell:
    """
    Double well potential function and its derivatives.

    This class implements the double well potential commonly used in phase
    field models and other applications. The potential is defined as:

    f(pf) = scaling * pf^2 * (1 - pf)^2

    The class also provides convenient functions for related potentials and
    their derivatives, useful for time discretization schemes.

    Attributes:
        scaling (float): A scaling factor applied to all potential values.
                        Default is 1.0.
    """

    def __init__(self, scaling=1.0):
        """
        Initialize the DoubleWell potential.

        Args:
            scaling (float, optional): Scaling factor for the potential.
                Defaults to 1.0.
        """
        self.scaling = scaling

    def __call__(self, pf):
        """
        Evaluate the double well potential.

        Args:
            pf (float or array): Phase field value(s) in [0, 1].

        Returns:
            float or array: The potential value(s).
        """
        return self.scaling * pf**2 * (1 - pf) ** 2

    def prime(self, pf):
        """
        Evaluate the derivative of the double well potential.

        Args:
            pf (float or array): Phase field value(s).

        Returns:
            float or array: The derivative value(s).
        """
        return self.scaling * (2 * pf - 6 * pf**2 + 4 * pf**3)

    def c(self, pf):
        """
        Evaluate the convex part of the potential decomposition.

        Used in convex-concave decomposition for time discretization.

        Args:
            pf (float or array): Phase field value(s).

        Returns:
            float or array: The convex potential value(s).
        """
        return self.scaling * (((pf - 0.5) ** 4 + 1 / 16))

    def e(self, pf):
        """
        Evaluate the concave (error) part of the potential decomposition.

        Used in convex-concave decomposition. Note that __call__ = c - e.

        Args:
            pf (float or array): Phase field value(s).

        Returns:
            float or array: The concave potential value(s).
        """
        return self.scaling * 0.5 * (pf - 0.5) ** 2

    def cprime(self, pf):
        """
        Evaluate the derivative of the convex part.

        Args:
            pf (float or array): Phase field value(s).

        Returns:
            float or array: The derivative value(s) of c.
        """
        return self.scaling * 4.0 * (pf - 0.5) ** 3

    def eprime(self, pf):
        """
        Evaluate the derivative of the concave part.

        Note that prime = cprime - eprime.

        Args:
            pf (float or array): Phase field value(s).

        Returns:
            float or array: The derivative value(s) of e.
        """
        return self.scaling * (pf - 0.5)

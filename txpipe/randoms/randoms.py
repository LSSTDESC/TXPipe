import numpy as np
import scipy.stats


def random_points_in_triangle(p1, p2, p3, n):
    """Generate random points uniformly distributed in a triangle.

    The points can be in any number of dimensions.

    Params
    ------
    p1: 1D array
        Vector pointing to the first vertex
    p2: 1D array
        Vector pointing to the second vertex
    p3: 1D array
        Vector pointing to the third vertex
    n: int
        Number of points to generate

    Returns
    -------
    p: array
        n * ndim array of points in the triangle

    """
    # Find the edge vector
    v1 = p2 - p1
    v2 = p3 - p1

    # Generate 2n random uniform values
    # as a1 and a2
    a = np.random.uniform(0.0, 1.0, 2 * n)
    a1 = a[:n]
    a2 = a[-n:]

    # Find any points that flipped over to being outside
    # the triangle and flip them back in
    w = np.where(a1 + a2 > 1)
    a1[w] = 1 - a1[w]
    a2[w] = 1 - a2[w]

    # Generate p_{ij} = p1_{j} + a1_{i}*v1_{j} + a2_{i}*v2_{j}
    # which should be uniformly distributed in the quadrilateral
    p = p1[np.newaxis, :] + np.einsum("i,j->ij", a1, v1) + np.einsum("i,j->ij", a2, v2)

    return p


def random_points_in_quadrilateral(p1, p2, p3, p4, n):
    """Generate random points uniformly distributed in a quadrilateral.

    The points can be in any number of dimensions.  If two of them
    co-incide so you are actually specifying a triangle it will still work.

    As long as you specify the vertices consistently either clockwise or
    anti-clockwise is okay.

    Params
    ------
    p1: 1D array
        Vector pointing to the first vertex
    p2: 1D array
        Vector pointing to the second vertex
    p3: 1D array
        Vector pointing to the third vertex
    p4: 1D array
        Vector pointing to the fourth vertex
    n: int
        Number of points to generate

    Returns
    -------
    p: array
        n * ndim array of points in the quadrilateral

    """

    # Find the edge vectors
    v1 = p2 - p1
    v2 = p3 - p2
    v3 = p4 - p3
    v4 = p1 - p4

    # Find the area of the two triangles making up the quadrilateral,
    # And therefore the expected fraction of the points in each half.
    A1 = 0.5 * np.linalg.norm(np.cross(v1, v2))
    A2 = 0.5 * np.linalg.norm(np.cross(v3, v4))
    f1 = abs(A1) / abs(A1 + A2)

    # We choose the actual number of points according to a binomial distribution
    n1 = scipy.stats.binom.rvs(n, f1)
    n2 = n - n1

    # Now generate the points in each half-triangle
    x1 = random_points_in_triangle(p1, p2, p3, n1)
    x2 = random_points_in_triangle(p1, p3, p4, n2)

    # Group the points and return
    return np.vstack((x1, x2))

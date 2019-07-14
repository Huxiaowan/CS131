import numpy as np

M = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
a = np.array([[1, 1, 0]])
b = np.array([-1, 2, 5])


def dot_product(vector1, vector2):
    """ Implement dot product of the two vectors.
    Args:
        vector1: numpy array of shape (x, n)
        vector2: numpy array of shape (n, x)

    Returns:
        out: numpy array of shape (x,x) (scalar if x = 1)
    """
    out = None
    # YOUR CODE HERE
    out = np.dot(vector1, vector2)
    pass
    # END YOUR CODE

    return out


aDotB = dot_product(a, b)
print (aDotB)


def matrix_mult(M, vector1, vector2):
    """ Implement (vector1.T * vector2) * (M * vector1)
    Args:
        M: numpy matrix of shape (x, n)
        vector1: numpy array of shape (1, n)
        vector2: numpy array of shape (n, 1)

    Returns:
        out: numpy matrix of shape (1, x)
    """
    out = None
    # YOUR CODE HERE
    out = np.dot(M, vector1.T)
    out = np.dot(vector1, vector2) * out
    # out = np.dot(out, vector1.T)
    pass
    # END YOUR CODE

    return out


ans = matrix_mult(M, a, b)
print (ans)

#print(get_singular_values(M, 1))
#print(get_singular_values(M, 2))


def svd(matrix):
    """ Implement Singular Value Decomposition
    Args:
        matrix: numpy matrix of shape (m, n)

    Returns:
        u: numpy array of shape (m, m)
        s: numpy array of shape (k)
        v: numpy array of shape (n, n)
    """
    u = None
    s = None
    v = None
    # YOUR CODE HERE
    u, s, v = np.linalg.svd(matrix)
    pass
    # END YOUR CODE

    return u, s, v


def get_singular_values(matrix, n):
    """ Return top n singular values of matrix
    Args:
        matrix: numpy matrix of shape (m, w)
        n: number of singular values to output

    Returns:
        singular_values: array of shape (n)
    """
    singular_values = None
    u, s, v = svd(matrix)
    # YOUR CODE HERE
    singular_values = np.diag(s)
    singular_values = singular_values[:n, :n]
    #singular_values = s[:n, :n]
    pass
    # END YOUR CODE
    return singular_values


print(get_singular_values(M, 1))
print(get_singular_values(M, 2))


def eigen_decomp(matrix):
    """ Implement Eigen Value Decomposition
    Args:
        matrix: numpy matrix of shape (m, )

    Returns:
        w: numpy array of shape (m, m) such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
    """
    w = None
    v = None
    # YOUR CODE HERE
    w, v = np.linalg.eig(matrix)
    pass
    # END YOUR CODE
    return w, v


def get_eigen_values_and_vectors(matrix, num_values):
    """ Return top n eigen values and corresponding vectors of matrix
    Args:
        matrix: numpy matrix of shape (m, m)
        num_values: number of eigen values and respective vectors to return

    Returns:
        eigen_values: array of shape (n)
        eigen_vectors: array of shape (m, n)
    """
    w, v = eigen_decomp(matrix)
    eigen_values = w[:num_values]
    eigen_vectors = v[:num_values]
    # YOUR CODE HERE
    pass
    # END YOUR CODE
    return eigen_values, eigen_vectors


M = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
val, vec = get_eigen_values_and_vectors(M[:, :3], 1)
print("Values = \n", val)
print("Vectors = \n", vec)
val, vec = get_eigen_values_and_vectors(M[:, :3], 2)
print("Values = \n", val)
print("Vectors = \n", vec)

import unittest
from fractions import Fraction

import matrix


def swap_row(a, row_1, row_2):
    pass


def scalar_multiply(a, multiple, working_row):
    return [[a[row][col] * multiple if working_row == row else a[row][col] for col in range(len(a[0]))] for row in range(len(a))]


def add_row(a, multiple, original_row, working_row):
    return [[a[row][col] + (multiple * a[original_row][col]) if working_row == row else a[row][col] for col in range(len(a[0]))] for row in range(len(a))]


def eye(m):
    return [[1 if col == row else 0 for row in range(m)] for col in range(m)]


def get_dims(a):
    return len(a), len(a[0])


def partition(a, b):
    a_dims = get_dims(a)
    b_dims = get_dims(b)

    a_nrows = a_dims[0]
    a_ncols = a_dims[1]
    b_nrows = b_dims[0]
    b_ncols = b_dims[1]

    if a_nrows == b_nrows:
        return [[a[row][col] if col < a_ncols else b[row][col - a_ncols] for col in range(a_ncols + b_ncols)] for row in range(a_nrows)]

    else:
        raise Exception("The number of rows in matrix A and matrix B do not match!")


def validate_square_matrix(a, b=None):
    a_dims = get_dims(a)
    a_nrows = a_dims[0]
    a_ncols = a_dims[1]

    if b is not None:
        b_dims = get_dims(b)
        b_nrows = b_dims[0]
        b_ncols = b_dims[1]

    if b is not None and a_nrows == a_ncols and b_nrows == b_ncols:
        return to_ref(a, a_nrows, b)

    elif b is None and a_nrows == a_ncols:
        return to_ref(a, a_nrows)

    else:
        raise Exception("Please ensure matrix passed into function is a square matrix.")


def to_ref(a, m, b=None):  # O(n^2)?
    a = scalar_multiply(a, 1 / a[0][0], 0)

    for row in range(1, len(a)):  # add loop above for looping through cols in matrix
        if a[row][0] == 0 or a[row][0] == Fraction(0):
            continue
        else:
            a = add_row(a, 0 - a[row][0], 0, row)

    return a


class TestMatrixOps(unittest.TestCase):
    def test_eye(self):
        self.assertEqual(A, B)

    def test_dims(self):
        self.assertEqual(get_dims(A), (5, 5))
        self.assertEqual(get_dims(C), (4, 3))

    def test_partition(self):
        self.assertEqual(partition(D, eye(2)), E)
        with self.assertRaises(Exception): partition(D, eye(3))


if __name__ == "__main__":
    A = eye(5)
    
    B = [[1, 0, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 0, 1]]
    
    C = [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9],
         [10, 11, 12]]
    
    D = [[1, 2],
         [3, 4]]

    E = [[1, 2, 1, 0],
         [3, 4, 0, 1]]

    F = [[3, 0, 6, 4],
         [0, 1, 3, 2],
         [1, 3, 3, 0],
         [0, 1, 1, 0]]

    G = [[2, -4, 0, 6, 12],
         [12, 7, -7, 0, 0],
         [0, -5, 6, -8, 5],
         [3, 9, 1, 1, -7],
         [0, 0, -6, 4, 2]]

    F = matrix.to_frac(F)
    G = matrix.to_frac(G)

    # unittest.main()

    print(matrix.from_frac(validate_square_matrix(F)))
    print(matrix.from_frac(validate_square_matrix(G)))

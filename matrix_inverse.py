import unittest
from fractions import Fraction
# from matrix import from_frac, to_frac, matmul


def swap_row(a, row_1, row_2):
    temp_matrix = a
    temp_matrix[row_1], temp_matrix[row_2] = temp_matrix[row_2], temp_matrix[row_1]
    return temp_matrix


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

    if b is not None and a_nrows == a_ncols and b_nrows == b_ncols and a_nrows == b_nrows and a_ncols == b_ncols:
        return to_rref(a, b)

    elif b is None and a_nrows == a_ncols:
        return to_rref(a)

    else:
        raise Exception("Please ensure matrix passed into function is a square matrix.")


def to_ref(a, b=None):  # O(n^2)?
    for piv_row_num in range(len(a)):
        pivot_row = a[piv_row_num]

        if pivot_row[piv_row_num] is int and pivot_row[piv_row_num] == 0:
            for row_num in range(piv_row_num + 1, len(a)):
                other_row = a[row_num]

                if other_row[piv_row_num] != 0:
                    a = swap_row(a, piv_row_num, row_num)
                    if b is not None:
                        b = swap_row(b, piv_row_num, row_num)
                    break

                else:
                    continue

        elif isinstance(pivot_row[piv_row_num], Fraction) and pivot_row[piv_row_num] == Fraction(0):
            for row_num in range(piv_row_num + 1, len(a)):
                other_row = a[row_num]

                if other_row[piv_row_num] != Fraction(0):
                    a = swap_row(a, piv_row_num, row_num)
                    if b is not None:
                        b = swap_row(b, piv_row_num, row_num)
                    break

                else:
                    continue

        pivot_row = a[piv_row_num]
        
        if pivot_row[piv_row_num] is int and pivot_row[piv_row_num] not in [0, 1]:
            multiplier = 1 / pivot_row[piv_row_num]
            a = scalar_multiply(a, multiplier, piv_row_num)
            if b is not None:
                b = scalar_multiply(b, multiplier, piv_row_num)

        elif isinstance(pivot_row[piv_row_num], Fraction) and pivot_row[piv_row_num] not in[Fraction(0), Fraction(1)]:
            multiplier = Fraction(1) / pivot_row[piv_row_num]
            a = scalar_multiply(a, multiplier, piv_row_num)
            if b is not None:
                b = scalar_multiply(b, multiplier, piv_row_num)

        else:  # do nothing
            pass

        for row_num in range(piv_row_num + 1, len(a)):
            other_row = a[row_num]
            if other_row[piv_row_num] is int and other_row[piv_row_num] != 0:
                multiplier = 0 - other_row[piv_row_num]
                a = add_row(a, multiplier, piv_row_num, row_num)
                if b is not None:
                    b = add_row(b, multiplier, piv_row_num, row_num)

            elif isinstance(other_row[piv_row_num], Fraction) and other_row[piv_row_num] != Fraction(0):
                multiplier = Fraction(0) - other_row[piv_row_num]
                a = add_row(a, multiplier, piv_row_num, row_num)
                if b is not None:
                    b = add_row(b, multiplier, piv_row_num, row_num)

            else:
                continue

    return a, b


def to_rref(a, b=None):
    a, b = to_ref(a, b)

    for row_num in range(len(a) - 1, -1, -1):
        row = a[row_num]

        for other_row_num in range(row_num - 1, -1, -1):
            other_row = a[other_row_num]

            if other_row[row_num] is int and other_row[row_num] != 0:
                multiplier = 0 - other_row[row_num]
                a = add_row(a, multiplier, row_num, other_row_num)
                if b is not None:
                    b = add_row(b, multiplier, row_num, other_row_num)

            elif isinstance(other_row[row_num], Fraction) and other_row[row_num] != Fraction(0):
                multiplier = Fraction(0) - other_row[row_num]
                a = add_row(a, multiplier, row_num, other_row_num)
                if b is not None:
                    b = add_row(b, multiplier, row_num, other_row_num)

            else:
                continue

    if b is not None:
        return b

    else:
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

    def test_row_swap(self):
        self.assertEqual(swap_row(eye(4), 2, 3), K)

    def test_gauss_jordan_elim(self):
        self.assertEqual(from_frac(matmul(validate_square_matrix(to_frac(D), to_frac(eye(len(D)))), to_frac(D))), eye(len(D)))
        self.assertEqual(from_frac(matmul(validate_square_matrix(to_frac(F), to_frac(eye(len(F)))), to_frac(F))), eye(len(F)))
        self.assertEqual(from_frac(matmul(validate_square_matrix(to_frac(G), to_frac(eye(len(G)))), to_frac(G))), eye(len(G)))
        self.assertEqual(from_frac(matmul(validate_square_matrix(to_frac(H), to_frac(eye(len(H)))), to_frac(H))), eye(len(H)))
        self.assertEqual(from_frac(matmul(validate_square_matrix(to_frac(J), to_frac(eye(len(J)))), to_frac(J))), eye(len(J)))
        self.assertEqual(from_frac(matmul(validate_square_matrix(to_frac(K), to_frac(eye(len(K)))), to_frac(K))), eye(len(K)))
        self.assertEqual(from_frac(matmul(validate_square_matrix(to_frac(L), to_frac(eye(len(L)))), to_frac(L))), eye(len(L)))


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

    H = [[1, 5, -1],
         [2, 2, -2],
         [-1, 4, 3]]

    J = [[7, 2, 3, 0],
         [1, 1, 0, 7],
         [0, 0, 1, 6],
         [0, 1, 5, 0]]

    K = [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 1, 0]]

    L = [[7, 2, 3, 0],
         [1, 1, 0, 7],
         [0, 0, 0, 6],
         [0, 1, 5, 0]]

    # unittest.main()

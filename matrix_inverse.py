import unittest


def swap_row(a, row_1, row_2):
    pass


def scalar_multipy(a, row):
    pass


def add_row(a, multiple, row_1, row_2):
    pass


def eye(m):
    return [[1 if col == row else 0 for row in range(m)] for col in range(m)]


def get_dims(a):
    return (len(a), len(a[0]))

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

def to_ref(a, b=None):
    pass


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

    unittest.main()

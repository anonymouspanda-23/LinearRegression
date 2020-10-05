from fractions import Fraction


def matmul(a, b):
    a_ncols = len(a[0])
    b_nrows = len(b)

    if a_ncols != b_nrows:
        raise Exception("The number of columns in A is not equal to the number of rows in B! Is your matrix 2 dimensional?")
    
    return [[sum([a[i][m]*b[m][j] for m in range(len(a[0]))]) for j in range(len(b[0]))] for i in range(len(a))]
    

def transpose(a):
    return list(map(list, zip(*a)))


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


def inverse(a, b=None):
    a_dims = get_dims(a)
    a_nrows = a_dims[0]
    a_ncols = a_dims[1]

    if b is not None:
        b_dims = get_dims(b)
        b_nrows = b_dims[0]
        b_ncols = b_dims[1]

    else:
        b_nrows = 0
        b_ncols = 0

    if b is not None and a_nrows == a_ncols and b_nrows == b_ncols and a_nrows == b_nrows and a_ncols == b_ncols:
        a, b = to_row_reduced_echelon_format(a, b)

    elif b is None and a_nrows == a_ncols:
        a, b = to_row_reduced_echelon_format(a)

    else:
        raise Exception("Please ensure matrix passed into function is a square matrix.")

    if b is not None:
        return b
    else:
        return a


def reduce_pivot_row(a, pivot_row_number, b):
    pivot_row = a[pivot_row_number]
    value_type = type(pivot_row[pivot_row_number])

    if pivot_row[pivot_row_number] not in [value_type(0), value_type(1)]:
        multiplier = 1 / pivot_row[pivot_row_number]
        a = scalar_multiply(a, multiplier, pivot_row_number)
        if b is not None:
            b = scalar_multiply(b, multiplier, pivot_row_number)

    return a, b


def reduce_non_pivot_rows(a, pivot_row_num, non_pivot_row_num, b):
    non_pivot_row = a[non_pivot_row_num]
    value_type = type(non_pivot_row[pivot_row_num])

    if non_pivot_row[pivot_row_num] != value_type(0):
        multiplier = value_type(0) - non_pivot_row[pivot_row_num]
        a = add_row(a, multiplier, pivot_row_num, non_pivot_row_num)
        if b is not None:
            b = add_row(b, multiplier, pivot_row_num, non_pivot_row_num)

    return a, b


def validate_pivot_row(a, pivot_row_num, b):
    pivot_row = a[pivot_row_num]
    value_type = type(pivot_row[pivot_row_num])

    if pivot_row[pivot_row_num] == value_type(0):
        for non_pivot_row_num in range(pivot_row_num + 1, len(a)):
            non_pivot_row = a[non_pivot_row_num]

            if non_pivot_row[pivot_row_num] != value_type(0):
                a = swap_row(a, pivot_row_num, non_pivot_row_num)
                if b is not None:
                    b = swap_row(b, pivot_row_num, non_pivot_row_num)
                break

            else:
                continue

    return a, b


def to_row_echelon_format(a, b=None):  # O(n^2)?
    for piv_row_num in range(len(a)):
        a, b = validate_pivot_row(a, piv_row_num, b)
        a, b = reduce_pivot_row(a, piv_row_num, b)

        for other_row_num in range(piv_row_num + 1, len(a)):
            a, b = reduce_non_pivot_rows(a, piv_row_num, other_row_num, b)

    return a, b


def to_row_reduced_echelon_format(a, b=None):
    a, b = to_row_echelon_format(a, b)

    for pivot_row_num in range(len(a) - 1, -1, -1):
        for other_row_num in range(pivot_row_num - 1, -1, -1):
            a, b = reduce_non_pivot_rows(a, pivot_row_num, other_row_num, b)

    return a, b


def to_frac(a):
    return [[Fraction(value) for value in row] for row in a]


def from_frac(a):
    return [[float(value.numerator)/float(value.denominator) for value in row] for row in a]


def disp_mat(a):
    str_repr = ""
    for row in a:
        str_repr = str_repr + str(row) + '\n'

    print(str_repr)


if __name__ == '__main__':
    x = [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [1, 10], [1, 11], [1, 12]]
    y = [[5], [7], [9], [11], [13], [15], [17], [19], [21], [23], [25], [27]]

    x = to_frac(x)
    y = to_frac(y)

    x_transpose = transpose(x)
    ans = matmul(matmul(inverse(matmul(x_transpose, x), eye(len(matmul(x_transpose, x)))), x_transpose), y)

    theta = from_frac(ans)

    print(theta)

    pred = [[1, 16], [1, 99], [1, 1325]]

    pred = matmul(pred, theta)

    print(pred)

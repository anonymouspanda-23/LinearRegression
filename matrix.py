from decimal import Decimal
from fractions import Fraction
from matrix_inverse import eye, validate_square_matrix


def matmul(a, b):
    a_ncols = len(a[0])
    b_nrows = len(b)

    if a_ncols != b_nrows:
        raise Exception("The number of columns in A is not equal to the number of rows in B! Is your matrix 2 dimensional?")
    
    return [[sum([a[i][m]*b[m][j] for m in range(len(a[0]))]) for j in range(len(b[0]))] for i in range(len(a))]
    

def transpose(a):
    # return [[a[j][i] for j in range(len(a))] for i in range(len(a[0]))]
    return list(map(list, zip(*a)))


def eliminate(r1, r2, col, target=0):
    fac = (r2[col]-target) / r1[col]
    for i in range(len(r2)):
        r2[i] -= fac * r1[i]

def gauss(a):
    for i in range(len(a)):
        if a[i][i] == 0:
            for j in range(1, len(a)):
                if a[i][j] != Fraction(0):
                    a[i], a[j] = a[j], a[i]
                    break
            else:
                print("MATRIX NOT INVERTIBLE")
                return Fraction(-1)
        for j in range(i+1, len(a)):
            eliminate(a[i], a[j], i)
    for i in range(len(a)-1, -1, -1):
        for j in range(i-1, -1, -1):
            eliminate(a[i], a[j], i)
    for i in range(len(a)):
        eliminate(a[i], a[i], i, target=1)
    return a


def inverse(a):
    tmp = [[] for _ in a]
    for i, row in enumerate(a):
        assert len(row) == len(a)
        tmp[i].extend(row + [0]*i + [1] + [0]*(len(a)-i-1))
    gauss(tmp)
    ret = []
    for i in range(len(tmp)):
        ret.append(tmp[i][len(tmp[i])//2:])
    return ret


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
    ans = matmul(matmul(validate_square_matrix(matmul(x_transpose, x), eye(len(matmul(x_transpose, x)))), x_transpose), y)

    theta = from_frac(ans)

    print(theta)

    pred = [[1, 16], [1, 99], [1, 1325]]

    pred = matmul(pred, theta)

    print(pred)

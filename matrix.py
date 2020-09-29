def matmul(a,b):
    a_nrows = len(a)
    a_ncols = len(a[0])

    b_nrows = len(b)
    b_ncols = len(b[0])

    if a_ncols != b_nrows:
        return 1
    
    zip_b = zip(*b)
    # uncomment next line if python 3 : 
    zip_b = list(zip_b)
    return [[sum(ele_a*ele_b for ele_a, ele_b in zip(row_a, col_b)) 
             for col_b in zip_b] for row_a in a]

def transpose(a):
    return [[a[j][i] for j in range(len(a))] for i in range(len(a[0]))]

if __name__ == '__main__':
    a = [[1, 2, 3], [4, 5, 6]]
    b = [[1, 2], [3, 4], [5, 6]]
    
    c = matmul(a, b)
    
    print(c)


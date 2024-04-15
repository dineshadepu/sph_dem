def normalize_R_orientation(orien=[1., 0., 0.]):
    a1, a2, a3, b1, b2, b3 = declare('matrix(3)', 6)

    a1[0] = orien[0]
    a1[1] = orien[3]
    a1[2] = orien[6]

    a2[0] = orien[1]
    a2[1] = orien[4]
    a2[2] = orien[7]

    a3[0] = orien[2]
    a3[1] = orien[5]
    a3[2] = orien[8]

    # norm of col0
    na1 = (a1[0]**2. + a1[1]**2. + a1[2]**2.)**0.5
    b1[0] = a1[0] / na1
    b1[1] = a1[1] / na1
    b1[2] = a1[2] / na1

    b1_dot_a2 = b1[0] * a2[0] + b1[1] * a2[1] + b1[2] * a2[2]
    b2[0] = a2[0] - b1_dot_a2 * b1[0]
    b2[1] = a2[1] - b1_dot_a2 * b1[1]
    b2[2] = a2[2] - b1_dot_a2 * b1[2]
    nb2 = (b2[0]**2. + b2[1]**2. + b2[2]**2.)**0.5
    b2[0] = b2[0] / nb2
    b2[1] = b2[1] / nb2
    b2[2] = b2[2] / nb2

    b1_dot_a3 = b1[0] * a3[0] + b1[1] * a3[1] + b1[2] * a3[2]
    b2_dot_a3 = b2[0] * a3[0] + b2[1] * a3[1] + b2[2] * a3[2]
    b3[0] = a3[0] - b1_dot_a3 * b1[0] - b2_dot_a3 * b2[0]
    b3[1] = a3[1] - b1_dot_a3 * b1[1] - b2_dot_a3 * b2[1]
    b3[2] = a3[2] - b1_dot_a3 * b1[2] - b2_dot_a3 * b2[2]
    nb3 = (b3[0]**2. + b3[1]**2. + b3[2]**2.)**0.5
    b3[0] = b3[0] / nb3
    b3[1] = b3[1] / nb3
    b3[2] = b3[2] / nb3

    orien[0] = b1[0]
    orien[3] = b1[1]
    orien[6] = b1[2]
    orien[1] = b2[0]
    orien[4] = b2[1]
    orien[7] = b2[2]
    orien[2] = b3[0]
    orien[5] = b3[1]
    orien[8] = b3[2]


def find_transpose(a=[1.0, 0.0], result=[1.0, 0.0]):
    """Get the transpose of a matrix

    Stores the result in `result`.

    Parameters
    ----------

    a: list
    result: list
    """
    result[0] = a[0]
    result[1] = a[3]
    result[2] = a[6]

    result[3] = a[1]
    result[4] = a[4]
    result[2] = a[6]

    result[6] = a[2]
    result[7] = a[5]
    result[8] = a[8]

import numpy as np
import matplotlib.pyplot as plt
from pysph.tools.geometry import get_2d_block, get_2d_tank, get_3d_block, rotate, remove_overlap_particles
from pysph.base.utils import get_particle_array


def hydrostatic_tank_2d(fluid_length=1., fluid_height=2.,
                        tank_height=2.3, tank_layers=2,
                        fluid_spacing=0.1, tank_spacing=0.1, close=False):
    xf, yf = get_2d_block(dx=fluid_spacing,
                          length=fluid_length,
                          height=fluid_height)

    xt_4 = np.array([])
    yt_4 = np.array([])
    if close == False:
        xt_1, yt_1 = get_2d_block(dx=fluid_spacing,
                                  length=tank_layers*fluid_spacing,
                                  height=tank_height+fluid_spacing/2.)
        xt_1 -= max(xt_1) - min(xf) + fluid_spacing
        yt_1 += min(yf) - min(yt_1)

        xt_2, yt_2 = get_2d_block(dx=fluid_spacing,
                                  length=tank_layers*fluid_spacing,
                                  height=tank_height+fluid_spacing/2.)
        xt_2 += max(xf) - min(xt_2) + fluid_spacing
        yt_2 += min(yf) - min(yt_2)

    else:
        xt_1, yt_1 = get_2d_block(dx=fluid_spacing,
                                  length=tank_layers*fluid_spacing,
                                  height=fluid_height + tank_layers * fluid_spacing)
        xt_1 -= max(xt_1) - min(xf) + fluid_spacing
        yt_1 += min(yf) - min(yt_1)

        xt_2, yt_2 = get_2d_block(dx=fluid_spacing,
                                  length=tank_layers*fluid_spacing,
                                  height=fluid_height + tank_layers * fluid_spacing)
        xt_2 += max(xf) - min(xt_2) + fluid_spacing
        yt_2 += min(yf) - min(yt_2)

        xt_3, yt_3 = get_2d_block(dx=fluid_spacing,
                                length=max(xt_2) - min(xt_1),
                                height=tank_layers*fluid_spacing)
        yt_3[:] = yt_3[:] - (max(yt_3) - min(yf)) - fluid_spacing

        xt_4, yt_4 = get_2d_block(dx=fluid_spacing,
                                length=max(xt_2) - min(xt_1),
                                height=tank_layers*fluid_spacing)
        yt_4[:] = yt_4[:] + max(yf) - min(yt_4) + fluid_spacing

    xt_3, yt_3 = get_2d_block(dx=fluid_spacing,
                              length=max(xt_2) - min(xt_1),
                              height=tank_layers*fluid_spacing)
    yt_3[:] = yt_3[:] - (max(yt_3) - min(yf)) - fluid_spacing

    xt = np.concatenate([xt_1, xt_2, xt_3, xt_4])
    yt = np.concatenate([yt_1, yt_2, yt_3, yt_4])

    # plt.scatter(xt_3, yt_3)
    # plt.show()

    return xf, yf, xt, yt


def get_truncated_circle_from_3d_block(diameter=1., spacing=0.05):
    x, y, z = get_3d_block(dx=spacing,
                           length=diameter * 1.25,
                           height=diameter * 1.25,
                           depth=diameter * 1.25)
    indices = []
    for i in range(len(x)):
        if x[i]**2. + y[i]**2. + z[i]**2. <= (diameter / 2.)**2.:
            indices.append(i)

    indices = np.asarray([indices])
    # sphere coordinates
    x_s, y_s, z_s = x[indices], y[indices], z[indices]
    x_s = x_s.ravel()
    y_s = y_s.ravel()
    z_s = z_s.ravel()
    return x_s, y_s, z_s


def create_circle_1(diameter=1, spacing=0.05, center=None):
    dx = spacing
    x = [0.0]
    y = [0.0]
    r = spacing
    nt = 0
    radius = diameter / 2.

    tmp_dist = radius - spacing/2.
    i = 0
    while tmp_dist > spacing/2.:
        perimeter = 2. * np.pi * tmp_dist
        no_of_points = int(perimeter / spacing) + 1
        theta = np.linspace(0., 2. * np.pi, no_of_points)
        for t in theta[:-1]:
            x.append(tmp_dist * np.cos(t))
            y.append(tmp_dist * np.sin(t))
        i = i + 1
        tmp_dist = radius - spacing/2. - i * spacing

    x = np.array(x)
    y = np.array(y)
    x, y = (t.ravel() for t in (x, y))
    if center is None:
        return x, y
    else:
        return x + center[0], y + center[1]


def create_wedge(half_length=0.25, angle=30, spacing=3*1e-3):
    """
    angle in degrees
    """
    wedge_height = np.tan(angle * np.pi / 180) * half_length

    # create points in height direction
    x = np.array([0.], dtype=int)
    y = np.array([0.], dtype=int)
    y_reference = np.arange(0., wedge_height, spacing)
    for i in range(len(y_reference)):
        x_len = y_reference[i] / np.tan(angle * np.pi / 180)
        x_local = np.arange(-x_len, x_len, spacing)
        y_local = np.ones_like(x_local) * y_reference[i]
        x = np.concatenate((x, x_local))
        y = np.concatenate((y, y_local))

    return x, y


def create_wedge_1(half_length=0.25, angle=30, spacing=3*1e-3):
    """
    angle in degrees
    """
    # create a 2d block
    # side length equal to the wedge hypotenuse value
    radians = angle * np.pi / 180
    height = half_length / np.cos(radians)
    x_left, y_left = get_2d_block(dx=spacing,
                                  length=2. * half_length,
                                  height=height,
                                  center=[0., 0.])
    z_left = np.zeros_like(x_left)
    # rotate the block 30 degrees
    x_left, y_left, z_left = rotate(x=x_left, y=y_left, z=z_left, angle=90-angle)
    pa_left = get_particle_array(x=x_left, y=y_left, z=z_left, h=spacing)
    min_x = np.min(x_left)
    index = np.where(x_left == min_x)
    y_cond = y_left[index[0]]
    # delete indices which are above y
    delete_cond = np.where(y_left > y_cond)
    pa_left.remove_particles(delete_cond[0])
    # x_left = pa_left.x
    # y_left = pa_left.y
    # z_left = pa_left.z

    # create a 2d block
    # side length equal to the wedge hypotenuse value
    radians = angle * np.pi / 180
    height = half_length / np.cos(radians)
    x_right, y_right = get_2d_block(dx=spacing,
                                    length=2. * half_length,
                                    height=height,
                                    center=[0., 0.])
    z_right = np.zeros_like(x_right)
    # rotate the block 30 degrees
    x_right, y_right, z_right = rotate(x=x_right, y=y_right, z=z_right, angle=-(90-angle))
    pa_right = get_particle_array(x=x_right, y=y_right, z=z_right, h=spacing)
    # delete indices which are above y
    delete_cond = np.where(y_right > y_cond)
    pa_right.remove_particles(delete_cond[0])
    # x_right = pa_right.x
    # y_right = pa_right.y
    # z_right = pa_right.z

    # move the right particle array so that it aligns pointly with the bottom
    min_y_left = np.min(pa_left.y)
    index = np.where(pa_left.y == min_y_left)
    x_min_y_left = pa_left.x[index[0]]

    min_y_right = np.min(pa_right.y)
    index = np.where(pa_right.y == min_y_right)
    x_min_y_right = pa_right.x[index[0]]
    # now move the right block
    pa_right.x[:] += x_min_y_left - x_min_y_right

    remove_overlap_particles(pa_right, pa_left, 0.5 * spacing)
    x_left = pa_left.x
    y_left = pa_left.y
    x_right = pa_right.x
    y_right = pa_right.y

    x = np.concatenate([x_right, x_left])
    y = np.concatenate([y_right, y_left])
    z = np.zeros_like(x)
    return x, y


def get_fluid_tank_3d(fluid_length,
                      fluid_height,
                      fluid_depth,
                      tank_length,
                      tank_height,
                      tank_layers,
                      fluid_spacing,
                      tank_spacing,
                      hydrostatic=False):
    """
    length is in x-direction
    height is in y-direction
    depth is in z-direction
    """
    xf, yf, zf = get_3d_block(dx=fluid_spacing,
                              length=fluid_length,
                              height=fluid_height,
                              depth=fluid_depth)

    # create a tank layer on the left
    xt_left, yt_left, zt_left = get_3d_block(dx=fluid_spacing,
                                             length=tank_spacing *
                                             (tank_layers - 1),
                                             height=tank_height,
                                             depth=fluid_depth)

    xt_right, yt_right, zt_right = get_3d_block(dx=fluid_spacing,
                                                length=tank_spacing *
                                                (tank_layers - 1),
                                                height=tank_height,
                                                depth=fluid_depth)

    # adjust the left wall of tank
    xt_left += np.min(xf) - np.max(xt_left) - tank_spacing
    yt_left += np.min(yf) - np.min(yt_left) + 0. * tank_spacing

    # adjust the right wall of tank
    xt_right += np.max(xf) - np.min(xt_right) + tank_spacing
    if hydrostatic is False:
        xt_right += tank_length - fluid_length

    yt_right += np.min(yf) - np.min(yt_right) + 0. * tank_spacing

    # create the wall in the front
    xt_front, yt_front, zt_front = get_3d_block(
        dx=fluid_spacing,
        length=np.max(xt_right) - np.min(xt_left),
        height=tank_height,
        depth=tank_spacing * (tank_layers - 1))
    xt_front += np.min(xt_left) - np.min(xt_front)
    yt_front += np.min(yf) - np.min(yt_front) + 0. * tank_spacing
    zt_front += np.max(zt_left) - np.min(zt_front) + tank_spacing * 1

    # create the wall in the back
    xt_back, yt_back, zt_back = get_3d_block(
        dx=fluid_spacing,
        length=np.max(xt_right) - np.min(xt_left),
        height=tank_height,
        depth=tank_spacing * (tank_layers - 1))
    xt_back += np.min(xt_left) - np.min(xt_back)
    yt_back += np.min(yf) - np.min(yt_back) + 0. * tank_spacing
    zt_back += np.min(zt_left) - np.max(zt_back) - tank_spacing * 1

    # create the wall in the bottom
    xt_bottom, yt_bottom, zt_bottom = get_3d_block(
        dx=fluid_spacing,
        length=np.max(xt_right) - np.min(xt_left),
        height=tank_spacing * (tank_layers - 1),
        depth=np.max(zt_front) - np.min(zt_back))
    xt_bottom += np.min(xt_left) - np.min(xt_bottom)
    yt_bottom += np.min(yt_left) - np.max(yt_bottom) - tank_spacing * 1

    xt = np.concatenate([xt_left, xt_right, xt_front, xt_back, xt_bottom])
    yt = np.concatenate([yt_left, yt_right, yt_front, yt_back, yt_bottom])
    zt = np.concatenate([zt_left, zt_right, zt_front, zt_back, zt_bottom])
    return xf, yf, zf, xt, yt, zt


def create_tank_2d_from_block_2d(xf, yf, tank_length, tank_height,
                                 tank_spacing, tank_layers, close=False):
    """
    This is mainly used by granular flows

    Tank particles radius is spacing / 2.
    """
    ####################################
    # create the left wall of the tank #
    ####################################
    xleft, yleft = get_2d_block(dx=tank_spacing,
                                length=(tank_layers - 1) * tank_spacing,
                                height=tank_height,
                                center=[0., 0.])
    xleft += min(xf) - max(xleft) - tank_spacing
    yleft += min(yf) - min(yleft)

    xright = xleft + abs(min(xleft)) + tank_length + tank_spacing
    yright = yleft

    xbottom, ybottom = get_2d_block(dx=tank_spacing,
                                    length=max(xright) - min(xleft),
                                    height=(tank_layers - 1) * tank_spacing,
                                    center=[0., 0.])
    xbottom += min(xleft) - min(xbottom)
    ybottom += min(yleft) - max(ybottom) - tank_spacing

    xtop = np.array([])
    ytop = np.array([])
    if close is True:
        xtop, ytop = get_2d_block(dx=tank_spacing,
                                  length=max(xright) - min(xleft),
                                  height=(tank_layers - 1) * tank_spacing,
                                  center=[0., 0.])
        xtop += min(xleft) - min(xtop)
        ytop += max(yleft) - min(ytop) - tank_spacing

    x = np.concatenate([xleft, xright, xbottom, xtop])
    y = np.concatenate([yleft, yright, ybottom, ytop])

    return x, y


def translate_system_with_left_corner_as_origin(x, y, z):
    translation = [min(x), min(y), min(z)]
    x[:] = x[:] - min(x)
    y[:] = y[:] - min(y)
    z[:] = z[:] - min(z)
    return translation


def test_hydrostatic_tank():
    xf, yf, xt, yt = hydrostatic_tank_2d(1., 1., 1.5, 3, 0.1, 0.1 / 2.)
    plt.scatter(xt, yt)
    plt.scatter(xf, yf)
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()


def test_create_tank_2d_from_block_2d():
    xf, yf = get_2d_block(0.1, 1., 1.)
    xt, yt = create_tank_2d_from_block_2d(xf, yf, 2., 2., 0.1, 3)
    plt.scatter(xt, yt)
    plt.scatter(xf, yf)
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()


# test_create_tank_2d_from_block_2d()


def get_files_at_given_times_from_log(files, times, logfile):
    import re
    result = []
    time_pattern = r"output at time\ (\d+(?:\.\d+)?)"
    file_count, time_count = 0, 0
    with open(logfile, 'r') as f:
        for line in f:
            if time_count >= len(times):
                break
            t = re.findall(time_pattern, line)
            if t:
                if float(t[0]) in times:
                    result.append(files[file_count])
                    time_count += 1
                elif float(t[0]) > times[time_count]:
                    result.append(files[file_count])
                    time_count += 1
                file_count += 1
    return result

# x, y, z = create_wedge_1()
# plt.clf()
# plt.axes().set_aspect('equal')
# plt.scatter(x, y)
# plt.show()

def get_cylindrical_fluid_tank_3d(fluid_length,
                                  fluid_height,
                                  fluid_depth,
                                  tank_length,
                                  tank_height,
                                  tank_layers,
                                  fluid_spacing,
                                  tank_spacing,
                                  hydrostatic=False):
    """
    length is in x-direction
    height is in y-direction
    depth is in z-direction
    """
    # discretize the fluid in y direction
    extended_height = tank_height + tank_layers * fluid_spacing
    y_layer = np.arange(0., extended_height, fluid_spacing)
    x = np.array([])
    z = np.array([])
    y = np.array([])
    for i in range(len(y_layer)):
    # for i in range(1):
        # print(i)
        diameter = fluid_length + 2. * tank_layers * fluid_spacing
        _x, _z = create_circle_1(diameter, fluid_spacing)
        _y = np.ones_like(_x) * y_layer[i]

        x = np.concatenate((x, _x))
        y = np.concatenate((y, _y))
        z = np.concatenate((z, _z))

    # remove particles outside the circle
    fluid_min = min(y_layer) + (tank_layers - 1) * fluid_spacing
    indices_fluid = []
    indices_tank_deletable = []
    for i in range(len(x)):
        if y[i] > fluid_min and y[i] <= fluid_height:
            if np.sqrt(x[i]*x[i] + z[i]*z[i]) - fluid_length/2. < 1e-10:
                indices_tank_deletable.append(i)
                indices_fluid.append(i)
        else:
            if y[i] > fluid_height:
                if np.sqrt(x[i]*x[i] + z[i]*z[i]) - fluid_length/2. < 1e-10:
                    indices_tank_deletable.append(i)

    x_fluid = x[indices_fluid]
    y_fluid = y[indices_fluid]
    z_fluid = z[indices_fluid]
    x_tank = np.delete(x, indices_tank_deletable)
    y_tank = np.delete(y, indices_tank_deletable)
    z_tank = np.delete(z, indices_tank_deletable)

    # create the bottom of the tank


    return x_fluid, y_fluid, z_fluid, x_tank, y_tank, z_tank


def get_3d_block_rfc(dx=0.01, width=1.0, length=1.0, depth=1.0,
                     center=np.array([0., 0., 0.])):
    """
    Generates a 3d block of particles with the length, height and depth
    parallel to x, y and z axis respectively.

    Paramters
    ---------
    dx : a number which is the spacing required
    width : a number which is the length of the block
    length : a number which is the height of the block
    depth : a number which is the depth of the block
    center : 1d array like object which is the center of the block

    Returns
    -------
    x : 1d numpy array with x coordinates of the block particles
    y : 1d numpy array with y coordinates of the block particles
    z : 1d numpy array with z coordinates of the block particles
    """

    n1 = int(width / dx) + 1
    n2 = int(length / dx) + 1
    n3 = int(depth / dx) + 1
    x, y, z = np.mgrid[-width / 2.:width / 2.:n1 * 1j, -length /
                       2.:length / 2.:n2 * 1j, -depth / 2.:depth / 2.:n3 * 1j]
    x, y, z = np.ravel(x), np.ravel(y), np.ravel(z)
    return x + center[0], y + center[1], z + center[2]


def get_fluid_tank_new_rfc_3d(fluid_width,
                              fluid_length,
                              fluid_depth,
                              tank_length,
                              tank_depth,
                              tank_layers,
                              fluid_spacing,
                              tank_spacing,
                              hydrostatic=False):
    """We follow this figure in this page
    https://en.wikipedia.org/wiki/Three-dimensional_space, and the papers follow
    similar notation.

    X x Y x Z

    X is the width (out of the plane)
    Y is the length (in plane length)
    Z is the depth (in plane height)

    A better example to see is
    https://www.sciencedirect.com/science/article/pii/S0029801821016644#sec3

    3.2 cube floating example
    """
    xf, yf, zf = get_3d_block_rfc(dx=fluid_spacing,
                                  width=fluid_width,
                                  length=fluid_length,
                                  depth=fluid_depth)

    # create a tank layer on the left
    xt_left, yt_left, zt_left = get_3d_block_rfc(dx=fluid_spacing,
                                                 width=fluid_width,
                                                 length=tank_spacing * (tank_layers - 1),
                                                 depth=tank_depth)

    xt_right, yt_right, zt_right = get_3d_block_rfc(dx=fluid_spacing,
                                                    width=fluid_width,
                                                    length=tank_spacing * (tank_layers - 1),
                                                    depth=tank_depth)

    # adjust the left wall of tank
    xt_left += np.min(xf) - np.min(xt_left)
    yt_left += np.min(yf) - np.max(yt_left) - fluid_spacing
    zt_left += np.min(zf) - np.min(zt_left)

    # adjust the right wall of tank
    xt_right += np.min(xf) - np.min(xt_right)
    yt_right += np.max(yf) - np.min(yt_right) + fluid_spacing
    zt_right += np.min(zf) - np.min(zt_right)
    if hydrostatic is False:
        yt_right += tank_length - fluid_length

    # create the wall in the front
    xt_front, yt_front, zt_front = get_3d_block_rfc(
        dx=fluid_spacing,
        width=tank_spacing * (tank_layers - 1),
        length=np.max(yt_right) - np.min(yt_left),
        depth=tank_depth)
    xt_front += np.max(xt_left) - np.min(xt_front) + fluid_spacing * 1.
    yt_front += np.min(yt_left) - np.min(yt_front) + 0. * tank_spacing
    zt_front += np.min(zt_left) - np.min(zt_front)

    # create the wall in the front
    xt_back, yt_back, zt_back = get_3d_block_rfc(
        dx=fluid_spacing,
        width=tank_spacing * (tank_layers - 1),
        length=np.max(yt_right) - np.min(yt_left),
        depth=tank_depth)
    xt_back -= np.max(xt_back) - np.min(xt_left) + fluid_spacing * 1.
    yt_back += np.min(yt_left) - np.min(yt_back) + 0. * tank_spacing
    zt_back += np.min(zt_left) - np.min(zt_back)

    # create the wall in the bottom
    xt_bottom, yt_bottom, zt_bottom = get_3d_block_rfc(
        dx=fluid_spacing,
        width=np.max(xt_front) - np.min(xt_back),
        length=np.max(yt_right) - np.min(yt_left),
        depth=tank_spacing * (tank_layers - 1))

    xt_bottom += np.max(xt_front) - np.max(xt_bottom)
    yt_bottom += np.min(yt_left) - np.min(yt_bottom)
    zt_bottom -= np.max(zt_bottom) - np.min(zt_left)
    zt_bottom -= fluid_spacing * 1.

    xt = np.concatenate([xt_left, xt_right, xt_front, xt_back, xt_bottom])
    yt = np.concatenate([yt_left, yt_right, yt_front, yt_back, yt_bottom])
    zt = np.concatenate([zt_left, zt_right, zt_front, zt_back, zt_bottom])
    return xf, yf, zf, xt, yt, zt

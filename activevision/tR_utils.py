import numpy as np


def camera_to_world_tR(tcw, Rcw):
    """ Find camera to world coordinate system parameters.

        Subscript cw --> World to Camera
        Subscript wc --> Camera to World

    Parameters
    ----------
    tcw: numpy array
        Translation parameter of World to Camera coordinate of shape (3,1).
    Rcw: numpy array
        Rotation parameter of World to Camera coordinate of shape (3,3).

    Returns
    -------
    twc: numpy array of shape (3,1)
        Translation parameter of Camera to World coordinate.
    Rwc: numpy array of shape (3,3)
        Rotation parameter of Camera to World coordinate.

    Notes
    -------
        Xw = Rcw Xc + Tcw
        --> (Rcw)-1 Xw = Xc + (Rcw)-1 Tcw
        --> Xc = [(Rcw)-1] Xw  + [-(Rcw)-1 Tcw]
    """

    Rwc = np.linalg.inv(Rcw)
    twc = -np.dot(Rwc, tcw)

    return twc, Rwc


def inter_camera_tR(twc1, Rwc1, tc2w, Rc2w):
    """ Find translation and rotation parameter between camera coordinates.
        Using translation and rotation parameters of Camera-1 to World
        coordinates and those of World to Camera-2 coordinate, finds the
        effective translation and rotation parameters from Camera-1 to Camera-2
        coordinate system.

    Parameters
    ----------
    twc1: numpy array of shape (3,1)
        Translation parameter of Camera-1 coordinate to World.
    Rwc1: numpy array of shape (3,3)
        Rotation parameter of Camera-1 coordinate to World.
    tc2w: numpy array of shape (3,1)
        Translation parameter of World to Camera-2 coordinate.
    Rc2w: numpy array of shape (3,3)
        Rotation parameter of World to Camera-2 coordinate.

    Returns
    -------
    Rc2c1: numpy array of shape (3,3)
        Rotation parameter of Camera-1 to Camera-2 coordinate.
    tc2c1: numpy array of shape (3,1)
        Translation parameter of Camera-1 to Camera-2 coordinate.
    """
    Rc2c1 = np.matmul(Rc2w, Rwc1)
    tc2c1 = np.matmul(Rc2w, twc1) + tc2w

    return tc2c1, Rc2c1

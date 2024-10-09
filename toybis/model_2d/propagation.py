"""
Set of functions to move between different reference frames
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

def _icrs2comrs(ra,dec):#,parallax,pmra,pmdec,satellite_efimerides,relativistic_factors):
    """
    This function has to account for two things:
    1) proper motion and parallactic effects due to 
        the satellite orbit around the Earth and Sun
    2) Aberration and light bending due to special
        and general relativity, respectively
    """
    #TO-DO: for now, this function is left empty

    phi_c = ra
    lambda_c = dec

    return phi_c,lambda_c


def _comrs2fovrs(phi_c,lambda_c,rx_at,ry_at,angle_at):
    """
    This function accounts for the astrometric attitude 
    at each exposure and consists of a rotation.

    Input:
        - phi_c: longitude in CoMRS (in radiant)
        - lambda_c: latitude in CoMRS (in radiant)
        - rx_at: first component of the rotation axis
        - ry_at: second component of the rotation axis
        - angle_at: rotation angle (in radiant)

    Output:
        - eta: longitude in FoVRS (in radiant)
        - zeta: latitude in FoVRS (in radiant)

    NOTE: the third component of the rotation axis, 
    rz_at, is calculated on the fly to impose that the
    quaternion has module 1.
    """
    #compute the third component of the rotation axis imposing modulus 1
    if (rx_at**2+ry_at**2)>1:
        #To-Do: either FIX THIS or constrain the gradient descent
        raise ValueError("There is an issue with your definition of quaternion.")
    rz_at = np.sqrt(1-rx_at**2-ry_at**2)

    #construct the rotator (FOR NOW, WE RELY ON SCIPY)
        #since we already ensured that the modulus is one:
    q_inv = R.from_quat([-rx_at*np.sin(angle_at),
                         -ry_at*np.sin(angle_at),
                         -rz_at*np.sin(angle_at),
                         np.cos(angle_at)])
    #u = q_inv*v*q

    #construct the vector to be rotated
    v = np.array([np.cos(phi_c)*np.cos(lambda_c),
                  np.sin(phi_c)*np.cos(lambda_c),
                  np.sin(lambda_c)])

    #rotate
    u = q_inv.apply(v)

    #obtain new angles
    eta = np.arctan(u[1]/u[0])
    zeta = np.arcsin(u[2])

    return eta,zeta

#TO_DO: define _fovrs2fprs (gnomonic projection) and _fprs2drs (image deformation)
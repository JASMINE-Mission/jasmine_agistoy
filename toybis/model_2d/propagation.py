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


def _comrs2fovrs_fromquat(phi_c,lambda_c,rx_at,ry_at,angle_at):#,parallax,pmra,pmdec
    """
    This function accounts for the astrometric attitude 
    at each exposure and consists of a rotation.

    Input:
        - phi_c: longitude in CoMRS (in radiant)
        - lambda_c: latitude in CoMRS (in radiant)
        - rx_at: first component of the rotation axis
        - ry_at: second component of the rotation axis
        - angle_at: TOTAL rotation angle (in radiant)

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
        #A) since we already ensured that the modulus is one, the inverse is the conjugate
        #B) we divide the angle by 2 because this way of rotating produces a rotation
        #of twice the angle procured
    q_inv = R.from_quat([-rx_at*np.sin(angle_at/2),
                         -ry_at*np.sin(angle_at/2),
                         -rz_at*np.sin(angle_at/2),
                         np.cos(angle_at/2)])

    #construct the vector to be rotated
    v = np.array([np.cos(phi_c)*np.cos(lambda_c),
                  np.sin(phi_c)*np.cos(lambda_c),
                  np.sin(lambda_c)])

    #rotate
    u = q_inv.apply(v)

    #obtain new angles
        #NOTE: we can safely use arctan2 like this because 
        # latitude is always between -Pi/2 and Pi/2 and
        # thus its cosine is always positive
        #NOTE2: this will return angles between -Pi and Pi
    eta = np.arctan2(u[1],u[0])
    zeta = np.arcsin(u[2])
        #TO-DO: rotate proper motions as well

    return eta,zeta


def _comrs2fovrs(phi_c,lambda_c,pt_ra,pt_dec,pt_rot):
    """
    This function accounts for the astrometric attitude 
    at each exposure and consists of a rotation.

    Input:
        - phi_c: longitude in CoMRS (in radiant)
        - lambda_c: latitude in CoMRS (in radiant)
        - pt_ra: right ascension of the centre of the FoV (in radiant)
        - pt_dec: declination of the centre of the FoV (in radiant)
        - pt_rot: TOTAL rotation angle, counter-clockwise (in radiant)

    Output:
        - eta: longitude in FoVRS (in radiant)
        - zeta: latitude in FoVRS (in radiant)
        
        #rotation matrix
    rot = np.array([[ca*cd,-sa*cr+ca*sd*sr,sa*sr+ca*sd*cr],
                     [sa*cd,ca*cr+sa*sd*sr,-ca*sr+sa*sd*cr],
                     [-sd,cd*sr,cd*cr]])
    rot_trans = np.array([[ca*cd,sa*cd,-sd],
                     [-sa*cr+ca*sd*sr,ca*cr+sa*sd*sr,cd*sr],
                     [sa*sr+ca*sd*cr,-ca*sr+sa*sd*cr,cd*cr]])
    
        #vector to rotate (in CoMRS)
    v = np.array([np.cos(phi_c)*np.cos(lambda_c),
                  np.sin(phi_c)*np.cos(lambda_c),
                  np.sin(lambda_c)])
    """
    #trigonometry
    ca = np.cos(pt_ra)
    sa = np.sin(pt_ra)
    cd = np.cos(-pt_dec)
    sd = np.sin(-pt_dec)
    cr = np.cos(pt_rot)
    sr = np.sin(pt_rot)
    cp = np.cos(phi_c)
    sp = np.sin(phi_c)
    cl = np.cos(lambda_c)
    sl = np.sin(lambda_c)
    #rotated vector
    u0 = cp*cl*ca*cd + sp*sl*sa*cd - sl*sd
    u1 = cp*cl*(ca*sd*sr-sa*cr) + sp*sl*(ca*cr+sa*sd*sr) + sl*cd*sr
    u2 = cp*cl*(sa*sr+ca*sd*cr) + sp*sl*(sa*sd*cr-ca*sr) + sl*cd*cr

    #obtain new angles
        #NOTE: we can safely use arctan2 like this because 
        # latitude is always between -Pi/2 and Pi/2 and
        # thus its cosine is always positive
        #NOTE2: this will return angles between -Pi and Pi
    eta = np.arctan2(u1,u0)
    zeta = np.arcsin(u2)

        #TO-DO: rotate proper motions as well
    return eta,zeta

#TO-DO: define _fovrs2fprs (gnomonic projection) and _fprs2drs (image deformation)
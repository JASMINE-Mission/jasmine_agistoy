"""
Set of functions to move between different reference frames
"""
import jax.numpy as jnp
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
        - phi_c: longitude in CoMRS (in radian)
        - lambda_c: latitude in CoMRS (in radian)
        - rx_at: first component of the rotation axis
        - ry_at: second component of the rotation axis
        - angle_at: TOTAL rotation angle (in radian)

    Output:
        - eta: longitude in FoVRS (in radian)
        - zeta: latitude in FoVRS (in radian)

    NOTE: the third component of the rotation axis, 
    rz_at, is calculated on the fly to impose that the
    quaternion has module 1.
    """
    #compute the third component of the rotation axis imposing modulus 1
    if (rx_at**2+ry_at**2)>1:
        #To-Do: either FIX THIS or constrain the gradient descent
        raise ValueError("There is an issue with your definition of quaternion.")
    rz_at = jnp.sqrt(1-rx_at**2-ry_at**2)

    #construct the rotator (FOR NOW, WE RELY ON SCIPY)
        #A) since we already ensured that the modulus is one, the inverse is the conjugate
        #B) we divide the angle by 2 because this way of rotating produces a rotation
        #of twice the angle procured
    q_inv = R.from_quat([-rx_at*np.sin(angle_at/2),
                         -ry_at*np.sin(angle_at/2),
                         -rz_at*np.sin(angle_at/2),
                         np.cos(angle_at/2)])

    #construct the vector to be rotated
    v = jnp.array([np.cos(phi_c)*np.cos(lambda_c),
                  np.sin(phi_c)*np.cos(lambda_c),
                  np.sin(lambda_c)])

    #rotate
    u = q_inv.apply(v)

    #obtain new angles
        #NOTE: we can safely use arctan2 like this because 
        # latitude is always between -Pi/2 and Pi/2 and
        # thus its cosine is always positive
        #NOTE2: this will return angles between -Pi and Pi
    eta = jnp.arctan2(u[1],u[0])
    zeta = jnp.arcsin(u[2])

    return eta,zeta


def _comrs2fovrs(phi_c,lambda_c,pt_ra,pt_dec,pt_rot):
    """
    This function accounts for the astrometric attitude 
    at each exposure and consists of a rotation.

    Input:
        - phi_c: longitude in CoMRS (in radian)
        - lambda_c: latitude in CoMRS (in radian)
        - pt_ra: right ascension of the centre of the FoV (in radian)
        - pt_dec: declination of the centre of the FoV (in radian)
        - pt_rot: TOTAL rotation angle, counter-clockwise (in radian)

    Output:
        - eta: longitude in FoVRS (in radian)
        - zeta: latitude in FoVRS (in radian)
        
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
    ca = jnp.cos(pt_ra)
    sa = jnp.sin(pt_ra)
    cd = jnp.cos(-pt_dec)
    sd = jnp.sin(-pt_dec)
    cr = jnp.cos(pt_rot)
    sr = jnp.sin(pt_rot)
    cp = jnp.cos(phi_c)
    sp = jnp.sin(phi_c)
    cl = jnp.cos(lambda_c)
    sl = jnp.sin(lambda_c)
    #rotated vector
    u0 = cp*cl*ca*cd + sp*cl*sa*cd - sl*sd
    u1 = cp*cl*(ca*sd*sr-sa*cr) + sp*cl*(ca*cr+sa*sd*sr) + sl*cd*sr
    u2 = cp*cl*(sa*sr+ca*sd*cr) + sp*cl*(sa*sd*cr-ca*sr) + sl*cd*cr

    #obtain new angles
        #NOTE: we can safely use arctan2 like this because 
        # latitude is always between -Pi/2 and Pi/2 and
        # thus its cosine is always positive
        #NOTE2: this will return angles between -Pi and Pi
    eta = jnp.arctan2(u1,u0)
    zeta = jnp.arcsin(u2)

    return eta,zeta

def _fovrs2fprs(eta,zeta,F):
    """
    This function performs the gnomonic projection of the
    FoV spherical coordinates (eta,zeta) and accounts for
    the focal lenght F.
    The resulting x and y cartesian coordinates move, 
    respectively, along the eta and zeta axis. 
    The equations implemented here correspond to the
    general gnomonic projection. It assumes
    that the telescope DOES NOT invert the images (TBD).
    In other words, stars at positive eta (zeta) will get
    a positive x (y) value. 

    Input:
        - eta: longitude in FoVRS (in radian)
        - zeta: latitude in FoVRS (in radian)
        - F: Focal lenght (converts units of radians 
                to physical units on the focal plane)

    Output:
        - x_f: x-coordinate in the FPRS (units given by F)
        - y_f: y-coordinate in the FPRS (units given by F)
    """

    x_f = F*jnp.tan(eta)
    y_f = F*jnp.tan(zeta)/jnp.cos(eta)

    return x_f,y_f


def _comrs2fprs(phi_c,lambda_c,pt_ra,pt_dec,pt_rot,F):
    """
    This function performs the gnomonic projection of the
    CoMRS spherical coordinates (phi_c,lambda_c) after
    properly considering the attitude and the focal lenght F.
    It by-passes the FoVRS coordinates.
    The resulting x and y cartesian coordinates move, 
    respectively, along the eta and zeta axis (FoVRS). 
    The equations implemented here correspond to the
    gnomonic projection at the equator. It assumes
    that the telescope DOES NOT invert the images (TBD).
    In other words, stars at positive eta (zeta) will get
    a positive x (y) value. 

    Input:
        - phi_c: longitude in CoMRS (in radian)
        - lambda_c: latitude in CoMRS (in radian)
        - pt_ra: right ascension of the centre of the FoV (in radian)
        - pt_dec: declination of the centre of the FoV (in radian)
        - pt_rot: TOTAL rotation angle, counter-clockwise (in radian)
        - F: Focal lenght (converts units of radians 
                to physical units on the focal plane)

    Output:
        - x_f: x-coordinate in the FPRS (units given by F)
        - y_f: y-coordinate in the FPRS (units given by F)
    """
    #trigonometry
    cd = jnp.cos(pt_dec)
    sd = jnp.sin(pt_dec)
    cr = jnp.cos(pt_rot)
    sr = jnp.sin(pt_rot)
    cl = jnp.cos(lambda_c)
    sl = jnp.sin(lambda_c)
    cpr = jnp.cos(phi_c - pt_ra)
    spr = jnp.sin(phi_c-pt_ra)

    #true angular distance between satellite pointing and source
    cosr = sd*sl + cd*cl*cpr
    #projected (gnomonic) angles
    zeta_proj = (sl*cd-sd*cl*cpr)/cosr
    eta_proj = + (cl*spr)/cosr

    #rotate and scale
    x_f = F * (cr*eta_proj + sr*zeta_proj)
    y_f = F * (cr*zeta_proj - sr*eta_proj)

    return x_f,y_f

#TO-DO: define _fprs2drs (image deformation: it should include a hardcoded rotation of each detector w.r.t. the FPRS)
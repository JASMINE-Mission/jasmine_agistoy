"""
Set of functions to move between different reference frames
"""
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation as R

__all__ = [
    '_icrs2comrs',
    '_comrs2fovrs',
    '_fovrs2fprs',
    '_comrs2fprs',
    '_comrs2fovrs_fromquat',
]

#speed of light in m/s
_ERFA_CMPS = 299792458.0 

# Schwarzschild radius of the Sun (au)
_ERFA_SRS = 1.97412574336e-8

def _icrs2comrs(ra:float,dec:float,time:float,_jamsime_ephemeris:"(float,float,float,float,float,float)",
                _Msun: float=1,_Mjupyter: float=0.0009545942339693249, _limiter = 1e-8) -> "(float,float)":
    """
    This function accounts for two things:
    1) (TO-DO) proper motion and parallactic effects due to 
        the satellite orbit around the Earth and Sun
    2) stellar aberration and light bending due to special
        and general relativity, respectively

    Inputs:
        - ra: true right ascension of the source in ICRS (radians)
        - dec: true declination of the source in ICRS (radians)
        - time: time of observation in TCB (Modified Julian Day)
        - _jasmine_ephemeris: array containing the position and velocities
             of the satellite's centre of Mass in the Barycentric Celestial 
             Reference System. Positions in AU, velocities in m/s.
        - _Msun: mass of the Sun in units of Solar masses.

    Outputs:
        - phi_c: longitudinal coordinate in the CoMRS (radians)
        - lambda_c: latitudinal coordinate in the CoMRS (radians)

    """
    #TO-DO: for now, this function is left empty
        #add parallax,pmra,pmdec:

    #compute geometric direction in ICRS
    q_geo = jnp.array([
        jnp.cos(ra) * jnp.cos(dec),
        jnp.sin(ra) * jnp.cos(dec),
        jnp.sin(dec)
    ])

    #normalise satellite velocity to speed of light
    satvel = _jamsime_ephemeris[3:]
        #observer's velocity with respect to the Solar System barycenter in units of c
    satvelnorm = satvel/_ERFA_CMPS
        #reciprocal of Lorenz factor
    beta = jnp.sqrt(jnp.sum(satvel**2,axis=0))/_ERFA_CMPS
    bm1 = np.sqrt(1-np.linalg.norm(beta)**2)

    #measure distance from Sun in AU
    satpos = _jamsime_ephemeris[:3]
    distance_from_sun = jnp.sqrt(jnp.sum(satpos**2,axis=0))

    #apply light deflection by the Sun to obtain apparent direction
    qdqpe = (q_geo * (q_geo + satpos)).sum(axis=0, keepdims=True)
    w = _Msun * _ERFA_SRS / distance_from_sun / jnp.clip(qdqpe, a_min=_limiter)

    eq = jnp.cross(satpos, q_geo)
    peq = jnp.cross(q_geo, eq)

    q_apa = q_geo + w * peq

    norm = jnp.sqrt(jnp.sum(q_apa**2, axis=0, keepdims=True))

    q_apa = q_apa / norm

    #apply stellar aberration to update apparent direction
    pdv = jnp.dot(q_apa,satvelnorm)
    w1 = 1.0 + pdv / (1.0 + bm1)
    w2 = _ERFA_SRS / distance_from_sun
    q_apa = bm1 * q_apa \
        + w1 * satvelnorm + w2 * (satvelnorm + pdv * q_apa)
    norm = jnp.sqrt(jnp.sum(q_apa ** 2, axis=0, keepdims=True))
    q_apa = q_apa / norm

    #return to spherical angles
    phi_c = jnp.arctan2(q_apa[1],q_apa[0])
    lambda_c = jnp.arctan2(q_apa[2],jnp.sqrt(q_apa[0]**2 + q_apa[1]**2))

    return phi_c,lambda_c


def _comrs2fovrs_fromquat(phi_c:float,lambda_c:float,rx_at:float,ry_at:float,angle_at:float) -> "(jnp.ndarray,jnp.ndarray)":
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
    zeta = jnp.arctan2(u[2],jnp.sqrt(u[0]**2 + u[1]**2)) #equation 12 of Lindegren+12

    return eta,zeta


def _comrs2fovrs(phi_c:float,lambda_c:float,pt_ra:float,pt_dec:float,pt_rot:float) -> "(jnp.ndarray,jnp.ndarray)":
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
    zeta = jnp.arctan2(u2,jnp.sqrt(u0**2 + u1**2)) #equation 12 of Lindegren+12

    return eta,zeta

def _fovrs2fprs(eta:float,zeta:float,F:float) -> "(jnp.ndarray,jnp.ndarray)":
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


def _comrs2fprs(phi_c:float,lambda_c:float,pt_ra:float,pt_dec:float,pt_rot:float,F:float) -> "(jnp.ndarray,jnp.ndarray)":
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
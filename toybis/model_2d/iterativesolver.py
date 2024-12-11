"""
Set of functions that are used for the bloc iterative astrometric solution
"""

from jax import vmap,jacrev,numpy as jnp
from jax.scipy.linalg import cho_factor,cho_solve
import matplotlib.pyplot as plt

__all__ = [
    '_Jacobian_autodiff',
    'update_source_inner',
    'update_attitude_inner',
    'update_calibration_inner',
    'update_source',
    'update_attitude',
    'update_calibration',
    'iterate_source',
    'iterate_attitude',
    'iterate_calibration'
]

def plot_residuals(r,filename):
    fig = plt.figure()
    plt.plot(r)
    fig.savefig(filename)
    plt.clf() 
    plt.close()

def _Jacobian_autodiff(model,argnum,axis):
    """
    Generate the mapping function that computes the jacobian of the function "model"
    with respect to the especified variable (given by argnum). The axis has to be
    a tuple with "None" at the location of the axis that is going to be mapped, 
    and zero in all other positions.

    Input:
        - model: a function that generates predictions from the model parameters.
            The inputs of model are always (source,attitude,calibration,time), in that order
        - argnum: 
            - 0 to derivate wrt the source parameters
            - 1 to derivate wrt the attitude parameters
            - 2 to derivate wrt the calibration parameters
        - axis:
            - which axis to iterate over

    Output:
        - The jacobian matrix of the bloc.
    """
    
    return vmap(jacrev(model,argnums=(argnum)),axis)


def update_source_inner(src, att, cal, ephemeris,obs,priors,sig_prior,model,telescope_metaparams,_min_nobs=10,_store_residuals=True):
    """
    Solver linear problem to find the "optimal" source
    parameters of ONE source.

    Input:
        src: source parameters of the source of interest (1+num_srcparams)
            - First column: source_id
            - Second to last columns: source parameters
        att: attitude parameters (M x (2+num_attparams))
            - First column: exposure_id
            - Second column: time of exposure
            - Third to last columns: attitude parameters
        cal: calibration parameters (P x (1+num_calparams))
            - First column: calibration_unit_id
            - Second to last columns: calibration parameters
        ephemeris: ephemeris of the satellite (M x 6):
            - Positions (in AU) and velocities (in m/s) for
            each observation.
        obs: array of observations ((2*num_obs)x5).
            Must contain at least the following columns: 
            - associated source_id
            - associated exposure_id
            - associated calibration_unit_id
            - associated detector_id
            - axis (either 0 or 1, or None existent)
            - value of the observation (can be one column or two)
            - value of the uncertainty in the observation (same as observations)
        priors: prior on the source parameters (N x (1+num_srcparams))
        sig_prior: uncertainty on the prior on the source parameters 
                        (N x (1+num_srcparams))
        model: forward modeling function that predicts the observations
            from all the model parameters.
            Inputs (at call) in order:
                - source parameters
                - attitude parameters
                - calibration parameters
                - time
                - ephemeris
                - detector ID
        telescope_metaparams: dictionary containing the relevant
            metaparameters of the mission. It should contain:
                - 'F': nominal focal length
                - 'sX': nominal pixel size along [same units as F]
                - 'sY': nominal pixel size across [same units as F]
                - 'kappa0': position in pixels of the first usable column
                - 'mu0': position in pixels of the first usable row
                - 'kappaC': horizontal pixel position of the centre 
                    of the detector in the DRS.
                - 'muC': vertical pixel position of the centre 
                    of the detector in the DRS.
                - 'nCol': number of usable columns in one detector
                - 'nRow': number of usable rows in one detector
                - 'x0'&'y0': FPRS coordinates of the nominal foot point of
                        the optical telescope axis on the focal plane
                        [same units as F]
                - 'xC': array containing the FPRS x-coordinate of the centre
                    of each detector [same units as F]
                - 'yC': array containing the FPRS y-coordinate of the centre
                    of each detector [same units as F]
                - 'R': array containing the orientation in FPRS coordinates
                    of each detector
                - 'srcid_index': number of the column in the observation array
                    containing the source_id
                - 'expid_index': number of the column in the observation array
                    containing the exposure_id
                - 'calid_index': number of the column in the observation array
                    containing the calibration_unit_id
                - 'detid_index': number of the column in the observation array
                    containing the detector_id
                - 'axis_index': number of the column in the observation array
                    containing the axis information (0 - horizontal, 1 - vertical)
                - 'obs_index': number of the column in the observation array
                    containing the actual measurement
                - 'obssig_index': number of the column in the observation array
                    containing the uncertainty measurement
        detector_metaparams: array containing, in each row, the following values
                - First column: detector ID
                - Second column: FPRS x-coordinate of the centre of the detector
                - Third column: FPRS y-coordinate of the centre of the detector
                - Forth to Seventh columns: components of the rotation matrix
                    that defines the orientation in FPRS coordinates
                    of the detector as [R00 R10 R01 R11] = [[R00, R01],
                                                            [R10, R11]]
                    The corresponding rotation angle is theta = arctan2(R01,R00)
                    (right handed || counter clock-wise)
        _min_nobs: minimum number of observations to attempt a solution.
            Below this number, the passed values for the guess are returned. 

        where N is the number of sources that we want to solve for,
        M is the total number of exposures taken (each one at time T), 
        and P is the number of calibration units used.

    Returns:
        The updated source parameters of all sources.
    """
    
    #search for the source of interest
        #locate observations related to this source id
    observations = obs[obs[:, telescope_metaparams["srcid_index"]] == src[0]]
        #locate the parameters of the exposures when this source id was observed
    mask_att = jnp.isin(att[:,0],observations[:,telescope_metaparams["expid_index"]])
    exposures = att[mask_att]
        #filter also the ephemeris to be used
    ephis = ephemeris[mask_att]
        #locate the parameters of the calibration units when this source id was observed
    mask_cal = jnp.isin(cal[:,0],observations[:,telescope_metaparams["calid_index"]])
    calibrations = cal[mask_cal]

    #if we do not have enough observations, do not even try
    if len(exposures)<_min_nobs:
        return src[1:]


    #prepare the Jacobian matrix wrt source parameters
    _Jds = jacrev(model,argnums=(0))

    c_ = []
    Ds_ = []
    o_ = []
    #since the shape of exposures and calibrations arrays are not the same, need to iterate
    for i,a in enumerate(exposures):
        # at this point, we should have only ONE observation
        oo_ = observations[observations[:,telescope_metaparams["expid_index"]]==a[0]]
        if telescope_metaparams["axis_index"] is None:
            #observations are given in two columns
            oo_data = oo_[:,telescope_metaparams["obs_index"]:telescope_metaparams["obs_index"]+2]
            if len(oo_data)!=1:
                raise ValueError("Got an unexpected number of observations! Expected 1 but got {}.".format(int(len(oo_data))))
            o_.append(oo_data[0])
        else:
            #observations are given in one column
            oo_data = oo_[:,telescope_metaparams["obs_index"]]
            if len(oo_data)!=2:
                raise ValueError("Got an unexpected number of observations! Expected 1 but got {}.".format(int(len(oo_data)/2)))
            o_.append(oo_data)
        #store the value of the detector where this source has been observed (one per pair)
        detector_id = oo_[0,telescope_metaparams["detid_index"]]

        #find the right calibration unit
        mask_c = jnp.isin(calibrations[:,0],oo_[:,telescope_metaparams["expid_index"]])
            #There should have only one calibration unit c_
        if jnp.sum(mask_c)>1:
            raise ValueError("Got more than one calibration unit per exposure!\n"+\
                            "\tSource ID: {}\n\tExposure ID: {}".format(src[0],a[0]))
        cc_ = calibrations[mask_c][0]
        
        c_.append(model(src[1:],a[2:],cc_[1:],a[1],ephis[i],detector_id))
        Ds_.append(jnp.hstack(_Jds(src[1:],a[2:],cc_[1:],a[1],ephis[i],detector_id)))

    c_arr = jnp.array(c_)
    Ds_arr = jnp.array(Ds_)
    o_arr = jnp.array(o_)

    c = jnp.vstack((c_arr[:,:int(c_arr.shape[1]/2)],c_arr[:,int(c_arr.shape[1]/2):]))
    Ds = jnp.vstack((Ds_arr[:,:int(Ds_arr.shape[1]/2)],Ds_arr[:,int(Ds_arr.shape[1]/2):]))
    o = jnp.vstack((o_arr[:,:int(o_arr.shape[1]/2)],o_arr[:,int(o_arr.shape[1]/2):]))

    if _store_residuals:
        plot_residuals(o - c,"{}/source_residuals_{}.png".format(telescope_metaparams["residuals_folder"],src[0]))
    
    if priors is None:
        #compute the normal matrix as usual 
        N = Ds.T @ Ds
        b = Ds.T @ (o - c)
    else:
        #check that we have also the corresponding uncertainties
        if sig_prior is None:
            raise ValueError("If the priors are given, you need to provide also their uncertainties!")
        else:
            #select priors on the parameters of the source of interest
            priors_ = priors[priors[:,0]==src[0]][0]
            sig_prior_ = sig_prior[sig_prior[:,0]==src[0]][0]

        #preprare observational covariance matrix
        if telescope_metaparams["axis_index"] is None:
                #uncertainties given in two columns
            s =  observations[:, telescope_metaparams["obssig_index"]:telescope_metaparams["obssig_index"]+2]
            #TO-DO: FIX THIS MESS!
            s = jnp.vstack((s[:,:int(s.shape[1]/2)],s[:,int(s.shape[1]/2):])).flatten()
        else:
            s0 =  observations[observations[:,telescope_metaparams["axis_index"]]==0, telescope_metaparams["obssig_index"]]
            s1 = observations[observations[:,telescope_metaparams["axis_index"]]==1, telescope_metaparams["obssig_index"]]
            s = jnp.hstack((s0,s1))
        S = jnp.diag(s**(-2))
        #print("Shape of s: ",s.shape)
        #print("Shape of S: ",S.shape)
        #print("Shape of Ds: ",Ds.shape)

        aux = jnp.diag(sig_prior_[1:]**(-2))

        #print("Shape of Sigma_prior: ",aux.shape)
        #print("Shape of matmul: ",(jnp.matmul(aux,(priors_[1:]-src[1:]))).shape)
        #print("Shape of other part: ",(priors_[1:]-src[1:]).shape)
        #print("Shape of first half of b: ",(Ds.T @ S @ (o - c)).shape)
        
        #compute the normal matrix accounting for a gaussian prior
        N = Ds.T @ S @ Ds + jnp.diag(sig_prior_[1:]**(-2))
        b = Ds.T @ S @ (o - c) + jnp.matmul(aux,(priors_[1:]-src[1:])).reshape((len(src[1:]),1))
        
    
    #print("Shape of N: ",N.shape)
    #print("Shape of b: ",b.shape)
    #solve for the difference between current (assumed) parameters and optimal parameters
    cfac = cho_factor(N)
    delta = cho_solve(cfac, b)
    #print("Shape of delta: ",delta.shape)

    #DEBUG
    if src[0]%500==0:
        print(src[0])
        print(src[1:].shape,delta.shape,delta.flatten())
    
    return src[1:] + delta.flatten()


def update_source(src, att, cal,ephemeris,obs,priors,sig_prior,model,telescope_metaparams,detector_metaparams,_min_nobs=10):
  ''' Updates of the source parameters

    Input:
        src: source parameters (N x (1+num_srcparams))
            - First column: source_id
            - Second to last columns: source parameters
        att: attitude parameters (M x (2+num_attparams))
            - First column: exposure_id
            - Second column: time of exposure
            - Third to last columns: attitude parameters
        cal: calibration parameters (P x (1+num_calparams))
            - First column: calibration_unit_id
            - Second to last columns: calibration parameters
        ephemeris: ephemeris of the satellite (M x 6):
            - Positions (in AU) and velocities (in m/s) for
            each observation.
        obs: array of observations ((2*num_obs)x5)
            Must contain at least the following columns: 
            - associated source_id
            - associated exposure_id
            - associated calibration_unit_id
            - associated detector_id
            - axis (either 0 or 1)
            - value of the observation
            - value of the uncertainty in the observation
        priors: prior on the source parameters (N x (1+num_srcparams))
        sig_prior: uncertainty on the prior on the source parameters 
                    (N x (1+num_srcparams))
        model: forward modeling function that predicts the observations
            from all the model parameters.
            Inputs (at call) in order:
                - source parameters
                - attitude parameters
                - calibration parameters
                - time
                - ephemeris
                - detector ID
        telescope_metaparams: dictionary containing the relevant
            metaparameters of the mission. It should contain:
                - 'F': nominal focal length
                - 'sX': nominal pixel size along [same units as F]
                - 'sY': nominal pixel size across [same units as F]
                - 'kappa0': position in pixels of the first usable column
                - 'mu0': position in pixels of the first usable row
                - 'kappaC': horizontal pixel position of the centre 
                    of the detector in the DRS.
                - 'muC': vertical pixel position of the centre 
                    of the detector in the DRS.
                - 'nCol': number of usable columns in one detector
                - 'nRow': number of usable rows in one detector
                - 'x0'&'y0': FPRS coordinates of the nominal foot point of
                        the optical telescope axis on the focal plane
                        [same units as F]
                - 'xC': array containing the FPRS x-coordinate of the centre
                    of each detector [same units as F]
                - 'yC': array containing the FPRS y-coordinate of the centre
                    of each detector [same units as F]
                - 'R': array containing the orientation in FPRS coordinates
                    of each detector
                - 'srcid_index': number of the column in the observation array
                    containing the source_id
                - 'expid_index': number of the column in the observation array
                    containing the exposure_id
                - 'calid_index': number of the column in the observation array
                    containing the calibration_unit_id
                - 'detid_index': number of the column in the observation array
                    containing the detector_id
                - 'axis_index': number of the column in the observation array
                    containing the axis information (0 - horizontal, 1 - vertical)
                - 'obs_index': number of the column in the observation array
                    containing the actual measurement
                - 'obssig_index': number of the column in the observation array
                    containing the uncertainty measurement
        detector_metaparams: array containing, in each row, the following values
                - First column: detector ID
                - Second column: FPRS x-coordinate of the centre of the detector
                - Third column: FPRS y-coordinate of the centre of the detector
                - Forth to Seventh columns: components of the rotation matrix
                    that defines the orientation in FPRS coordinates
                    of the detector as [R00 R10 R01 R11] = [[R00, R01],
                                                            [R10, R11]]
                    The corresponding rotation angle is theta = arctan2(R01,R00)
                    (right handed || counter clock-wise)
        _min_nobs: minimum number of observations to attempt a solution.
            Below this number, the passed values for the guess are returned. 

        where N is the number of sources that we want to solve for,
        M is the total number of exposures taken (each one at time T), 
        and P is the number of calibration units used.

    Returns:
    The updated source parameters of all sources.
  '''
  #loop through each source and find their optimal source parameters
  return jnp.vstack([update_source_inner(src_, att, cal,ephemeris,obs,priors,sig_prior,model,telescope_metaparams,detector_metaparams,_min_nobs) for src_ in src])


def update_attitude_inner(src, att,cal,ephemeris,obs,model,telescope_metaparams,detector_metaparams,_min_nstars=100):
    """
    Solver linear problem to find the "optimal" attitude
    parameters of ONE exposure.

    Input:
        src: source parameters (N x (1+num_srcparams))
            - First column: source_id
            - Second to last columns: source parameters
        att: attitude parameters of the exposure of interest (2+num_attparams)
            - First column: exposure_id
            - Second column: time of exposure
            - Third to last columns: attitude parameters
        cal: calibration parameters (P x (1+num_calparams))
            - First column: calibration_unit_id
            - Second to last columns: calibration parameters
        ephemeris: ephemeris of the satellite (1 x 6):
            - Position (in AU) and velocity (in m/s) for
            the exposure of interest observation.
        obs: array of observations ((2*num_obs)x5)
            Must contain at least the following columns: 
            - associated source_id
            - associated exposure_id
            - associated calibration_unit_id
            - associated detector_id
            - axis (either 0 or 1)
            - value of the observation
            - value of the uncertainty in the observation
        model: forward modeling function that predicts the observations
            from all the model parameters.
            Inputs (at call) in order:
                - source parameters
                - attitude parameters
                - calibration parameters
                - time
                - ephemeris
                - detector ID
        telescope_metaparams: dictionary containing the relevant
            metaparameters of the mission. It should contain:
                - 'F': nominal focal length
                - 'sX': nominal pixel size along [same units as F]
                - 'sY': nominal pixel size across [same units as F]
                - 'kappa0': position in pixels of the first usable column
                - 'mu0': position in pixels of the first usable row
                - 'kappaC': horizontal pixel position of the centre 
                    of the detector in the DRS.
                - 'muC': vertical pixel position of the centre 
                    of the detector in the DRS.
                - 'nCol': number of usable columns in one detector
                - 'nRow': number of usable rows in one detector
                - 'x0'&'y0': FPRS coordinates of the nominal foot point of
                        the optical telescope axis on the focal plane
                        [same units as F]
                - 'xC': array containing the FPRS x-coordinate of the centre
                    of each detector [same units as F]
                - 'yC': array containing the FPRS y-coordinate of the centre
                    of each detector [same units as F]
                - 'R': array containing the orientation in FPRS coordinates
                    of each detector
                - 'srcid_index': number of the column in the observation array
                    containing the source_id
                - 'expid_index': number of the column in the observation array
                    containing the exposure_id
                - 'calid_index': number of the column in the observation array
                    containing the calibration_unit_id
                - 'detid_index': number of the column in the observation array
                    containing the detector_id
                - 'axis_index': number of the column in the observation array
                    containing the axis information (0 - horizontal, 1 - vertical)
                - 'obs_index': number of the column in the observation array
                    containing the actual measurement
                - 'obssig_index': number of the column in the observation array
                    containing the uncertainty measurement
        detector_metaparams: array containing, in each row, the following values
                - First column: detector ID
                - Second column: FPRS x-coordinate of the centre of the detector
                - Third column: FPRS y-coordinate of the centre of the detector
                - Forth to Seventh columns: components of the rotation matrix
                    that defines the orientation in FPRS coordinates
                    of the detector as [R00 R10 R01 R11] = [[R00, R01],
                                                            [R10, R11]]
                    The corresponding rotation angle is theta = arctan2(R01,R00)
                    (right handed || counter clock-wise)
        _min_nstars: minimum number of sources in the exposure to attempt a solution.
            Below this number, the passed values for the guess are returned. 

    Output:
    updated attitude parameters of this exposure
    """

    #locate the observational data of the sources observed at this exposure
    observations = obs[obs[:, telescope_metaparams["expid_index"]] == att[0]]

    if telescope_metaparams["axis_index"] is None:
        #store the value of the detector where each source has been observed
        detector_ids = observations[:,telescope_metaparams["detid_index"]]
    else:
        #store the value of the detector where each source has been observed (one per pair)
        detector_ids = observations[::2,telescope_metaparams["detid_index"]]
        #locate the source parameters of the sources observed at this exposure
    mask_src = jnp.isin(src[:,0],observations[:,telescope_metaparams["srcid_index"]])
    sources = src[mask_src]

    #if we do not have enough sources in the exposure, do not even try
    if len(sources)<_min_nstars:
        return att[2:]
        
    #detector_ids and sources should have the same number of entries!
    if len(sources)!= len(detector_ids):
        raise ValueError("Got an inconsistent number of sources ({}) ".format(len(sources))+\
                        "and detector IDs ({})!".format(len(detector_ids)))

        #locate the parameters of the calibration unit corresponding to this attitude
        #NOTE: we are guaranteed to have only one! => we select the first item
    mask_cal = jnp.isin(cal[:,0],observations[:,telescope_metaparams["calid_index"]])
    if jnp.sum(mask_cal)!=1:
        raise ValueError("Found an unexpected number of Calibration Units related to this exposure ({})!".format(att[0])+\
                        " Expected 1 but found {}.".format(jnp.sum(mask_cal)))
    calibrations = cal[mask_cal][0]

    #create iterable version of the model
        #for each source we should have only one observation, therefore
        # we iterate along sources AND observations at the same time
    _iterate_src = vmap(model, (0, None, None, None,None,0), 0)
    #prepare the Jacobian matrix wrt attitude parameters
        #the jacobian has to be iterated over all sources
    _Jda = _Jacobian_autodiff(model,1,(0,None, None, None,None,0))

    #compute the predicted observations
    c = jnp.hstack(_iterate_src(sources[:,1:], att[2:],calibrations[1:], att[1],ephemeris[0],detector_ids))
    #create vector of observations to compare to

    if telescope_metaparams["axis_index"] is None: 
        o = jnp.hstack((observations[:, telescope_metaparams["obs_index"]],
                        observations[:, telescope_metaparams["obs_index"]+1]))
    else:
        o0 =  observations[observations[:,telescope_metaparams["axis_index"]]==0, telescope_metaparams["obs_index"]]
        o1 = observations[observations[:,telescope_metaparams["axis_index"]]==1, telescope_metaparams["obs_index"]]
        o = jnp.hstack((o0,o1))


    #compute design matrix
    Da = jnp.vstack(_Jda(sources[:,1:],att[2:],calibrations[1:],att[1],ephemeris,detector_ids))

    plot_residuals(o - c,"Residuals/attitude_residuals_{}.png".format(att[0]))

    #compute normal matrix
    N = Da.T @ Da
    b = Da.T @ (o - c)
    
    ##solve for the difference between current (assumed) parameters and optimal parameters
    cfac = cho_factor(N)
    delta = cho_solve(cfac, b)
    return att[2:] + delta

def update_attitude(src, att,cal,ephemeris,obs,model,telescope_metaparams,detector_metaparams,_min_nstars=100):
  ''' Updates of the attitude parameters

    Input:
        src: source parameters (N x (1+num_srcparams))
            - First column: source_id
            - Second to last columns: source parameters
        att: attitude parameters (M x (2+num_attparams))
            - First column: exposure_id
            - Second column: time of exposure
            - Third to last columns: attitude parameters
        cal: calibration parameters (P x (1+num_calparams))
            - First column: calibration_unit_id
            - Second to last columns: calibration parameters
        ephemeris: ephemeris of the satellite (M x 6):
            - Positions (in AU) and velocities (in m/s) for
            each observation.
        obs: array of observations ((2*num_obs)x5)
            Must contain at least the following columns: 
            - associated source_id
            - associated exposure_id
            - associated calibration_unit_id
            - associated detector_id
            - axis (either 0 or 1)
            - value of the observation
            - value of the uncertainty in the observation
        model: forward modeling function that predicts the observations
            from all the model parameters.
            Inputs (at call) in order:
                - source parameters
                - attitude parameters
                - calibration parameters
                - time
                - ephemeris
                - detector ID
        telescope_metaparams: dictionary containing the relevant
            metaparameters of the mission. It should contain:
                - 'F': nominal focal length
                - 'sX': nominal pixel size along [same units as F]
                - 'sY': nominal pixel size across [same units as F]
                - 'kappa0': position in pixels of the first usable column
                - 'mu0': position in pixels of the first usable row
                - 'kappaC': horizontal pixel position of the centre 
                    of the detector in the DRS.
                - 'muC': vertical pixel position of the centre 
                    of the detector in the DRS.
                - 'nCol': number of usable columns in one detector
                - 'nRow': number of usable rows in one detector
                - 'x0'&'y0': FPRS coordinates of the nominal foot point of
                        the optical telescope axis on the focal plane
                        [same units as F]
                - 'xC': array containing the FPRS x-coordinate of the centre
                    of each detector [same units as F]
                - 'yC': array containing the FPRS y-coordinate of the centre
                    of each detector [same units as F]
                - 'R': array containing the orientation in FPRS coordinates
                    of each detector
                - 'srcid_index': number of the column in the observation array
                    containing the source_id
                - 'expid_index': number of the column in the observation array
                    containing the exposure_id
                - 'calid_index': number of the column in the observation array
                    containing the calibration_unit_id
                - 'detid_index': number of the column in the observation array
                    containing the detector_id
                - 'axis_index': number of the column in the observation array
                    containing the axis information (0 - horizontal, 1 - vertical)
                - 'obs_index': number of the column in the observation array
                    containing the actual measurement
                - 'obssig_index': number of the column in the observation array
                    containing the uncertainty measurement
        detector_metaparams: array containing, in each row, the following values
                - First column: detector ID
                - Second column: FPRS x-coordinate of the centre of the detector
                - Third column: FPRS y-coordinate of the centre of the detector
                - Forth to Seventh columns: components of the rotation matrix
                    that defines the orientation in FPRS coordinates
                    of the detector as [R00 R10 R01 R11] = [[R00, R01],
                                                            [R10, R11]]
                    The corresponding rotation angle is theta = arctan2(R01,R00)
                    (right handed || counter clock-wise)
        _min_nstars: minimum number of sources in the exposure to attempt a solution.
            Below this number, the passed values for the guess are returned. 

        where N is the number of sources that we want to solve for,
        M is the total number of exposures taken (each one at time T), 
        and P is the number of calibration units used.

    Returns:
    The updated attitude parameters at all the exposures.
  '''
  #loop through each exposure and find their optimal attitude parameters
  return jnp.vstack([update_attitude_inner(src, att_,cal,ephemeris[i],obs,model,telescope_metaparams,detector_metaparams,_min_nstars) for i,att_ in enumerate(att)])



def update_calibration_inner(src, att,cal,ephemeris,obs,model,telescope_metaparams,detector_metaparams,_min_nobs=200):
    """
    Solver linear problem to find the "optimal" calibration
    parameters of ONE calibration unit.

    Input:
        src: source parameters (N x (1+num_srcparams))
            - First column: source_id
            - Second to last columns: source parameters
        att: attitude parameters of the exposure of interest (M x (2+num_attparams))
            - First column: exposure_id
            - Second column: time of exposure
            - Third to last columns: attitude parameters
        cal: calibration parameters (1+num_calparams)
            - First column: calibration_unit_id
            - Second to last columns: calibration parameters
        ephemeris: ephemeris of the satellite (1 x 6):
            - Position (in AU) and velocity (in m/s) for
            the exposure of interest observation.
        obs: array of observations ((2*num_obs)x5)
            Must contain at least the following columns: 
            - associated source_id
            - associated exposure_id
            - associated calibration_unit_id
            - associated detector_id
            - axis (either 0 or 1)
            - value of the observation
            - value of the uncertainty in the observation
        model: forward modeling function that predicts the observations
            from all the model parameters.
            Inputs (at call) in order:
                - source parameters
                - attitude parameters
                - calibration parameters
                - time
                - ephemeris
                - detector ID
        telescope_metaparams: dictionary containing the relevant
            metaparameters of the mission. It should contain:
                - 'F': nominal focal length
                - 'sX': nominal pixel size along [same units as F]
                - 'sY': nominal pixel size across [same units as F]
                - 'kappa0': position in pixels of the first usable column
                - 'mu0': position in pixels of the first usable row
                - 'kappaC': horizontal pixel position of the centre 
                    of the detector in the DRS.
                - 'muC': vertical pixel position of the centre 
                    of the detector in the DRS.
                - 'nCol': number of usable columns in one detector
                - 'nRow': number of usable rows in one detector
                - 'x0'&'y0': FPRS coordinates of the nominal foot point of
                        the optical telescope axis on the focal plane
                        [same units as F]
                - 'xC': array containing the FPRS x-coordinate of the centre
                    of each detector [same units as F]
                - 'yC': array containing the FPRS y-coordinate of the centre
                    of each detector [same units as F]
                - 'R': array containing the orientation in FPRS coordinates
                    of each detector
                - 'srcid_index': number of the column in the observation array
                    containing the source_id
                - 'expid_index': number of the column in the observation array
                    containing the exposure_id
                - 'calid_index': number of the column in the observation array
                    containing the calibration_unit_id
                - 'detid_index': number of the column in the observation array
                    containing the detector_id
                - 'axis_index': number of the column in the observation array
                    containing the axis information (0 - horizontal, 1 - vertical)
                - 'obs_index': number of the column in the observation array
                    containing the actual measurement
                - 'obssig_index': number of the column in the observation array
                    containing the uncertainty measurement
        detector_metaparams: array containing, in each row, the following values
                - First column: detector ID
                - Second column: FPRS x-coordinate of the centre of the detector
                - Third column: FPRS y-coordinate of the centre of the detector
                - Forth to Seventh columns: components of the rotation matrix
                    that defines the orientation in FPRS coordinates
                    of the detector as [R00 R10 R01 R11] = [[R00, R01],
                                                            [R10, R11]]
                    The corresponding rotation angle is theta = arctan2(R01,R00)
                    (right handed || counter clock-wise)
        _min_nobs: minimum number of observations in the calibration unit to attempt a solution.
            Below this number, the passed values for the guess are returned. 

    Output:
    updated calibration parameters of this calibration unit
    """

    #locate the observational data of the sources observed with this calibration unit
    observations = obs[obs[:, telescope_metaparams["calid_index"]] == cal[0]]

    #if we do not have enough sources in the exposure, do not even try
    if len(observations)<_min_nobs:
        return cal[1:]

    #create iterable version of the model
    _iterate_att = vmap(model, (None,0,None,0,0,0), 0) #(None, 0,None, 0,0)

    #prepare the Jacobian matrix wrt attitude parameters
        #the jacobian has to be iterated over all sources
    _Jdc = _Jacobian_autodiff(model,2,(None,0,None,0,0,0))

        #locate the source parameters of the sources observed with this calibration unit
    mask_src = jnp.isin(src[:,0],observations[:,telescope_metaparams["srcid_index"]])
    sources = src[mask_src]
        #locate the attitude parameters of the sources observed with this calibration unit
    mask_att = jnp.isin(att[:,0],observations[:,telescope_metaparams["expid_index"]])
    attitude = att[mask_att]
    ephis = ephemeris[mask_att]

    c_ = []
    Dc_ = []
    o_ = []
    #to preserve the ordering, we must be careful
        #for each source, iterate over the exposures where it is observed
    for s in sources:
        #select relevant observations
        oo_ = observations[observations[:,telescope_metaparams["srcid_index"]]==s[0]]

        if telescope_metaparams["axis_index"] is None:
            #store the value of the detector where each source has been observed
            detector_ids = oo_[:,telescope_metaparams["detid_index"]]
        else:
            #store the value of the detector where each source has been observed (one per pair)
            detector_ids = oo_[::2,telescope_metaparams["detid_index"]]
        #select only the exposures where that source is observed
        mask_a = jnp.isin(attitude[:,0],oo_[:,1])
        a_ = attitude[mask_a]

            #detector_ids and a_ should have the same number of entries!
        if len(a_)!= len(detector_ids):
            raise ValueError("Got an inconsistent number of exposures ({}) ".format(len(a_))+\
                            "and detector IDs ({})!".format(len(detector_ids)))
                            
        #compute the predicted observations
        c_.append(jnp.column_stack(_iterate_att(s[1:],a_[:,2:],cal[1:],a_[:,1],ephis[mask_a],detector_ids)))
        #compute design matrix
        Dc_.append(jnp.column_stack(_Jdc(s[1:],a_[:,2:],cal[1:],a_[:,1],ephis[mask_a],detector_ids)))


    #unpack
    c_arr = jnp.vstack(c_)
    c = jnp.vstack((c_arr[:,:int(c_arr.shape[1]/2)],c_arr[:,int(c_arr.shape[1]/2):])).flatten()
    Dc_arr = jnp.vstack(Dc_)
    Dc = jnp.vstack((Dc_arr[:,:int(Dc_arr.shape[1]/2)],Dc_arr[:,int(Dc_arr.shape[1]/2):]))


    if telescope_metaparams["axis_index"] is None: 
        o = jnp.hstack((observations[:, telescope_metaparams["obs_index"]],
                        observations[:, telescope_metaparams["obs_index"]+1]))
    else:
        o0 =  observations[observations[:,telescope_metaparams["axis_index"]]==0, telescope_metaparams["obs_index"]]
        o1 = observations[observations[:,telescope_metaparams["axis_index"]]==1, telescope_metaparams["obs_index"]]
        o = jnp.hstack((o0,o1))

    plot_residuals(o - c,"Residuals/calibration_residuals_{}.png".format(cal[0]))

    #compute normal matrix
    N = Dc.T @ Dc
    b = Dc.T @ (o - c)
    
    ##solve for the difference between current (assumed) parameters and optimal parameters
    cfac = cho_factor(N)
    delta = cho_solve(cfac, b)
    return cal[1:] + delta


def update_calibration(src, att,cal,ephemeris,obs,model,telescope_metaparams,detector_metaparams,_min_nobs=200):
  ''' Updates of the attitude parameters

    Input:
        src: source parameters (N x (1+num_srcparams))
            - First column: source_id
            - Second to last columns: source parameters
        att: attitude parameters (M x (2+num_attparams))
            - First column: exposure_id
            - Second column: time of exposure
            - Third to last columns: attitude parameters
        cal: calibration parameters (P x (1+num_calparams))
            - First column: calibration_unit_id
            - Second to last columns: calibration parameters
        ephemeris: ephemeris of the satellite (M x 6):
            - Positions (in AU) and velocities (in m/s) for
            each observation.
        obs: array of observations ((2*num_obs)x5)
            Must contain at least the following columns: 
            - associated source_id
            - associated exposure_id
            - associated calibration_unit_id
            - associated detector_id
            - axis (either 0 or 1)
            - value of the observation
            - value of the uncertainty in the observation
        model: forward modeling function that predicts the observations
            from all the model parameters.
            Inputs (at call) in order:
                - source parameters
                - attitude parameters
                - calibration parameters
                - time
                - ephemeris
                - detector ID
        telescope_metaparams: dictionary containing the relevant
            metaparameters of the mission. It should contain:
                - 'F': nominal focal length
                - 'sX': nominal pixel size along [same units as F]
                - 'sY': nominal pixel size across [same units as F]
                - 'kappa0': position in pixels of the first usable column
                - 'mu0': position in pixels of the first usable row
                - 'kappaC': horizontal pixel position of the centre 
                    of the detector in the DRS.
                - 'muC': vertical pixel position of the centre 
                    of the detector in the DRS.
                - 'nCol': number of usable columns in one detector
                - 'nRow': number of usable rows in one detector
                - 'x0'&'y0': FPRS coordinates of the nominal foot point of
                        the optical telescope axis on the focal plane
                        [same units as F]
                - 'xC': array containing the FPRS x-coordinate of the centre
                    of each detector [same units as F]
                - 'yC': array containing the FPRS y-coordinate of the centre
                    of each detector [same units as F]
                - 'R': array containing the orientation in FPRS coordinates
                    of each detector
                - 'srcid_index': number of the column in the observation array
                    containing the source_id
                - 'expid_index': number of the column in the observation array
                    containing the exposure_id
                - 'calid_index': number of the column in the observation array
                    containing the calibration_unit_id
                - 'detid_index': number of the column in the observation array
                    containing the detector_id
                - 'axis_index': number of the column in the observation array
                    containing the axis information (0 - horizontal, 1 - vertical)
                - 'obs_index': number of the column in the observation array
                    containing the actual measurement
                - 'obssig_index': number of the column in the observation array
                    containing the uncertainty measurement
        detector_metaparams: array containing, in each row, the following values
                - First column: detector ID
                - Second column: FPRS x-coordinate of the centre of the detector
                - Third column: FPRS y-coordinate of the centre of the detector
                - Forth to Seventh columns: components of the rotation matrix
                    that defines the orientation in FPRS coordinates
                    of the detector as [R00 R10 R01 R11] = [[R00, R01],
                                                            [R10, R11]]
                    The corresponding rotation angle is theta = arctan2(R01,R00)
                    (right handed || counter clock-wise)
        _min_nobs: minimum number of observations in the calibration unit to attempt a solution.
            Below this number, the passed values for the guess are returned. 

        where N is the number of sources that we want to solve for,
        M is the total number of exposures taken (each one at time T), 
        and P is the number of calibration units used.

    Returns:
    The updated calibration parameters at all the exposures.
  '''
  #loop through each exposure and find their optimal attitude parameters
  return jnp.vstack([update_calibration_inner(src, att,cal_,ephemeris,obs,model,telescope_metaparams,detector_metaparams,_min_nobs) for i,cal_ in enumerate(cal)])



def iterate_calibration(src_assumed,att_assumed,cal_guess,ephemeris,obs,model,telescope_metaparams,detector_metaparams,n_iter=1,_min_nobs=200):
    """
    Iterate n_iter times to reach the optimal attitude parameters given some
    source and calibration parameters, as well as the observations and a model
    to predict the observations from the model parameters.

    Input:
        src_assumed: assumed source parameters (N x (1+num_srcparams))
            - First column: source_id
            - Second to last columns: source parameters
        att_assumed: assumed attitude parameters (M x (2+num_attparams))
            - First column: exposure_id
            - Second column: time of exposure
            - Third to last columns: attitude parameters
        cal_guess: initial guess calibration parameters (P x (1+num_calparams))
            - First column: calibration_unit_id
            - Second to last columns: calibration parameters
        ephemeris: ephemeris of the satellite (M x 6):
            - Positions (in AU) and velocities (in m/s) for
            each observation.
        obs: array of observations ((2*num_obs)x5)
            Must contain at least the following columns: 
            - associated source_id
            - associated exposure_id
            - associated calibration_unit_id
            - associated detector_id
            - axis (either 0 or 1)
            - value of the observation
            - value of the uncertainty in the observation
        model: forward modeling function that predicts the observations
            from all the model parameters.
            Inputs (at call) in order:
                - source parameters
                - attitude parameters
                - calibration parameters
                - time
                - ephemeris
                - detector ID
        telescope_metaparams: dictionary containing the relevant
            metaparameters of the mission. It should contain:
                - 'F': nominal focal length
                - 'sX': nominal pixel size along [same units as F]
                - 'sY': nominal pixel size across [same units as F]
                - 'kappa0': position in pixels of the first usable column
                - 'mu0': position in pixels of the first usable row
                - 'kappaC': horizontal pixel position of the centre 
                    of the detector in the DRS.
                - 'muC': vertical pixel position of the centre 
                    of the detector in the DRS.
                - 'nCol': number of usable columns in one detector
                - 'nRow': number of usable rows in one detector
                - 'x0'&'y0': FPRS coordinates of the nominal foot point of
                        the optical telescope axis on the focal plane
                        [same units as F]
                - 'xC': array containing the FPRS x-coordinate of the centre
                    of each detector [same units as F]
                - 'yC': array containing the FPRS y-coordinate of the centre
                    of each detector [same units as F]
                - 'R': array containing the orientation in FPRS coordinates
                    of each detector
                - 'srcid_index': number of the column in the observation array
                    containing the source_id
                - 'expid_index': number of the column in the observation array
                    containing the exposure_id
                - 'calid_index': number of the column in the observation array
                    containing the calibration_unit_id
                - 'detid_index': number of the column in the observation array
                    containing the detector_id
                - 'axis_index': number of the column in the observation array
                    containing the axis information (0 - horizontal, 1 - vertical)
                - 'obs_index': number of the column in the observation array
                    containing the actual measurement
                - 'obssig_index': number of the column in the observation array
                    containing the uncertainty measurement
        detector_metaparams: array containing, in each row, the following values
                - First column: detector ID
                - Second column: FPRS x-coordinate of the centre of the detector
                - Third column: FPRS y-coordinate of the centre of the detector
                - Forth to Seventh columns: components of the rotation matrix
                    that defines the orientation in FPRS coordinates
                    of the detector as [R00 R10 R01 R11] = [[R00, R01],
                                                            [R10, R11]]
                    The corresponding rotation angle is theta = arctan2(R01,R00)
                    (right handed || counter clock-wise)
        n_iter: number of loops (if the initial guess is "far" from the
                true values, the non-linearity of the problem will 
                require a few iterations to converge. Probably somewhere
                between 5 and 10.)
         _min_nobs: minimum number of observations in the calibration unit to attempt a solution.
            Below this number, the passed values for the guess are returned. 
    
    Output:
        - Improved calibration parameters
    """
    for i in range(n_iter):
        cal_guess = jnp.column_stack((cal_guess[:,:1],
                                          update_calibration(src_assumed,att_assumed,cal_guess,ephemeris,obs,
                                                             model,telescope_metaparams,detector_metaparams,_min_nobs)))
    return cal_guess


def iterate_attitude(src_assumed,att_guess,cal_assumed,ephemeris,obs,model,telescope_metaparams,detector_metaparams,n_iter=1,_min_nstars=100):
    """
    Iterate n_iter times to reach the optimal attitude parameters given some
    source and calibration parameters, as well as the observations and a model
    to predict the observations from the model parameters.

    Input:
        src_assumed: assumed source parameters (N x (1+num_srcparams))
            - First column: source_id
            - Second to last columns: source parameters
        att_guess: initial guess for the attitude parameters (M x (2+num_attparams))
            - First column: exposure_id
            - Second column: time of exposure
            - Third to last columns: attitude parameters
        cal_assumed: assumed calibration parameters (P x (1+num_calparams))
            - First column: calibration_unit_id
            - Second to last columns: calibration parameters
        ephemeris: ephemeris of the satellite (M x 6):
            - Positions (in AU) and velocities (in m/s) for
            each observation.
        obs: array of observations ((2*num_obs)x5)
            Must contain at least the following columns: 
            - associated source_id
            - associated exposure_id
            - associated calibration_unit_id
            - associated detector_id
            - axis (either 0 or 1)
            - value of the observation
            - value of the uncertainty in the observation
        model: forward modeling function that predicts the observations
            from all the model parameters.
            Inputs (at call) in order:
                - source parameters
                - attitude parameters
                - calibration parameters
                - time
                - ephemeris
                - detector ID
        telescope_metaparams: dictionary containing the relevant
            metaparameters of the mission. It should contain:
                - 'F': nominal focal length
                - 'sX': nominal pixel size along [same units as F]
                - 'sY': nominal pixel size across [same units as F]
                - 'kappa0': position in pixels of the first usable column
                - 'mu0': position in pixels of the first usable row
                - 'kappaC': horizontal pixel position of the centre 
                    of the detector in the DRS.
                - 'muC': vertical pixel position of the centre 
                    of the detector in the DRS.
                - 'nCol': number of usable columns in one detector
                - 'nRow': number of usable rows in one detector
                - 'x0'&'y0': FPRS coordinates of the nominal foot point of
                        the optical telescope axis on the focal plane
                        [same units as F]
                - 'xC': array containing the FPRS x-coordinate of the centre
                    of each detector [same units as F]
                - 'yC': array containing the FPRS y-coordinate of the centre
                    of each detector [same units as F]
                - 'R': array containing the orientation in FPRS coordinates
                    of each detector
                - 'srcid_index': number of the column in the observation array
                    containing the source_id
                - 'expid_index': number of the column in the observation array
                    containing the exposure_id
                - 'calid_index': number of the column in the observation array
                    containing the calibration_unit_id
                - 'detid_index': number of the column in the observation array
                    containing the detector_id
                - 'axis_index': number of the column in the observation array
                    containing the axis information (0 - horizontal, 1 - vertical)
                - 'obs_index': number of the column in the observation array
                    containing the actual measurement
                - 'obssig_index': number of the column in the observation array
                    containing the uncertainty measurement
        detector_metaparams: array containing, in each row, the following values
                - First column: detector ID
                - Second column: FPRS x-coordinate of the centre of the detector
                - Third column: FPRS y-coordinate of the centre of the detector
                - Forth to Seventh columns: components of the rotation matrix
                    that defines the orientation in FPRS coordinates
                    of the detector as [R00 R10 R01 R11] = [[R00, R01],
                                                            [R10, R11]]
                    The corresponding rotation angle is theta = arctan2(R01,R00)
                    (right handed || counter clock-wise)
        n_iter: number of loops (if the initial guess is "far" from the
                true values, the non-linearity of the problem will 
                require a few iterations to converge. Probably somewhere
                between 5 and 10.)
        _min_nstars: minimum number of sources in the exposure to attempt a solution.
            Below this number, the passed values for the guess are returned.
    
    Output:
        - Improved attitude parameters
    """
    for i in range(n_iter):
        att_guess = jnp.column_stack((att_guess[:,:2],
                                          update_attitude(src_assumed,att_guess,cal_assumed,ephemeris,obs,
                                                          model,telescope_metaparams,detector_metaparams,_min_nstars)))
    return att_guess

def iterate_source(src_guess,att_assumed,cal_assumed,ephemeris,obs,priors,sig_priors,model,telescope_metaparams,detector_metaparams,n_iter=1,_min_nobs=10):
    """
    Iterate n_iter times to reach the optimal source parameters given some
    attitude and calibration parameters, as well as the observations and a model
    to predict the observations from the model parameters.

    Input:
        src_guess: initial guess for the  source parameters (N x (1+num_srcparams))
            - First column: source_id
            - Second to last columns: source parameters
        att_assumed: assumed attitude parameters (M x (2+num_attparams))
            - First column: exposure_id
            - Second column: time of exposure
            - Third to last columns: attitude parameters
        cal_assumed: assumed calibration parameters (P x (1+num_calparams))
            - First column: calibration_unit_id
            - Second to last columns: calibration parameters
        ephemeris: ephemeris of the satellite (M x 6):
            - Positions (in AU) and velocities (in m/s) for
            each observation.
        obs: array of observations ((2*num_obs)x5)
            Must contain at least the following columns: 
            - associated source_id
            - associated exposure_id
            - associated calibration_unit_id
            - associated detector_id
            - axis (either 0 or 1)
            - value of the observation
            - value of the uncertainty in the observation
        priors: prior on the source parameters (N x (1+num_srcparams))
        sig_prior: uncertainty on the prior on the source parameters 
                    (N x (1+num_srcparams))
        model: forward modeling function that predicts the observations
            from all the model parameters.
            Inputs (at call) in order:
                - source parameters
                - attitude parameters
                - calibration parameters
                - time
                - ephemeris
                - detector ID
        telescope_metaparams: dictionary containing the relevant
            metaparameters of the mission. It should contain:
                - 'F': nominal focal length
                - 'sX': nominal pixel size along [same units as F]
                - 'sY': nominal pixel size across [same units as F]
                - 'kappa0': position in pixels of the first usable column
                - 'mu0': position in pixels of the first usable row
                - 'kappaC': horizontal pixel position of the centre 
                    of the detector in the DRS.
                - 'muC': vertical pixel position of the centre 
                    of the detector in the DRS.
                - 'nCol': number of usable columns in one detector
                - 'nRow': number of usable rows in one detector
                - 'x0'&'y0': FPRS coordinates of the nominal foot point of
                        the optical telescope axis on the focal plane
                        [same units as F]
                - 'xC': array containing the FPRS x-coordinate of the centre
                    of each detector [same units as F]
                - 'yC': array containing the FPRS y-coordinate of the centre
                    of each detector [same units as F]
                - 'R': array containing the orientation in FPRS coordinates
                    of each detector
                - 'srcid_index': number of the column in the observation array
                    containing the source_id
                - 'expid_index': number of the column in the observation array
                    containing the exposure_id
                - 'calid_index': number of the column in the observation array
                    containing the calibration_unit_id
                - 'detid_index': number of the column in the observation array
                    containing the detector_id
                - 'axis_index': number of the column in the observation array
                    containing the axis information (0 - horizontal, 1 - vertical)
                - 'obs_index': number of the column in the observation array
                    containing the actual measurement
                - 'obssig_index': number of the column in the observation array
                    containing the uncertainty measurement
        detector_metaparams: array containing, in each row, the following values
                - First column: detector ID
                - Second column: FPRS x-coordinate of the centre of the detector
                - Third column: FPRS y-coordinate of the centre of the detector
                - Forth to Seventh columns: components of the rotation matrix
                    that defines the orientation in FPRS coordinates
                    of the detector as [R00 R10 R01 R11] = [[R00, R01],
                                                            [R10, R11]]
                    The corresponding rotation angle is theta = arctan2(R01,R00)
                    (right handed || counter clock-wise)
        n_iter: number of loops (if the initial guess is "far" from the
                true values, the non-linearity of the problem will 
                require a few iterations to converge. Probably somewhere
                between 2 and 5.)
        _min_nobs: minimum number of observations to attempt a solution.
            Below this number, the passed values for the guess are returned. 
    
    Output:
        - Improved source parameters
    """
    for i in range(n_iter):
        src_guess = jnp.column_stack((src_guess[:,:1],update_source(src_guess,att_assumed,cal_assumed,ephemeris,obs,
                                        priors,sig_priors,model,telescope_metaparams,detector_metaparams,_min_nobs)))
    return src_guess

def bloc_iteration(src:"array",att:"array",cal:"array",ephemeris:"array",obs:"array",model:"function",
                   telescope_metaparams:"dict",detector_metaparams:"array",
                   src_niter:int,att_niter:int,cal_niter:int,niter:int,
                   priors=None,sig_priors=None,
                   _min_nobs:int=10,_min_nstars:int=100,_min_nobs_cal:int=200) -> "(array,array,array)":
    """
    Run the bloc iterative solver

    Input:
        src_guess: initial guess for the  source parameters (N x (1+num_srcparams))
            - First column: source_id
            - Second to last columns: source parameters
        att: assumed attitude parameters (M x (2+num_attparams))
            - First column: exposure_id
            - Second column: time of exposure
            - Third to last columns: attitude parameters
        cal: assumed calibration parameters (P x (1+num_calparams))
            - First column: calibration_unit_id
            - Second to last columns: calibration parameters
        ephemeris: ephemeris of the satellite (M x 6):
            - Positions (in AU) and velocities (in m/s) for
            each observation.
        obs: array of observations ((2*num_obs)x5)
            Must contain at least the following columns: 
            - associated source_id
            - associated exposure_id
            - associated calibration_unit_id
            - associated detector_id
            - axis (either 0 or 1)
            - value of the observation
            - value of the uncertainty in the observation
        model: forward modeling function that predicts the observations
            from all the model parameters.
            Inputs (at call) in order:
                - source parameters
                - attitude parameters
                - calibration parameters
                - time
                - ephemeris
                - detector ID
        telescope_metaparams: dictionary containing the relevant
            metaparameters of the mission. It should contain:
                - 'F': nominal focal length
                - 'sX': nominal pixel size along [same units as F]
                - 'sY': nominal pixel size across [same units as F]
                - 'kappa0': position in pixels of the first usable column
                - 'mu0': position in pixels of the first usable row
                - 'kappaC': horizontal pixel position of the centre 
                    of the detector in the DRS.
                - 'muC': vertical pixel position of the centre 
                    of the detector in the DRS.
                - 'nCol': number of usable columns in one detector
                - 'nRow': number of usable rows in one detector
                - 'x0'&'y0': FPRS coordinates of the nominal foot point of
                        the optical telescope axis on the focal plane
                        [same units as F]
                - 'xC': array containing the FPRS x-coordinate of the centre
                    of each detector [same units as F]
                - 'yC': array containing the FPRS y-coordinate of the centre
                    of each detector [same units as F]
                - 'R': array containing the orientation in FPRS coordinates
                    of each detector
                - 'srcid_index': number of the column in the observation array
                    containing the source_id
                - 'expid_index': number of the column in the observation array
                    containing the exposure_id
                - 'calid_index': number of the column in the observation array
                    containing the calibration_unit_id
                - 'detid_index': number of the column in the observation array
                    containing the detector_id
                - 'axis_index': number of the column in the observation array
                    containing the axis information (0 - horizontal, 1 - vertical)
                - 'obs_index': number of the column in the observation array
                    containing the actual measurement
                - 'obssig_index': number of the column in the observation array
                    containing the uncertainty measurement
        detector_metaparams: array containing, in each row, the following values
                - First column: detector ID
                - Second column: FPRS x-coordinate of the centre of the detector
                - Third column: FPRS y-coordinate of the centre of the detector
                - Forth to Seventh columns: components of the rotation matrix
                    that defines the orientation in FPRS coordinates
                    of the detector as [R00 R10 R01 R11] = [[R00, R01],
                                                            [R10, R11]]
                    The corresponding rotation angle is theta = arctan2(R01,R00)
                    (right handed || counter clock-wise)
        src_niter,att_niter,cal_niter,n_iter: number of loops for, 
            respectively, the source, attitude and calibration loops, and
            for the outer-most loop.
            (if the initial guess is "very far" from the true values, 
            the non-linearity of the problem will require a few 
            iterations to converge. Probably somewhere between 2 and 4.)
        priors: prior on the source parameters (N x (1+num_srcparams))
        sig_prior: uncertainty on the prior on the source parameters 
                    (N x (1+num_srcparams))
        _min_nobs: minimum number of observations to attempt a solution.
            Below this number, the passed values for the guess are returned. 
        _min_nstars: minimum number of sources in the exposure to attempt a solution.
            Below this number, the passed values for the guess are returned.
        _min_nobs_cal: minimum number of observations in the calibration unit to attempt a solution.
            Below this number, the passed values for the guess are returned. 
    
    Output:
        - Improved source parameters
        - Improved attitude parameters
        - Improved calibration parameters

    """
    for i in range(niter):
        #update attitude
        att = iterate_attitude(src,att,cal,ephemeris,obs,model,telescope_metaparams,detector_metaparams,att_niter,_min_nstars)
        print(att)
        #update calibration
        cal = iterate_calibration(src,att,cal,ephemeris,obs,model,telescope_metaparams,detector_metaparams,cal_niter,_min_nobs_cal)
        print(cal)
        #update sources
        src = iterate_source(src,att,cal,ephemeris,obs,priors,sig_priors,model,telescope_metaparams,detector_metaparams,src_niter,_min_nobs)
        
    return src,att,cal
"""
Set of functions that are used for the bloc iterative astrometric solution
"""

from jax import vmap,jacrev,numpy as jnp
from jax.scipy.linalg import cho_factor,cho_solve


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
            - if argnum is 0, it should be (None,0,0,0)
            - if argnum is 1, it should be (0,None,0,None)
            - if argnum is 2, it should be (0,0,None,0) or (0,0,None,None),
                depending on whether the calibration parameters depend
                on time or not.

    Output:
        - The jacobian matrix of the bloc.
    """
    
    return vmap(jacrev(model,argnums=(argnum)),axis)


def update_source_inner(src, att, cal, ephemeris,obs,sig_obs,priors,sig_prior,model,_min_nobs=10):
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
        obs: array of observations ((2*num_obs)x5)
            - First column: associated source_id
            - Second column: associated exposure_id
            - Third column: associated calibration_unit_id
            - Forth column: axis (either 0 or 1)
            - Fifth (last) column: value of the observation
        sig_obs: observational uncertainties ((2*num_obs)x1)
        priors: prior on the source parameters (N x (1+num_srcparams))
        sig_prior: uncertainty on the prior on the source parameters 
                        (N x (1+num_srcparams))
        model: forward modeling function that predicts the observations
            from all the model parameters.
                - inputs: source, attitude, calibration and time
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
    observations = obs[obs[:, 0] == src[0]]
        #locate the parameters of the exposures when this source id was observed
    mask = jnp.isin(att[:,0],observations[:,1])
    exposures = att[mask]

    #if we do not have enough observations, do not even try
    if len(exposures)<_min_nobs:
        return src[1:]

    #create iterative version of the model
    _iterate_att = vmap(model, (None, 0,None, 0,0), 0)
    #prepare the Jacobian matrix wrt source parameters
        #the jacobian has to be iterated over all exposures (To-Do: and calibration units)
    _Jds = _Jacobian_autodiff(model,0,(None,0,None,0,0)) 
    
    #check if priors are given:
    if priors is not None:
        if sig_prior is None:
            raise ValueError("If the priors are given, you need to provide also their uncertainties!")
        else:
            #select priors on the parameters of the source of interest
            priors_ = priors[priors[:,0]==src[0]][0]
            sig_prior_ = sig_prior[sig_prior[:,0]==src[0]][0]
            
    #compute the predicted observations at each exposure
    c = jnp.hstack(_iterate_att(src[1:], exposures[:,2:], cal, exposures[:,1],ephemeris[mask]))
    #create vector of observations to compare to
    o = observations[:, -1]
    
    #compute design matrix
    Ds = jnp.vstack(_Jds(src[1:], exposures[:,2:],cal,exposures[:,1],ephemeris[mask]))

    if priors is None:
        #compute the normal matrix as usual 
        N = Ds.T @ Ds
        b = Ds.T @ (o - c)
    else:
        if isinstance(sig_obs,(float,int)):
            S = jnp.eye(len(o))*sig_obs**(-2)
        elif isinstance(sig_obs,jnp.ndarray):
            s = sig_obs[mask]
            S = jnp.diag(s**(-2))
        else:
            raise ValueError("The observational uncertainties have to be either a scalar or a numpy array")
        #compute the normal matrix accounting for a gaussian prior
        N = Ds.T @ S @ Ds + jnp.diag(sig_prior_[1:]**(-2))
        b = Ds.T @ S @ (o - c) + jnp.matmul(jnp.diag(sig_prior_[1:]**(-2)),(priors_[1:]-src[1:]))
    #solve for the difference between current (assumed) parameters and optimal parameters
    cfac = cho_factor(N)
    delta = cho_solve(cfac, b)
    return src[1:] + delta


def update_source(src, att, cal,obs,sig_obs,priors,sig_prior,model,_min_nobs=10):
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
        obs: array of observations ((2*num_obs)x5)
        - First column: associated source_id
        - Second column: associated exposure_id
        - Third column: associated calibration_unit_id
        - Forth column: axis (either 0 or 1)
        - Fifth (last) column: value of the observation
        sig_obs: observational uncertainties ((2*num_obs)x1)
        priors: prior on the source parameters (N x (1+num_srcparams))
        sig_prior: uncertainty on the prior on the source parameters 
                    (N x (1+num_srcparams))
        model: forward modeling function that predicts the observations
        from all the model parameters.
            - inputs: source, attitude, calibration and time
        _min_nobs: minimum number of observations to attempt a solution.
            Below this number, the passed values for the guess are returned. 

        where N is the number of sources that we want to solve for,
        M is the total number of exposures taken (each one at time T), 
        and P is the number of calibration units used.

    Returns:
    The updated source parameters of all sources.
  '''
  #loop through each source and find their optimal source parameters
  return jnp.vstack([update_source_inner(src_, att, cal,obs,sig_obs,priors,sig_prior,model,_min_nobs) for src_ in src])


def update_source(src, att, cal,ephemeris,obs,sig_obs,priors,sig_prior,model,_min_nobs=10):
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
            - First column: associated source_id
            - Second column: associated exposure_id
            - Third column: associated calibration_unit_id
            - Forth column: axis (either 0 or 1)
            - Fifth (last) column: value of the observation
        sig_obs: observational uncertainties ((2*num_obs)x1)
        priors: prior on the source parameters (N x (1+num_srcparams))
        sig_prior: uncertainty on the prior on the source parameters 
                    (N x (1+num_srcparams))
        model: forward modeling function that predicts the observations
        from all the model parameters.
            - inputs: source, attitude, calibration and time
        _min_nobs: minimum number of observations to attempt a solution.
            Below this number, the passed values for the guess are returned. 

        where N is the number of sources that we want to solve for,
        M is the total number of exposures taken (each one at time T), 
        and P is the number of calibration units used.

    Returns:
    The updated source parameters of all sources.
  '''
  #loop through each source and find their optimal source parameters
  return jnp.vstack([update_source_inner(src_, att, cal,ephemeris,obs,sig_obs,priors,sig_prior,model,_min_nobs) for src_ in src])


def update_attitude_inner(src, att,cal,ephemeris,obs,model,_min_nstars=100):
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
            - First column: associated source_id
            - Second column: associated exposure_id
            - Third column: associated calibration_unit_id
            - Forth column: axis (either 0 or 1)
            - Fifth (last) column: value of the observation
        model: forward modeling function that predicts the observations
        from all the model parameters.
            - inputs: source, attitude, calibration and time
        _min_nstars: minimum number of sources in the exposure to attempt a solution.
            Below this number, the passed values for the guess are returned. 

    Output:
    updated attitude parameters of this exposure
    """

    #locate the observational data of the sources observed at this exposure
    observations = obs[obs[:, 1] == att[0]]
    #locate the source parameters of the sources observed at this exposure
    mask = jnp.isin(src[:,0],observations[:,0])
    sources = src[mask]

    #if we do not have enough sources in the exposure, do not even try
    if len(sources)<_min_nstars:
        return att[2:]

    #create iterable version of the model
    _iterate_src = vmap(model, (0, None, None, None,None), 0)
    #prepare the Jacobian matrix wrt attitude parameters
        #the jacobian has to be iterated over all sources (To-Do: and calibration units)
    _Jda = _Jacobian_autodiff(model,1,(0,None,None,None,None))

    #compute the predicted observations
    c = jnp.hstack(_iterate_src(sources[:,1:], att[2:],cal, att[1:],ephemeris))
    #create vector of observations to compare to
    o = observations[:, -1]

    #compute design matrix
    De = jnp.vstack(_Jda(sources[:,1:],att[2:],cal,att[1:],ephemeris))
    
    #compute normal matrix
    N = De.T @ De
    b = De.T @ (o - c)
    
    ##solve for the difference between current (assumed) parameters and optimal parameters
    cfac = cho_factor(N)
    delta = cho_solve(cfac, b)
    return att[2:] + delta

def update_attitude(src, att,cal,ephemeris,obs,model,_min_nstars=100):
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
            - First column: associated source_id
            - Second column: associated exposure_id
            - Third column: associated calibration_unit_id
            - Forth column: axis (either 0 or 1)
            - Fifth (last) column: value of the observation
        model: forward modeling function that predicts the observations
        from all the model parameters.
            - inputs: source, attitude, calibration and time
        _min_nstars: minimum number of sources in the exposure to attempt a solution.
            Below this number, the passed values for the guess are returned. 

        where N is the number of sources that we want to solve for,
        M is the total number of exposures taken (each one at time T), 
        and P is the number of calibration units used.

    Returns:
    The updated attitude parameters at all the exposures.
  '''
  #loop through each exposure and find their optimal attitude parameters
  return jnp.vstack([update_attitude_inner(src, att_,cal,ephemeris[i],obs,model,_min_nstars) for i,att_ in enumerate(att)])


def iterate_attitude(src_assumed,att_guess,cal_assumed,ephemeris,obs,model,n_iter=1,_min_nstars=100):
    """
    Iterate n_iter times to reach the optimal attitude parameters given some
    source and calibration parameters, as well as the observations and a model
    to predict the observations from the model parameters.

    Input:
        src_assumed: assumed source parameters (N x (1+num_srcparams))
            - First column: source_id
            - Second to last columns: source parameters
        att: initial guess for the attitude parameters (M x (2+num_attparams))
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
            - First column: associated source_id
            - Second column: associated exposure_id
            - Third column: associated calibration_unit_id
            - Forth column: axis (either 0 or 1)
            - Fifth (last) column: value of the observation
        model: forward modeling function that predicts the observations
            from all the model parameters.
            - inputs: source, attitude, calibration and time
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
                                          update_attitude(src_assumed,att_guess,cal_assumed,ephemeris,obs,model,_min_nstars)))
    return att_guess

def iterate_source(src_guess,att_assumed,cal_assumed,ephemeris,obs,sig_obs,priors,sig_priors,model,n_iter=1,_min_nobs=10):
    """
    Iterate n_iter times to reach the optimal source parameters given some
    attitude and calibration parameters, as well as the observations and a model
    to predict the observations from the model parameters.

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
            - First column: associated source_id
            - Second column: associated exposure_id
            - Third column: associated calibration_unit_id
            - Forth column: axis (either 0 or 1)
            - Fifth (last) column: value of the observation
        sig_obs: observational uncertainties ((2*num_obs)x1)
        priors: prior on the source parameters (N x (1+num_srcparams))
        sig_prior: uncertainty on the prior on the source parameters 
                    (N x (1+num_srcparams))
        model: forward modeling function that predicts the observations
            from all the model parameters.
            - inputs: source, attitude, calibration and time
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
        src_guess = jnp.column_stack((src_guess[:,:1],
                                          update_source(src_guess,att_assumed,cal_assumed,ephemeris,obs,sig_obs,priors,sig_priors,model,_min_nobs)))
    return src_guess

def bloc_iteration(src:"array",att:"array",cal:"array",ephemeris:"array",obs:"array",model:"function",src_niter:int,att_niter:int,niter:int,priors=None,sig_obs=None,sig_priors=None,_min_nobs:int=10,_min_nstars:int=100) -> "(array,array,array)":
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
            - First column: associated source_id
            - Second column: associated exposure_id
            - Third column: associated calibration_unit_id
            - Forth column: axis (either 0 or 1)
            - Fifth (last) column: value of the observation
        sig_obs: observational uncertainties ((2*num_obs)x1)
        priors: prior on the source parameters (N x (1+num_srcparams))
        sig_prior: uncertainty on the prior on the source parameters 
                    (N x (1+num_srcparams))
        model: forward modeling function that predicts the observations
            from all the model parameters.
            - inputs: source, attitude, calibration and time
        n_iter: number of loops (if the initial guess is "far" from the
                true values, the non-linearity of the problem will 
                require a few iterations to converge. Probably somewhere
                between 2 and 5.)
        _min_nobs: minimum number of observations to attempt a solution.
            Below this number, the passed values for the guess are returned. 
        _min_nstars: minimum number of sources in the exposure to attempt a solution.
            Below this number, the passed values for the guess are returned.
    
    Output:
        - Improved source parameters
        - Improved attitude parameters
        - Improved calibration parameters

    """
    for i in range(niter):
        #update attitude
        att = iterate_attitude(src,att,cal,ephemeris,obs,model,att_niter,_min_nstars)
       
        #update sources
        src = iterate_source(src,att,cal,ephemeris,obs,sig_obs,priors,sig_priors,model,src_niter,_min_nobs)
        
    return src,att,cal
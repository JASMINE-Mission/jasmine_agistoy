"""
Set of functions that are used to generate mock observations from a given model
"""
import numpy as np
from jax import vmap,numpy as jnp

def generate_mock_obs_simple(src, att,cal,model, noise=0.00):
    ''' Generate observations based on the given
     model from the parameters
    
    Arguments:
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
        model: forward modeling function that predicts the observations
        from all the model parameters.
            - inputs: source, attitude, calibration and time
        noise: gaussian noise to convolve the observations with
    
    
    Returns:
        - obs: array of observations ((2*num_obs)x5)
            - First column: associated source_id
            - Second column: associated exposure_id
            - Third column: associated calibration_unit_id
            - Forth column: axis (either 0 or 1)
            - Fifth column: value of the observation

    Note -- To-Do: add the calibration to everything
    '''
    #define the basic functions that generate predictions and map them
    _iterate_att = vmap(model, (None, 0, None,0), 0) #iterate over the exposures
    _iterate_full = vmap(_iterate_att, (0, None,None,None), 0) #iterate over the sources
    generate_predictions = lambda s, e,c, t: jnp.hstack(jnp.vstack(_iterate_full(s, e,c, t))) #wrapper

    #generate mock data
    obs = generate_predictions(src[:, 1:], att[:, 2:],cal, att[:, 1])
    #add observational errors to the mock data (if necessary)
    if noise > 0:
        obs = np.random.normal(obs, noise)
    #Add tags and info to each observation
        #TO-D0: add calibration unit id
        #TO-DO: find a faster way to do this that preserves 
            # the order of the alternating observations (x then y axis)
    ids_ = jnp.array([[src[i,0].item(),att[j,0].item(),k] for i in range(len(src)) \
                      for j in range(len(att)) for k in [0,1]])
    return jnp.column_stack([ids_, obs])


def generate_mock_obs(src, att,cal,model, FoV_size, noise=0.00):
    ''' Generate observations based on the given
     model from the parameters
    
    Arguments:
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
      sig_obs: observational uncertainties ((2*num_obs)x1)
      priors: prior on the source parameters (N x (1+num_srcparams))
      sig_prior: uncertainty on the prior on the source parameters 
                    (N x (1+num_srcparams))
      model: forward modeling function that predicts the observations
        from all the model parameters.
            - inputs: source, attitude, calibration and time
      Fov_size: vector containing the FoV size of each exposure
    
    Returns:
        - obs: array of observations ((2*num_obs)x5)
            - First column: associated source_id
            - Second column: associated exposure_id
            - Third column: associated calibration_unit_id
            - Forth column: axis (either 0 or 1)
            - Fifth column: value of the observation

    Note -- To-Do: add the calibration to everything
    '''
    #define the basic functions that generate predictions and map them
    _iterate_src = vmap(model, (0, None, None,None), 0) #iterate over the sources

    ids_ = []
    obs_ = []
    for i,_att in enumerate(att):
        #find sources within FoV
        FoV_centre = _att[2:4]
        FoV_size_ = FoV_size[i]
            #TO-DO: account for rotation of the FoV
        sources = src[(np.abs(src[:,1]-FoV_centre[0])<FoV_size_/2)&(np.abs(src[:,2]-FoV_centre[1])<FoV_size_/2)]
        obs_.append(_iterate_src(sources[:,1:],_att[2:], None,_att[1]))
        ids_.extend([(sid,_att[0]) for sid in sources[:,0]])

    #apply uncertainties to observations
    obs = np.hstack([np.hstack(o) for o in obs_])
    if noise > 0:
        obs = np.random.normal(obs, noise)

    #create an array of source_ids,exposure_ids and axis. 
        #since the observations are a single vector alternating
        #between x- and y-axis, we need to do it this way
    ids = np.zeros((len(obs),3))
    ids[::2] = np.column_stack((ids_,np.zeros(len(ids_))))
    ids[1::2] = np.column_stack((ids_,np.ones(len(ids_))))

    return np.column_stack((ids,obs))
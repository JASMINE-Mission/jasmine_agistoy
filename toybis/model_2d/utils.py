import numpy as np
import pandas as pd
import re
import ast
import jax.numpy as jnp


__all__ = [
    'L0','L1','L2','L3','L4','L5','Legendre2D',
    "get_quat_from_rot","get_rot_from_quat","get_rot_from_angles","get_angles_from_rot",
    "rotate_from_angles","rotate_from_rot","rotate_from_quat","prepare_quat","invert_quat",
    "_transform_quaternion_to_angles","quaternion2angles","read_djl_file"
]

def L0():
        return 1
def L1(z):
    return z
    #return 2*(z - 1/2)
def L2(z):
    return 1/2*(3*z**2 - 1)
    #return 6*(z - 1/2)**2 - 1/2
def L3(z):
    return 1/2*(5*z**3 - 3*z)
    #return 20*(z - 1/2)**3 - 3*(z - 1/2)
def L4(z):
    return 1/8*(35*z**4 - 30*z**2 + 3)
    #return 70*(z - 1/2)**4 - 15*(z - 1/2)**2 + 3/8
def L5(z):
    return 1/8*(63*z**5 - 70*z**3 + 15*z)
    #return 252*(z - 1/2)**5 - 70*(z - 1/2)**3 + 15/4*(z - 1/2)

def Legendre2D(x,y,*coeffs,min_order=0,max_order=5,_zeropoint = 0):
    """
    Returns the value of the 2D Legendre Polynomials evaluated at x&y coordinates.

    Input:
        - x,y: x and y coordinates
        - coeffs: depending on the number of orders considered, the amount of
            coefficients passed may vary. 
        - min_order: minimum order to consider (0 to 5)
        - max_order: maximum order to consider (5 to 0)
        - _zeropoint: since the higher order terms displace the origin,
            _zeropoint = (c20 + c02) / 2 - c22 / 4 - (c40 + c04) * 3 / 8
            prevents that when the first order terms are omitted. 

    Expected order of the coefficients is [Cij], with i + j = n, j increasing 
    from 0 to n (thus i decreasing from n to 0), and n increasing from 
    min_order to max_order.

    e.g.: min_order = 0, max_order = 2 => coeffs = [c00,c10,c01,c20,c11,c02]
    e.g.: min_order = 2, max_order = 3 => coeffs = [c20,c11,c02,c30,c21,c12,c33]
    
    """
    if min_order<0 or min_order>max_order:
        raise ValueError("Wrong min_order provided! It must be a number between 0 and max_order.")
    if max_order>5 or max_order<min_order:
        raise ValueError("Wrong max_order provided! It must be a number between min_order and 5.")

    num_coeff = int(((max_order+1)*(2 + max_order) - min_order*(1 + min_order))/2)

    if len(coeffs)!=num_coeff:
        raise ValueError("Number of provided coefficients is not correct!\nExpected {} but got {}!!".format(num_coeff,len(coeffs)))
    
    #order 0
    L0_val = coeffs[0]*L0()*L0() if min_order==0 else 0*x

    #order 1
    L1_val = coeffs[1-min_order]*L1(x)*L0() + \
            coeffs[2-min_order]*L0()*L1(y) if (min_order<=1 and max_order>=1) else 0*x

    #oder 2
    L2_val = coeffs[int(3 -0.5*min_order**2 - 0.5*min_order)]*L2(x)*L0() + \
            coeffs[int(4 -0.5*min_order**2 - 0.5*min_order)]*L1(x)*L1(y) + \
            coeffs[int(5 -0.5*min_order**2 - 0.5*min_order)]*L0()*L2(y) if (min_order<=2 and max_order>=2) else 0*x

    #order 3
    L3_val = coeffs[int(6 -0.5*min_order**2 - 0.5*min_order)]*L3(x)*L0() + \
          coeffs[int(7 -0.5*min_order**2 - 0.5*min_order)]*L2(x)*L1(y) + \
          coeffs[int(8 -0.5*min_order**2 - 0.5*min_order)]*L1(x)*L2(y) + \
          coeffs[int(9 -0.5*min_order**2 - 0.5*min_order)]*L0()*L3(y) if (min_order<=3 and max_order>=3) else 0*x

    #order 4
    L4_val = coeffs[int(10 -0.5*min_order**2 - 0.5*min_order)]*L4(x)*L0() + \
          coeffs[int(11 -0.5*min_order**2 - 0.5*min_order)]*L3(x)*L1(y) + \
          coeffs[int(12 -0.5*min_order**2 - 0.5*min_order)]*L2(x)*L2(y) + \
          coeffs[int(13 -0.5*min_order**2 - 0.5*min_order)]*L1(x)*L3(y) + \
          coeffs[int(14 -0.5*min_order**2 - 0.5*min_order)]*L0()*L4(y) if (min_order<=4 and max_order>=4) else 0*x

    #order 5
    L5_val = coeffs[int(15 -0.5*min_order**2 - 0.5*min_order)]*L5(x)*L0() + \
          coeffs[int(16 -0.5*min_order**2 - 0.5*min_order)]*L4(x)*L1(y) + \
          coeffs[int(17 -0.5*min_order**2 - 0.5*min_order)]*L3(x)*L2(y) + \
          coeffs[int(18 -0.5*min_order**2 - 0.5*min_order)]*L2(x)*L3(y) + \
          coeffs[int(19 -0.5*min_order**2 - 0.5*min_order)]*L1(x)*L4(y) + \
          coeffs[int(20 -0.5*min_order**2 - 0.5*min_order)]*L0()*L5(y) if (min_order<=5 and max_order==5) else 0*x

    return  _zeropoint + L0_val + L1_val + L2_val + L3_val + L4_val + L5_val


def get_quat_from_rot(rot):
    quat = np.array([rot[2,1]-rot[1,2],rot[0,2]-rot[2,0],rot[1,0]-rot[0,1],rot[0,0] +rot[1,1] +rot[2,2] +1])
    return quat/np.linalg.norm(quat)


def get_rot_from_quat(qi,qj,qk,q):
    return np.array([[1-2*(qj**2+qk**2),2*(qi*qj-qk*q),2*(qi*qk+qj*q)],
                     [2*(qj*qi + qk*q),1-2*(qi**2+qk**2),2*(qj*qk-qi*q)],
                     [2*(qk*qi - qj*q),2*(qk*qj+qi*q),1-2*(qi**2+qj**2)]])


def get_rot_from_angles(lon,lat,rot):
    #trigonometry
    ca = jnp.cos(lon)
    sa = jnp.sin(lon)
    cd = jnp.cos(-lat)
    sd = jnp.sin(-lat)
    cr = jnp.cos(rot)
    sr = jnp.sin(rot)
    return np.array([[ca*cd,sa*cd,-sd],
                    [(ca*sd*sr-sa*cr),(ca*cr+sa*sd*sr),cd*sr],
                     [(sa*sr+ca*sd*cr),(sa*sd*cr-ca*sr),cd*cr]])


def get_angles_from_rot(rot):
    coslat = np.sqrt(rot[1,2]**2 + rot[2,2]**2)
    lat = np.arctan2(rot[0,2],coslat)
        #careful with cases where coslat = 0
    if coslat==0:
        lon = np.arctan2(rot[0,1],rot[0,0])
        rot = np.arctan2(rot[1,2],rot[2,2])
    else:
        lon = np.arctan2(rot[0,1]/coslat,rot[0,0]/coslat)
        rot = np.arctan2(rot[1,2]/coslat,rot[2,2]/coslat)
    return np.array([lon,lat,rot])



def rotate_from_angles(ra,dec,lon,lat,rot):
    #trigonometry
    ca = np.cos(lon)
    sa = np.sin(lon)
    cd = np.cos(-lat)
    sd = np.sin(-lat)
    cr = np.cos(rot)
    sr = np.sin(rot)
    cp = np.cos(ra)
    sp = np.sin(ra)
    cl = np.cos(dec)
    sl = np.sin(dec)
    #rotated vector
    u0 = (cp*cl)*ca*cd            + (sp*cl)*sa*cd            - (sl)*sd
    u1 = (cp*cl)*(ca*sd*sr-sa*cr) + (sp*cl)*(ca*cr+sa*sd*sr) + (sl)*cd*sr
    u2 = (cp*cl)*(sa*sr+ca*sd*cr) + (sp*cl)*(sa*sd*cr-ca*sr) + (sl)*cd*cr
    
    return np.array([u0,u1,u2])


def rotate_from_rot(ra,dec,rot):
    v = np.array([np.cos(ra)*np.cos(dec),
              np.sin(ra)*np.cos(dec),
              np.sin(dec)])
    return rot@v

def rotate_from_quat(ra,dec,quat):
    v = np.array([np.cos(ra)*np.cos(dec),
              np.sin(ra)*np.cos(dec),
              np.sin(dec)])
    q_inv = R.from_quat(quat)
    return q_inv.apply(v)


def prepare_quat(rx,ry,angle):
    if (rx**2+ry**2)>1:
        #To-Do: either FIX THIS or constrain the gradient descent
        raise ValueError("There is an issue with your definition of quaternion.")
    rz = np.sqrt(1-rx**2-ry**2)
    
    return np.array([rx*np.sin(angle/2),
                         ry*np.sin(angle/2),
                         rz*np.sin(angle/2),
                         np.cos(angle/2)])

def invert_quat(quat):
    quat_norm = quat/np.linalg.norm(quat)
    quat_inv = quat_norm
    quat_inv[:-1] = -quat_norm[:-1]
    return quat_inv

def _transform_quaternion_to_angles(quaternion):
    #1) create inverse quaterion 
    quat_inv = invert_quat(quaternion)

    #2) create rotation array
    rot = get_rot_from_quat(*quat_inv)

    #3) transform into three angles
    lon,lat,rot = get_angles_from_rot(rot)
                           
    return lon,lat,rot

def quaternion2angles(quat_array):
    return np.array(list(map(_transform_quaternion_to_angles,quat_array)))


def read_djl_file(filename,sep=",",att_sep = ",",offset=4,mission_fieldname = "Mission",cal_fieldname = "Calibration",
                  eta_fieldname = "eta",zeta_fieldname = "zeta",rotator_fieldname = "R",xc_fieldname="xC",yc_fieldname="yC",
                 kappa0_fieldname="kappa0",mu0_fieldname="mu0",nCol_fieldname="nCol",nRow_fieldname="nRow",
                 kappaC_fieldname="kappaC",muC_fieldname="muC",srcid_fieldname="SourceID", expid_fieldname="ExposureID", 
                   calid_fieldname="ExposureID", detid_fieldname="DetectorID", srcid_index_name="srcid_index", 
                   expid_index_name="expid_index", calid_index_name="calid_index", detid_index_name="detid_index",
                  axis_index_name="axis_index",obs_index_name="obs_index",
                  observations_ordering="c"):

    """
    Read files created by DJ Legendre: https://scivi.tools/djlegendre/

    NOTE: it assumes that the first column is the attitude, that the last two are the osberved values, and that the
    two before the observations are the source parameters. All other columns can come in any order, as their position
    is given as an input to the function.

    Inputs:
        - observations_ordering: either "c" or "r" for, respectively, storing the observations horizontaly (i.e. along columns in a single row) or vertically (i.e. along alternating rows in a single column ). If "r" is used, then axis_index indicates the column containing the "axis", which identifies which axis the observations belong to. Otherwise, axis_index is None.

    Output:
        - src: array containing the source parameters [srcid,ra,dec]
        - att: array containing the attitude parameters [expid,ra_tel,dec_tel,rot_tel]
        - cal: array containing the calibration parameters [calid,As,Bs]
        - obs: array containing the observations [srcid,expid,calid,detid,axis,obs]
        - Model: dictionary containing the metaparameters of the model used
        - detectors: array containing the metaparameters describing each detector
            (information also available in "Model")
    """
    
    if att_sep	== sep:
        offset = offset
    else:
        offset = 0
    
    data = []
    with open(filename,"r") as f:
        for i,line in enumerate(f.readlines()):
            if i==0:
                model_properties = ast.literal_eval(line)
            elif i==1:
                columns = line.replace(" ","").replace("\n","").split(sep)
            else:
                line_split = line.split(sep)
                if att_sep	== sep:
                    attitude_i = np.array([re.sub("[^0-9.]", "", a) for a in line_split[:offset]]).astype(float)
                else:
                    attitude_i = np.array([re.sub("[^0-9.]", "", a) for a in line_split[0].split(att_sep)]).astype(float)
                ExposureID_i = int(line_split[offset])
                DetectorID_i = int(line_split[1+offset])
                ObservationID_i = int(line_split[2+offset])
                SourceID_i = int(line_split[3+offset])
                Upsilon_i = float(line_split[4+offset])
                Rho_i = float(line_split[5+offset])
                Kappa_i = float(line_split[6+offset])
                Mu_i = float(line_split[7+offset])
                source_i = [*attitude_i,ExposureID_i,DetectorID_i,ObservationID_i,SourceID_i,Upsilon_i,Rho_i,Kappa_i,Mu_i]
                data.append(source_i)

    data = np.array(data)

    #unpack calibration coefficients
    calibration = []
    for c in model_properties[cal_fieldname]:
        c_dict = {}
        for ci in c:
            c_dict.update(dict(ci))
        calibration.append(c_dict)

        #store in separate arrays
    As = []
    Bs = []
    for c in calibration:
        As.append([item for key,item in c.items() if key.startswith(eta_fieldname)])
        Bs.append([item for key,item in c.items() if key.startswith(zeta_fieldname)])


    # create dictionary with all the metaparameters
    model = model_properties[mission_fieldname]

    kappaC = model[kappa0_fieldname] + 0.5*(model[nCol_fieldname] - 1)
    muC = model[mu0_fieldname] + 0.5*(model[nRow_fieldname] - 1)

    model[kappaC_fieldname] = kappaC
    model[muC_fieldname] = muC

    #recycle offset to account for the location of the attitude
    offset = 3

    # define the indexes
    srcid_index = [i for i,col in enumerate(columns) if col.find(srcid_fieldname)>=0]
    if len(srcid_index)==0:
        raise ValueError("Couldn't find the Source ID column! Name provided: {}".format(srcid_fieldname))
    elif len(srcid_index)>1:
        raise ValueError("Found more than one match for the Source ID column! Name provided: {}".format(srcid_fieldname))
    srcid_index = srcid_index[0] + offset #due to the first four being the attitude
    
    #model[srcid_index_name] = srcid_index

    expid_index = [i for i,col in enumerate(columns) if col.find(expid_fieldname)>=0]
    if len(expid_index)==0:
        raise ValueError("Couldn't find the Exposure ID column! Name provided: {}".format(expid_fieldname))
    elif len(expid_index)>1:
        raise ValueError("Found more than one match for the Exposure ID column! Name provided: {}".format(expid_fieldname))
    expid_index = expid_index[0] + offset #due to the first four being the attitude
    #model[expid_index_name] = expid_index

    calid_index = [i for i,col in enumerate(columns) if col.find(calid_fieldname)>=0]
    if len(calid_index)==0:
        raise ValueError("Couldn't find the Calibration unit ID column! Name provided: {}".format(calid_fieldname))
    elif len(calid_index)>1:
        raise ValueError("Found more than one match for the Calibration unit ID column! Name provided: {}".format(calid_fieldname))
    calid_index = calid_index[0] + offset #due to the first four being the attitude
    #model[calid_index_name] = calid_index

    detid_index = [i for i,col in enumerate(columns) if col.find(detid_fieldname)>=0]
    if len(detid_index)==0:
        raise ValueError("Couldn't find the Detector ID column! Name provided: {}".format(detid_fieldname))
    elif len(detid_index)>1:
        raise ValueError("Found more than one match for the Detector ID column! Name provided: {}".format(detid_fieldname))
    detid_index = detid_index[0] + offset #due to the first four being the attitude
    #model[detid_index_name] = detid_index[0]

    #create metaparameter object for each detector
        #assumes that DetectorID are numbered from 0 (THEY ARE NOT IN THE FILES PRODUCED BY DJ LEGENDRE!!)
    detector_ids = np.arange(len(model[rotator_fieldname]))
    xC = np.array(model[xc_fieldname])
    yC = np.array(model[yc_fieldname])
    R = np.array(model[rotator_fieldname])
    detectors = np.column_stack((detector_ids,xC,yC,R))

    #create attitude array
        #assumes that the first 4 columns of data has the attitude
    angles = quaternion2angles(data[:,:4])
    #HACK: we are expecting to get a time associated with each exposure => for now, just put zeros
    time = np.zeros_like(data[:,expid_index])
    att = np.column_stack((data[:,expid_index],time,angles))
    att = np.unique(att,axis=0)
    if len(att)!=len(np.unique(data[:,expid_index])):
        raise ValueError("Something wrong with the exposures! "+\
                         "Got {} attitude parameters but expected to have values for {} exposures".format(len(att),len(np.unique(data[:,expid_index]))))

    #DEBUG
    if False:
        return data,np.column_stack((data[:,srcid_index],data[:,-4:-2])),srcid_index
        
    #create source array
        #assumes that the true source parameters are given in the last fourth and last third columns
    src = np.column_stack((data[:,srcid_index],data[:,-4:-2]))
    src = np.unique(src,axis=0)
    if len(src)!=len(np.unique(data[:,srcid_index])):
        raise ValueError("Something wrong with the sources! "+\
                         "Got {} source parameters but expected to have values for {} sources".format(len(src),len(np.unique(data[:,srcid_index]))))
    
    #create calibration array
        ##assumes that the Legendre coefficients given in the header are ordered in increasing
        ## values of calibration unit ID
    cal = np.column_stack((np.unique(data[:,calid_index]),np.array(As),np.array(Bs)))
    
    #create observations array
        #assumes that the observations are given in the last two columns
    if observations_ordering=="r":
        #use ravel to alternate between observations along axis 0 and 1
        o = data[:,-2:].ravel(order="C")
        axis = np.tile([0,1],len(data))
        #put observations in two columns
    elif observations_ordering=="c":
        o = data[:,-2:]
    else:
        raise ValueError("Invalid observations_ordering provided. Expected either 'r' or 'c' but got {}.".format(observations_ordering))
    
    src_ids = data[:,srcid_index]
    exp_ids = data[:,expid_index]
    cal_ids = data[:,calid_index]
    det_ids = data[:,detid_index] - 1 #(IN THE FILES PRODUCED BY DJ LEGENDRE TOOL THEY START AT 1!!)
    if observations_ordering=="r":
        #if observations in one column, need to repeat the ids
            #if observations come as 0,1,0,1,0,1 use np.repeat
        ids = np.repeat(np.column_stack((src_ids,exp_ids,cal_ids,det_ids)),2,0)
        model[axis_index_name] = 4
        model[obs_index_name] = 5
    elif observations_ordering=="c":
        ids = np.column_stack((src_ids,exp_ids,cal_ids,det_ids))
        model[axis_index_name] = None
        model[obs_index_name] = 4
        #since we forced the ordering, we need to store this information in model
    model[srcid_index_name] = 0
    model[expid_index_name] = 1
    model[calid_index_name] = 2
    model[detid_index_name] = 3

    if observations_ordering=="r":
        obs = np.column_stack((ids,axis,o))
    elif observations_ordering=="c":
        obs = np.column_stack((ids,o))
    
        
    return src,att,cal,obs,model,detectors


def read_ephemeris(filename):
    return pd.read_csv(filename).values
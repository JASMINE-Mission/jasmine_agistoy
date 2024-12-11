from .utils import read_djl_file,read_ephemeris
from .iterativesolver import bloc_iteration
from .propagation import _icrs2comrs,_comrs2fovrs,_fovrs2fprs,_fprs2drs

import time
import numpy as np
import jax
import jax.numpy as jnp

import argparse
try:
    from ConfigParser import RawConfigParser  # python 2
except ImportError:
    from configparser import RawConfigParser  # python 3

def read_ini(filename):
    ini = RawConfigParser()
    ini.optionxform=str  # do not convert key to lowercase
    ini.read(filename)
    return ini

def read_inputs():
    parser = argparse.ArgumentParser(description='Run JARBIS on a file containing observations.')
    
    parser.add_argument('observations_file', type=str, help='Name of the file to process (.csv)')
    parser.add_argument('ephemeris_file', type=str, help='Name of the file to containing the ephemeris (.csv)')
    parser.add_argument('residuals_folder',type=str,help="Path to the folder in which to store the residuals obtain at each stage of the bloc iteration.")
    parser.add_argument('solution_folder',type=str,help="Path to the folder in which to store the solution obtain for each stage of the bloc iteration.")
    parser.add_argument('suffix',type=str,help="Sufix to add to the name of each stored file.")

    parser.add_argument('--separator', type=str, default=",", help='Separator: "," for .csv files')
    parser.add_argument('--attitude_separator', type=str, default=",", help='Separator between the components of the attitude')
    parser.add_argument('--mission_fieldname', type=str, default="Mission", help='Key name in the first row of infile containing the mission definition parameters')
    parser.add_argument('--legendre_fieldname', type=str, default="Calibration", help='Key name in the first row of infile containing the Legendre coefficients used to simulate the data')
    parser.add_argument('--legendre_min_order', type=int, default=1, help='Minimum order of Legendre polynomials to consider (cannot be lower than 0)')
    parser.add_argument('--legendre_max_order', type=int, default=5, help='Maximum order of Legendre polynomials to consider (cannot be larger than 5)')
    parser.add_argument('--eta_fieldname', type=str, default="eta", help='Key name to find the A (horizontal) coeffients within the Legendre coefficient dictionary')
    parser.add_argument('--zeta_fieldname', type=str, default="zeta", help='Key name to find the B (vertical) coeffients within the Legendre coefficient dictionary')
    parser.add_argument('--rotator_fieldname', type=str, default="R", help='Field name that contains the rotators of the detectors')
    parser.add_argument('--F_fieldname', type=str, default="F", help='Field name that contains the Focal length')
    parser.add_argument('--xc_fieldname', type=str, default="xC", help='Field name that contains the x-coord FPRS of each detector centre')
    parser.add_argument('--yc_fieldname', type=str, default="yC", help='Field name that contains the y-coord FPRS of each detector centre')
    parser.add_argument('--kappa0_fieldname', type=str, default="kappa0", help='Position in pixels of the first usable column of the detectors')
    parser.add_argument('--mu0_fieldname', type=str, default="mu0", help='Position in pixels of the first usable row of the detectors')
    parser.add_argument('--nCol_fieldname', type=str, default="nCol", help='Number of usable columns in one detector')
    parser.add_argument('--nRow_fieldname', type=str, default="nRow", help='Number of usable rows in one detector')
    parser.add_argument('--kappaC_fieldname', type=str, default="kappaC", help='Horizontal pixel position of the centre of the detector in the DRS.')
    parser.add_argument('--muC_fieldname', type=str, default="muC", help='Vertical pixel position of the centre of the detector in the DRS.')
    parser.add_argument('--srcid_fieldname', type=str, default="SourceID", help='Name of the column in observations_file containing the sourceID')
    parser.add_argument('--expid_fieldname', type=str, default="ExposureID", help='Name of the column in observations_file containing the exposureID')
    parser.add_argument('--calid_fieldname', type=str, default="CalibrationID", help='Name of the column in observations_file containing the calibration_unitID')
    parser.add_argument('--detid_fieldname', type=str, default="DetectorID", help='Name of the column in observations_file containing the detectorID')
    parser.add_argument('--srcid_index_name', type=str, default="srcid_index", help='Keyword used by the solver functions. CAREFUL! THESE ARE HARDCODED!')
    parser.add_argument('--expid_index_name', type=str, default="expid_index", help='Keyword used by the solver functions. CAREFUL! THESE ARE HARDCODED!')
    parser.add_argument('--calid_index_name', type=str, default="calid_index", help='Keyword used by the solver functions. CAREFUL! THESE ARE HARDCODED!')
    parser.add_argument('--detid_index_name', type=str, default="detid_index", help='Keyword used by the solver functions. CAREFUL! THESE ARE HARDCODED!')
    parser.add_argument('--axis_index_name', type=str, default="axis_index", help='Keyword used by the solver functions. CAREFUL! THESE ARE HARDCODED!')
    parser.add_argument('--obs_index_name', type=str, default="obs_index", help='Keyword used by the solver functions. CAREFUL! THESE ARE HARDCODED!')
    parser.add_argument('--offset', type=int, default=4, help='Expected number of elements in the attitude provided in observations_file')

    parser.add_argument('--src_niter', type=int, default=1, help='Number of iterations in the source parameter bloc')
    parser.add_argument('--att_niter', type=int, default=1, help='Number of iterations in the attitude parameter bloc')
    parser.add_argument('--cal_niter', type=int, default=1, help='Number of iterations in the calibration parameter bloc')
    parser.add_argument('--niter', type=int, default=1, help='Number of iterations in the outer loop')
    parser.add_argument('--min_nobs', type=int, default=10, help='Minimum number of observations of a source to attempt a solution')
    parser.add_argument('--min_nstars', type=int, default=100, help='Minimum number of sources in an exposure to attempt a solution')
    parser.add_argument('--min_nobs_cal', type=int, default=200, help='Minimum number of observations in a calibration unit to attempt a solution')

    parser.add_argument('--observations_ordering', type=str, default="c", help='Whether to store the observations along columns (c) or along rows (r). If (c), then each observations has one row, if (r) then the observations are all in one column.')
    parser.add_argument('--priors', action=argparse.BooleanOptionalAction, help='Whether to use priors in the Source update or not.')
    parser.set_defaults(priors=False)

    parser.add_argument('--extension', type=str,default=".csv",help="Extension of the output files (for now, only CSV files are supported!).")

    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    tstart = time.time()

    #read input from the terminal
    args = read_inputs()   
    print("Read terminal")

        #read input file
    src,att,cal,obs,telescope,detectors = read_djl_file(args.observations_file,
                                                        args.separator,
                                                        args.attitude_separator,
                                                        args.offset,
                                                        args.mission_fieldname,
                                                        args.legendre_fieldname,
                                                        args.eta_fieldname,
                                                        args.zeta_fieldname,
                                                        args.rotator_fieldname,
                                                        args.xc_fieldname,
                                                        args.yc_fieldname,
                                                        args.kappa0_fieldname,
                                                        args.mu0_fieldname,
                                                        args.nCol_fieldname,
                                                        args.nRow_fieldname,
                                                        args.kappaC_fieldname,
                                                        args.muC_fieldname,
                                                        args.srcid_fieldname,
                                                        args.expid_fieldname,
                                                        args.calid_fieldname,
                                                        args.detid_fieldname,
                                                        args.srcid_index_name,
                                                        args.expid_index_name,
                                                        args.calid_index_name,
                                                        args.detid_index_name,
                                                        args.axis_index_name,
                                                        args.obs_index_name)
    print("Read observations file")

    #store output folders
    telescope["residuals_folder"] = args.residuals_folder
    telescope["src_output_filename"] = args.solution_folder + "/src_solution_" + args.suffix + args.extension 
    telescope["att_output_filename"] = args.solution_folder + "/att_solution_" + args.suffix + args.extension 
    telescope["cal_output_filename"] = args.solution_folder + "/cal_solution_" + args.suffix + args.extension

    #check number of Legendre coefficients
    num_coef = int(((args.legendre_max_order+1)*(2 + args.legendre_max_order) - args.legendre_min_order*(1 + args.legendre_min_order))/2)
    provided_coefs = (cal.shape[1]-1)/2
    if np.abs(provided_coefs - int(provided_coefs))!=0:
        raise ValueError("The number of coefficients provided for the horizontal and vertical direction do not match!")
    else:
        provided_coefs = int(provided_coefs)
    if provided_coefs == 21:
        print("21 Legendre coefficients provided.")
        print("It has been specified that only orders between {} and {} will be computed.".format(args.legendre_min_order,
                                                                                                  args.legendre_max_order))
        if num_coef<21:
            deficit = "({} less than 21) => Proceeding to remove the coefficients that will not be used!".format(21-num_coef)
            cal_id = cal[:,0]
            first_index = int((args.legendre_min_order*(1 + args.legendre_min_order))/2)
            calA = cal[:,1+first_index:1+first_index+num_coef]
            second_index = first_index + 21
            calB = cal[:,1+second_index:1+second_index+num_coef]
            cal = np.column_stack((cal_id,calA,calB))
        else:
            deficit = "(All accounted for!)"
        print("The number of coefficients to be calibrated is {}".format(num_coef)+deficit)
    else:
        print("{} Legendre coefficients provided.".format(provided_coefs))
        print("The number of expected coefficients is {}.".format(num_coef))
        if num_coef!=provided_coefs:
            raise ValueError("I must appologise, but I am not smart enough to figure out which coefficients to remove with the currently available information.")

    print("First row of the Calibration array (for visual check)")
    print(cal[0])

        #read ephemeris file
    ephemeris = read_ephemeris(args.ephemeris_file)

    if ephemeris.shape[1]!=6:
        raise ValueError("Ephemeris should be an array containing only 3 positions and 3 velocities!")
    if ephemeris.shape[0]!=len(att):
        raise ValueError("Ephemeris should contain the same number of entries as the Attitude array!")
    print("Read ephemeris file")

    #define model
        #HACK: HARDCODED MODEL! THIS SHOULD BE POSSIBLE TO DEFINE WITH THE INPUTS!
            #model for distortion only
    model = jax.jit(lambda s,a,c,t,e,did: _fprs2drs(s[0],s[1],c,did,telescope,jnp.array(detectors),min_order=args.legendre_min_order,max_order=args.legendre_max_order))

            #complete model without relativistic effects
                #NOTE: the focal length is NOT an adjustable parameter but instead the nominal value is used
                #TO-DO: add zeropoint as sum of "c" coefficients. Right now, the _fovrs2fprs only accepts one zpt, but two should be given
    #model = jax.jit(lambda s,a,c,t,e,did: _fprs2drs(_fovrs2fprs(*_comrs2fovrs(s[0],s[1],a[0],a[1],a[2]),1),c,did,telescope,jnp.array(detectors),min_order=args.legendre_min_order,max_order=args.legendre_max_order))

            #complete model with Ftheta projection
                #NOTE: the focal length is NOT an adjustable parameter but instead the nominal value is used
                #TO-DO: add zeropoint as sum of "c" coefficients. Right now, the _fovrs2fprs only accepts one zpt, but two should be given
    #model = jax.jit(lambda s,a,c,t,e,did: _fprs2drs(*_comrs2fovrs(s[0],s[1],a[0],a[1],a[2]),c,did,telescope,jnp.array(detectors),min_order=1,max_order=5))
    print("Created model")

    #run JARBIS
        #set initial guess for the calibration parameters to 0
    cal0 = cal + 0
    cal0[:,1:] = 0
        #actual run
    src_opt,att_opt,cal_opt = bloc_iteration(src,att,cal0,ephemeris,obs,model,telescope,detectors,
                    src_niter = args.src_niter,att_niter=args.att_niter,cal_niter=args.cal_niter,niter=args.niter,
                    _min_nobs=args.min_nobs,_min_nstars=args.min_nstars,_min_nobs_cal=args.min_nobs_cal)
    print("Ran JARBIS")

    #write results
    np.savetxt(telescope["src_output_filename"],src_opt,header="sourceID,ra0,dec0",delimiter=",")

    np.savetxt(telescope["att_output_filename"],att_opt,header="exposureID,time,ra_tel0,dec_tel0,rot_tel0",delimiter=",")

    coeffA_names = ','.join(["A"+str(order - j)+str(j) for order in np.arange(args.legendre_min_order,args.legendre_max_order+1,1)\
                           for j in np.arange(0,order+1,1)])
    coeffB_names = coeffA_names.replace("A","B")
    np.savetxt(telescope["att_output_filename"],cal_opt,header="calibrationID,"+coeffA_names+","+coeffB_names,delimiter=",")

    print("JARBIS has finished. It took {} seconds.".format(time.time()-tstart))
from lmfit import Model
from lmfit import Parameters, fit_report, minimize
import numpy as np
from lmfit import Model

"""
             ____            _   _               _ 
            / ___|__ _ _   _| |_(_) ___  _ __   | |
           | |   / _` | | | | __| |/ _ \| '_ \  | |
           | |__| (_| | |_| | |_| | (_) | | | | |_|
            \____\__,_|\__,_|\__|_|\___/|_| |_| (_)

 _____                      _                      _        _ 
| ____|_  ___ __   ___ _ __(_)_ __ ___   ___ _ __ | |_ __ _| |
|  _| \ \/ / '_ \ / _ \ '__| | '_ ` _ \ / _ \ '_ \| __/ _` | |
| |___ >  <| |_) |  __/ |  | | | | | | |  __/ | | | || (_| | |
|_____/_/\_\ .__/ \___|_|  |_|_| |_| |_|\___|_| |_|\__\__,_|_|
           |_|                                                
 ____                 _                                  _   
|  _ \  _____   _____| | ___  _ __  _ __ ___   ___ _ __ | |_ 
| | | |/ _ \ \ / / _ \ |/ _ \| '_ \| '_ ` _ \ / _ \ '_ \| __|
| |_| |  __/\ V /  __/ | (_) | |_) | | | | | |  __/ | | | |_ 
|____/ \___| \_/ \___|_|\___/| .__/|_| |_| |_|\___|_| |_|\__|
                             |_|                             
                             
                             
"""

def read_imfit_params(fileParams):
    dlines = [line for line in open(fileParams) if
              len(line.strip()) > 0 and line[0] != "#"]
    values=[]
    temp=[]
    for line in dlines:
        if line.split()[0] == 'FUNCTION' or line.split()[0] == 'GAIN' or \
                line.split()[0] == 'READNOISE':
            pass
        else:
#             print(float(line.split()[1]))
            temp.append(float(line.split()[1]))
        if line.split()[0]=='R_e':
    #         values['c1'] = {}
            values.append(np.asarray(temp))
            temp = []

    if dlines[-2].split()[1]=='FlatSky':
        values.append(np.asarray(float(dlines[-1].split()[1])))
    return(values)



def bn(n):
    """
    bn function from Cioti .... (1997);
    Used to define the relation between Rn (half-light radii) and total luminosity

    Parameters:
        n: sersic index
    """

    return 2. * n - 1. / 3. + 0 * ((4. / 405.) * n) + ((46. / 25515.) * n ** 2.0)


def sersic2D(xy, x0, y0, PA, ell, n, In, Rn):
    q = 1 - ell
    x, y = xy
    # x,y   = np.meshgrid(np.arange((size[1])),np.arange((size[0])))
    xx, yy = rotation(PA, x0, y0, x, y)
    # r     = (abs(xx)**(c+2.0)+((abs(yy))/(q))**(c+2.0))**(1.0/(c+2.0))
    r = np.sqrt((abs(xx) ** (2.0) + ((abs(yy)) / (q)) ** (2.0)))
    model = In * np.exp(-bn(n) * ((r / (Rn)) ** (1.0 / n) - 1.))
    return (model)


def FlatSky(data_level, a):
    return (a * data_level)

def setup_model_components(n_components=2):
    """
        Set up a single sersic component or a composition of n-sersic components.

        Uses the LMFIT objects to easily create model components.

        fi_ is just a prefix to distinguish parameter names.


    """
    if n_components == 1:
        smodel2D = Model(sersic2D, prefix='f1_') + Model(FlatSky, prefix='s_')
    if n_components > 1:
        smodel2D = Model(sersic2D, prefix='f1_')
        for i in range(2, n_components + 1):
            smodel2D = smodel2D + Model(sersic2D, prefix='f' + str(i) + '_')
        smodel2D = smodel2D + Model(FlatSky, prefix='s_')
    return (smodel2D)


def construct_model_parameters(params_values_init, n_components=3,
                               constrained=True,
                               init_params=0.25, final_params=4.0):
    if n_components is None:
        n_components = len(params_values_init) - 1

    smodel2D = setup_model_components(n_components=n_components)
    # print(smodel2D)
    model_temp = Model(sersic2D)
    dr = 3

    # params_values_init = [] #grid of parameter values, each row is the
    # parameter values of a individual component
    for i in range(0, n_components):
        x0, y0, PA, ell, n, In, Rn = params_values_init[i]

        if constrained == True:
            for param in model_temp.param_names:
                # print(param)
                smodel2D.set_param_hint('f' + str(i + 1) + '_' + param,
                                        value=eval(param),
                                        min=init_params * eval(param),
                                        max=final_params * eval(param))
                if param == 'n':
                    if i + 1 == 1:
                        print('Fixing sersic index of core to 0.5 ')
                        smodel2D.set_param_hint('f' + str(i + 1) + '_' + param,
                                                value=0.5, min=0.45, max=0.55)
                    else:
                        smodel2D.set_param_hint('f' + str(i + 1) + '_' + param,
                                                value=eval(param), min=0.4,
                                                max=1.0)
                if param == 'x0':
                    print('Limiting ', param)
                    smodel2D.set_param_hint('f' + str(i + 1) + '_' + param,
                                            value=eval(param),
                                            min=eval(param) - dr,
                                            max=eval(param) + dr)
                if param == 'y0':
                    # print('Limiting ',param)
                    smodel2D.set_param_hint('f' + str(i + 1) + '_' + param,
                                            value=eval(param),
                                            min=eval(param) - dr,
                                            max=eval(param) + dr)
                if param == 'ell':
                    smodel2D.set_param_hint('f' + str(i + 1) + '_' + param,
                                            value=eval(param), min=0.0, max=0.9)
                if param == 'PA':
                    smodel2D.set_param_hint('f' + str(i + 1) + '_' + param,
                                            value=eval(param), min=-180.0,
                                            max=180)
                if param == 'In':
                    smodel2D.set_param_hint('f' + str(i + 1) + '_' + param,
                                            value=eval(param),
                                            min=init_params * eval(param),
                                            max=20 * final_params * eval(param))

            smodel2D.set_param_hint('s_a', value=2, min=0.01, max=5.0)

        if constrained == False:
            for param in model_temp.param_names:
                smodel2D.set_param_hint('f' + str(i + 1) + '_' + param,
                                        value=eval(param), min=0.0)
                if param == 'n':
                    smodel2D.set_param_hint('f' + str(i + 1) + '_' + param,
                                            value=0.5, min=0.3, max=5)
            smodel2D.set_param_hint('s_a', value=2, min=0.00, max=50.0)

    params = smodel2D.make_params()
    print(smodel2D.param_hints)
    return (smodel2D, params)


# @property
def do_fit2D(imagename, params_values_init, ncomponents,
             init_params=0.25, final_params=4.0, constrained=True,
             special_name=''):
    data_2D = pf.getdata(imagename)
    PSF_BEAM = pf.getdata(
        imagename.replace('-image.cutout.fits', '-beampsf.fits'))
    size = data_2D.shape
    xy = np.meshgrid(np.arange((size[1])), np.arange((size[0])))
    background = 0.0
    FlatSky_level = mad_std(data_2D)
    nfunctions = ncomponents

    def residual_2D(params):
        dict_model = {}
        model = 0
        for i in range(1, nfunctions + 1):
            model = model + sersic2D(xy, params['f' + str(i) + '_x0'],
                                     params['f' + str(i) + '_y0'],
                                     params['f' + str(i) + '_PA'],
                                     params['f' + str(i) + '_ell'],
                                     params['f' + str(i) + '_n'],
                                     params['f' + str(i) + '_In'],
                                     params['f' + str(i) + '_Rn'])
        # print(model.shape)
        model = model + FlatSky(FlatSky_level, params['s_a'])

        MODEL_2D_conv = scipy.signal.fftconvolve(model, PSF_BEAM, 'same')
        residual = data_2D - MODEL_2D_conv + background  # - FlatSky(FlatSky_level, params['s_a'])
        return (residual)

    smodel2D, params = construct_model_parameters(params_values_init,
                                                  n_components=nfunctions,
                                                  init_params=init_params,
                                                  final_params=final_params,
                                                  constrained=constrained)

    mini = lmfit.Minimizer(residual_2D, params, max_nfev=10000,
                           nan_policy='omit', reduce_fcn='neglogcauchy')

    # initial minimization.
    method1 = 'differential_evolution'

    # take parameters from previous run, and re-optimize them.
    method2 = 'least_squares'

    if method1 == 'nelder':
        result_1 = mini.minimize(method='nelder')

    if method1 == 'differential_evolution':
        result_1 = mini.minimize(method='differential_evolution',
                                 options={'maxiter': 30000, 'workers': -1,
                                          'tol': 0.001, 'vectorized': True,
                                          'updating': 'deferred',
                                          'seed': 1}
                                 )

    if method2 == 'ampgo':
        result = mini.minimize(method='nelder',params=result_1.params,
                               options={'maxiter': 30000, 'maxfev' : 30000,
                                        'xatol': 1e-11, 'fatol': 1e-11,
                                        'disp' : True})

    if method2 == 'ampgo':
        #ampgo is not workin well. ???
        result = mini.minimize(method='ampgo',params=result_1.params,
                               maxfunevals=10000,totaliter=30,disp=True,
                               maxiter=5,glbtol=1e-8)

    if method2 == 'least_squares':
        #faster, usually converges and provides errors.
        #Good if used in second step (here).
        result = mini.minimize(method='least_squares', params=result_1.params,
                               max_nfev=30000,
                               tr_solver="exact", tr_options={'regularize': True},
                               # x_scale='jac',
                               ftol=1e-11, xtol=1e-11, gtol=1e-11, verbose=2,
                               loss="cauchy")  # ,f_scale=0.5, max_nfev=5000, verbose=2)
        #     # result = mini.minimize(method='least_squares',  params=result_1.params,
        #                        tr_solver="exact", tr_options={'regularize':True}, x_scale='jac',
        #                        ftol=1e-10,xtol=1e-10, gtol=1e-10, loss="cauchy",f_scale=0.5, max_nfev=5000, verbose=2)

    params = result.params

    # initialize the model
    model_temp = Model(sersic2D)
    model = 0
    size = ctn(crop_image).shape
    xy = np.meshgrid(np.arange((size[0])), np.arange((size[1])))
    model_dict = {}
    for i in range(1, ncomponents + 1):
        model_temp = sersic2D(xy, params['f' + str(i) + '_x0'],
                              params['f' + str(i) + '_y0'],
                              params['f' + str(i) + '_PA'],
                              params['f' + str(i) + '_ell'],
                              params['f' + str(i) + '_n'],
                              params['f' + str(i) + '_In'],
                              params['f' + str(i) + '_Rn']) + FlatSky(
            FlatSky_level, params['s_a']) / ncomponents

        model = model + model_temp

        model_dict['model_c' + str(i)] = model_temp
        model_dict['model_c' + str(i) + '_conv'] = scipy.signal.fftconvolve(
            model_temp, PSF_BEAM,
            'same')  # + FlatSky(FlatSky_level, params['s_a'])/ncomponents
        pf.writeto(imagename.replace('.fits', '') + "_" + str(
            ncomponents) + "C_model_component_" + str(
            i) + special_name + '.fits',
                   model_dict['model_c' + str(i) + '_conv'], overwrite=True)
        copy_header(imagename, imagename.replace('.fits', '') + "_" + str(
            ncomponents) + "C_model_component_" + str(
            i) + special_name + '.fits',
                    imagename.replace('.fits', '') + "_" + str(
                        ncomponents) + "C_model_component_" + str(
                        i) + special_name + '.fits')
        pf.writeto(imagename.replace('.fits', '') + "_" + str(
            ncomponents) + "C_dec_model_component_" + str(
            i) + special_name + '.fits', model_dict['model_c' + str(i)],
                   overwrite=True)
        copy_header(imagename, imagename.replace('.fits', '') + "_" + str(
            ncomponents) + "C_dec_model_component_" + str(
            i) + special_name + '.fits',
                    imagename.replace('.fits', '') + "_" + str(
                        ncomponents) + "C_dec_model_component_" + str(
                        i) + special_name + '.fits')

    model = model
    model_dict['model_total'] = model  # + FlatSky(FlatSky_level, params['s_a'])
    model_dict['model_total_conv'] = scipy.signal.fftconvolve(model, PSF_BEAM,
                                                              'same')  # + FlatSky(FlatSky_level, params['s_a'])

    model_dict['best_residual'] = data_2D - model_dict['model_total']
    model_dict['best_residual_conv'] = data_2D - model_dict['model_total_conv']

    pf.writeto(imagename.replace('.fits', '') + "_" + str(
        ncomponents) + "C_model" + special_name + '.fits',
               model_dict['model_total_conv'], overwrite=True)
    pf.writeto(imagename.replace('.fits', '') + "_" + str(
        ncomponents) + "C_residual" + special_name + ".fits",
               model_dict['best_residual_conv'], overwrite=True)
    copy_header(imagename, imagename.replace('.fits', '') + "_" + str(
        ncomponents) + "C_model" + special_name + '.fits',
                imagename.replace('.fits', '') + "_" + str(
                    ncomponents) + "C_model" + special_name + '.fits')
    copy_header(imagename, imagename.replace('.fits', '') + "_" + str(
        ncomponents) + "C_residual" + special_name + '.fits',
                imagename.replace('.fits', '') + "_" + str(
                    ncomponents) + "C_residual" + special_name + '.fits')

    with open(imagename.replace('.fits',
                                '._' + str(ncomponents) + 'C_de_fit.pickle'),
              "wb") as f:
        pickle.dump(result, f)

    return (result, mini, model_dict)





"""
This function accepts IMFIT config input files.


Usage:

    crop_image = # name of image data
    psf_name = # name of psf image data

    data_2D = pf.getdata(crop_image)
    config_file = # name of imfit config file 
    
    PSF_BEAM = pf.getdata(psf_name)
    
    imfit_conf_values = read_imfit_params(config_file)
 
    n_components = len(imfit_conf_values)-1
    construct_model_parameters(imfit_conf_values[0:-1],n_components=n_components,
        init_params = 0.25,final_params = 4.0,constrained=True)
    
    result_mini, mini,model_dict = do_fit2D(imagename=crop_image,
                                        params_values_init = imfit_conf_values[0:-1],
                                        ncomponents=n_components,constrained=True,
                                        init_params = 0.25,final_params = 4.0)
"""
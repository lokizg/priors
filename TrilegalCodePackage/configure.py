def getFileConfig():
    file_config = {}
    file_config['OutputFilePath'] = '/astro/users/danichao/trilegal_maps/strip_0_45_n64/strip_0_45_n64_09'
    file_config['ListFile'] = '/astro/users/danichao/trilegal_maps/strip_0_45_n64/list.strip_0_45_n64_09'

    return file_config

def getConfigParam():
    config_param = {}
    config_param['NSide'] = 64

    config_param['rmag_MIN'] = 13.75
    config_param['rmag_MAX'] = 27.25
    config_param['r_BinSize'] = 0.5
    config_param['rmag_Nsteps'] = int((config_param['rmag_MAX'] - config_param['rmag_MIN'])/config_param['r_BinSize']) + 1

    config_param['FeH_MIN'] = -2.5
    config_param['FeH_MAX'] = 1.0
    config_param['FeH_NPTS'] = 35
    config_param['Mr_FAINT'] = 15.0
    config_param['Mr_BRIGHT'] = -2.0
    config_param['Mr_NPTS'] = 85

    config_param['MapIndex'] = 10

    return config_param

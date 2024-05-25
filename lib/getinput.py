

def get_input_data(domain_file: str, dispersion_file: str) -> tuple[list[float], list[float]]:
    """
    Load domain and meteo characteristics.

    Args:
        domain_file:         containt "space" characteristics of the domain
                                (dimension in each axis, number of nodes in
                                each dimension)
        dispersion_file:     meteo-dispersion characteristics (ambient air
                                temperature, average wind velocity, wind rose)

    Returns:
        domain_params:       domain parameters as a list
        dispersion_params:   dispersion-meteo parameters as a list
    """

    # get all domain parameters
    domain_params = []
    f = open(domain_file, "r")
    for line in f:
        domain_params.append(int(line.strip()))
    f.close()

    # get all dispersion parameters
    dispersion_params = []
    f = open(dispersion_file, "r")
    for line in f:
        dispersion_params.append(float(line.strip()))
    f.close()

    return domain_params, dispersion_params


def get_source_data(source_file: str) -> list[float]:
    """
    Get relevant parameters for source in question.

    Args:
        source_file:     containt all neccessary parameters for source
                            source_x_coor: relative x coordinate of source in domain (number from <0,1> interval)
                            source_y_coor: relative y coordinate of source in domain (number from <0,1> interval)
                            source stack height [m] (format: double)
                            source stack internal diameter (outlet) [m] (format: double)
                            source stack flue gas velocity at the outlet [m/s] (format: double)
                            source stack flue gas temperature at the outlet [°K] (format: double)
                            source emission rate [g/s] (format:double)
                            source emission rate [g/s] (format:double) 
    Returns:
        source_params:   source parameters as a list
    """

    # get all source data from the file
    source_params = []
    f = open(source_file, "r")
    for line in f:
        source_params.append(float(line.strip()))
    f.close()

    return source_params


def get_power_ratios(power_ratios_file: str) -> list[float]:
    """
    Load nominal power outputs of all sources, expressed as <0,1> share of total power output of all sources

    Args:
        power_ratios_file:    file with power ratio of each source (one ratio on eac line, for each source)

    Returns:
        power_ratios_nominal: list of power ratios for each source
    """
    f = open(power_ratios_file, "r")
    power_ratios_nominal = []
    for line in f:
        power_ratios_nominal.append(float(line))

    f.close()
    return power_ratios_nominal


def format_num(num):
    """
    Auxiliary function, converting number to string in form of 0X (in case of single difit number)
    or XX (in case of double digit number)
    """
    if len(str(num)) < 2:
        return "0" + str(num)
    return str(num)


def get_input(source_folder: str, n_distributed_sources: int,
              main_source_name: str = "sourceMain",
              distributed_source_name: str = "sourceDistributed",
              domain_params_name: str = "domain",
              dispersion_params_name: str = "dispersion",
              power_ratios_name: str = "power_ratios_nominal") -> tuple[list[float], list[float], list[list[float]], list[float]]:
    """
    Load all data necessary for main program. 
    Get domain and meteo characteristics.
    Get relevant parameters for main source and all distributed sources.
    Get nominal power outputs of all sources, expressed as <0,1> share of total power output of all sources

    Args:
        source_folder:              name of top folder for input data
        n_distributed_sources:      number of distributed sources
        main_source_name:           prefix of file with data for main source
        distributed_source_name:    prefix of files with data for distributed sources
        domain_params_name:         prefix of file with domain characteristics 
        dispersion_params_name:     prefix of file with meteo characteristic
        power_ration_name:          prefix of file with power ratios

    Returns:
        domain_params:          "space" characteristics of the domain (dimension in each axis, number of nodes in each dimension)
        dispersion_params:      dispersion-meteo parameters as a list
        source_params_all:      list of list of all sources parameters
        power_ratios_nominal:   list of power ratios for each source
    """
    # get domain and dispersion characteristics
    domain_params, dispersion_params = get_input_data(source_folder + domain_params_name + ".txt",
                                                      source_folder + dispersion_params_name + ".txt")
    # get source characteristics
    source_files_names = [source_folder+main_source_name+".txt"]
    for i in range(n_distributed_sources):
        source_files_names.append(
            source_folder + distributed_source_name + "_" + format_num(i+1) + ".txt")

    source_params_all = []
    for file in source_files_names:
        source_params_all.append(get_source_data(file))

    # get nominal power output of all sources
    power_ratios_nominal = get_power_ratios(
        source_folder + power_ratios_name + ".txt")

    return domain_params, dispersion_params, source_params_all, power_ratios_nominal

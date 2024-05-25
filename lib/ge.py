# Module: ge.py
import numpy as np
import scipy.optimize as spopt
import matplotlib.pyplot as plt
from numpy.typing import NDArray


from lib import domain

# ===========================================================================================================================================#
# =================================================DISPERSION=COEFFICIENTS===================================================================#
# ===========================================================================================================================================#


def compute_sigma_zy(x_coor: float, class_indx: float) -> tuple[float, float]:
    """
    Compute dispersion coefficients sigma x and sigma y for the specified stability class, using Gifford's urban dispersion 
    coefficients equation [Baychok. M, Fundamentals of stack gas dispersion, 1979] - L,M,N parameters
    x_coor is defined as downwind distance from the source (x)

    Args: 
        x_coor:      "downwind" distance from the source, point for which dispersion coefficients are computed
        class_indx:  position of stability class in internal structures sigma_z_coef, sigma_y_coef

    Returns:
        sigma_z:    dispersion coefficient in vertical direction
        sigma_y:    dispersion coefficient in "lateral" direction
    """

    # after consideration, I think that having these coefficient in "plain sigth" directly in a code is better for readibility and understanding
    # of code. However, I moved them into stand-alone function and remove repeating of code
    sigma_z_A_coef = [240, 1.0, 0.5]
    sigma_z_B_coef = [240, 1.0, 0.5]
    sigma_z_C_coef = [200, 0.0, 0.0]
    sigma_z_D_coef = [140, 0.3, -0.5]
    sigma_z_E_coef = [80, 1.5, -0.5]
    sigma_z_F_coef = [80, 1.5, -0.5]

    sigma_y_A_coef = [320, 0.4, -0.5]
    sigma_y_B_coef = [320, 0.4, -0.5]
    sigma_y_C_coef = [220, 0.4, -0.5]
    sigma_y_D_coef = [160, 0.4, -0.5]
    sigma_y_E_coef = [110, 0.4, -0.5]
    sigma_y_F_coef = [110, 0.4, -0.5]

    sigma_z_coef = [sigma_z_A_coef, sigma_z_B_coef, sigma_z_C_coef,
                    sigma_z_D_coef, sigma_z_E_coef, sigma_z_F_coef]
    sigma_y_coef = [sigma_y_A_coef, sigma_y_B_coef, sigma_y_C_coef,
                    sigma_y_D_coef, sigma_y_E_coef, sigma_y_F_coef]

    sigma_z = (sigma_z_coef[class_indx][0]*(x_coor/1000)) * \
        (1+sigma_z_coef[class_indx][1] *
         (x_coor/1000))**sigma_z_coef[class_indx][2]
    sigma_y = (sigma_y_coef[class_indx][0]*(x_coor/1000)) * \
        (1+sigma_y_coef[class_indx][1] *
         (x_coor/1000))**sigma_y_coef[class_indx][2]

    return sigma_z, sigma_y


def compute_disp_coef(x_coor: float, stability_class: str) -> tuple[float, float]:
    """
    Compute dispersion coefficients sigma x and sigma y for the specified stability class

    Args:
        x_coor:          "downwind" distance from the source, point for which dispersion coefficients are computed
        stability_class: actual class of atmospheric stability

    Returns:
        sigma_z:    dispersion coefficient in vertical direction
        sigma_y:    dispersion coefficient in "lateral" direction
    """

    match stability_class:
        case "A":
            sigma_z, sigma_y = compute_sigma_zy(x_coor, 0)
        case "B":
            sigma_z, sigma_y = compute_sigma_zy(x_coor, 1)
        case "C":
            sigma_z, sigma_y = compute_sigma_zy(x_coor, 2)
        case "D":
            sigma_z, sigma_y = compute_sigma_zy(x_coor, 0)
        case "E":
            sigma_z, sigma_y = compute_sigma_zy(x_coor, 0)
        case "F":
            sigma_z, sigma_y = compute_sigma_zy(x_coor, 0)

    return sigma_z, sigma_y
# ===========================================================================================================================================#
# =================================================/DISPERSION=COEFFICIENTS==================================================================#
# ===========================================================================================================================================#


# ===========================================================================================================================================#
# ===========================================CONCENTRATION=IN=ONE=POINT======================================================================#
# ===========================================================================================================================================#
def gauss_disp_eq(x_coor: float, y_coor: float, z_coor: float,
                  source_params: list[float],
                  dispersion_params: list[float],
                  stability_class: str) -> float:
    """
    Compute concentration of pollutant at x,y coordinates in source-centered coordinate system, 
    corresponding to the nominal power-output of relevant source.

    Args:
        x_coor:      recpetion point distance downwind from source in km, y = lateral distance from downwind direction through the source, in km
        y_coor:      reception point lateral distance from downwind direction through the source, in km
        z_coor:      reception point horizontal distance from the terrain, in m

    Returns:
        C:          concentration value for specified point [x_coor, y_coor] in micrograms.m-3 (imission limit is formulated in micrograms.m-3)
    """

    # ===================================DISPERSION COEFFIENTS===================================================#
    # Compute vertical and horizontal dispersion coefficients,
    sigma_z, sigma_y = compute_disp_coef(x_coor, stability_class)

    # ===================================/DISPERSION COEFFIENTS===================================================#

    # ===================================EFFECTIVE=STACK=HEIGHT===================================================#
    # Compute effective stack height using Briggs equation for bent-over, buoyant plume [Baychok. M, Fundamentals of stack gas dispersion, 1979]
    F = 9.807 * source_params[4] * (source_params[3]**2) * \
        ((source_params[5] - dispersion_params[0])/source_params[5])
    eff_plume_height = 1.6 * \
        np.power(F, 1/3) * np.power(x_coor, 2/3) * (1/dispersion_params[1])
    # ===================================/EFFECTIVE=STACK=HEIGHT===================================================#

    # ===================================COMPUTE=CONCENTRATION========================================================#
    # Compute concentration in point with x,y coordinates = x_coor [km], y_coor [km] at the z_coor [m] height above the terrain
    if (x_coor > 0):
        horizontal_disp_term = np.power(
            np.e, (-(y_coor**2)/(2*(np.power(sigma_y, 2)))))
        vertical_disp_term = (np.e**(-((z_coor-eff_plume_height)**2)/(2*(sigma_z**2)))) + (
            np.e**(-((z_coor+eff_plume_height)**2)/(2*(sigma_z**2))))
        C = ((source_params[6])/(dispersion_params[1] * sigma_y *
             sigma_z * 2 * np.pi)) * horizontal_disp_term * vertical_disp_term
    else:
        C = 0
    # converting from g.m-3 to micrograms.m-3 (imission limit is formulated in micrograms.m-3)
    return C*1000000
    # ===================================/COMPUTE=CONCENTRATION========================================================#

# ============================================================================================================================================#
# ===========================================/CONCENTRATION=IN=ONE=POINT======================================================================#
# ============================================================================================================================================#


# ===========================================================================================================================================#
# ==============================================CONCENTRATIONS=ONE=SOURCE=ONE=WIND=DIRECTION=================================================#
# ===========================================================================================================================================#
def gauss_disp_eq_domain(source_params: list[float],
                         z_coor: float,
                         dispersion_params: list[float],
                         domain_params: list[float],
                         wind_direction: str,
                         stability_class: str) -> NDArray[np.float64]:
    """
    Compute concetration values for whole domain, with given parameters (resolution)
    for one source and specified wind direction (partial concentration field relevant corresponding to one wind direction)

    Args:
        source_params:       source parameters as a list
        z_coor:              reception point horizontal distance from the terrain, in m
        dispersion_params:   dispersion-meteo parameters as a list
        domain_params:       "space" characteristics of the domain (dimension in each axis, number of nodes in each dimension)
        wind_direction:      wind direction specification according to standard wind rose (N, NE, E, SE, S, SW, W, NW)  
        stability_class      stability class specification 

    Returns:
        partial_conc_field:   concetration values for whole domain for one source and specified wind direction
    """

    n = domain_params[2]
    partial_conc_field = domain.create_domain_matrix(n)  # create domain
    for i in range(n):
        for j in range(n):
            # get realative domain coord
            x_point_domain_coor = j/(n-1)
            y_point_domain_coor = ((n-1) - i)/(n-1)
            # transform them into source coordinate system
            point_x_coor_source, point_y_coor_source = domain.domain_to_source_coor(
                x_point_domain_coor, y_point_domain_coor, source_params, domain_params, wind_direction)
            # compute actual concentration for the point
            partial_conc_field[i, j] = gauss_disp_eq(
                point_x_coor_source, point_y_coor_source, z_coor, source_params, dispersion_params, stability_class)
    return partial_conc_field
# ===========================================================================================================================================#
# ==============================================/CONCENTRATIONS=ONE=SOURCE=ONE=WIND=DIRECTION================================================#
# ===========================================================================================================================================#


# ===========================================================================================================================================#
# ==============================================CONCENTRATIONS=ONE=SOURCE====================================================================#
# ===========================================================================================================================================#
def gauss_disp_eq_total_conc_field(source_params: list[float],
                                   dispersion_params: list[float],
                                   domain_params: list[float],
                                   stability_class: str) -> NDArray[np.float64]:
    """
    Compute total concentration field - average yearly concentrations due to the operation of one source
    in domain with specified meteo-characteristics 

    Args:
        source_params:       source parameters as a list
        dispersion_params:   dispersion-meteo parameters as a list
        domain_params:       "space" characteristics of the domain (dimension in each axis, number of nodes in each dimension)
        stability_class      stability class specification

    Returns:
        total_conc_field:     average yearly concentrations due to the operation of one source in domain with specified meteo-characteristics
    """
    n = domain_params[2]

    total_conc_field = domain.create_domain_matrix(n)  # create domain

    # Add concetration contribution for each wind direction
    wind_direction = ["N", "NW", "W", "SW", "S", "SE", "E", "NE"]
    disp_indx = 2
    for wind_dir in wind_direction:
        total_conc_field += dispersion_params[disp_indx]*gauss_disp_eq_domain(
            source_params, 2, dispersion_params, domain_params, wind_dir, stability_class)
        disp_indx += 1

    return total_conc_field
# ===========================================================================================================================================#
# ==============================================/CONCENTRATIONS=ONE=SOURCE===================================================================#
# ===========================================================================================================================================#


# ===========================================================================================================================================#
# =================================CONCENTRATIONS=MAIN=SOURCE=DISTRIBUTED=SOURCES=NOMINAL=OUTPUT=============================================#
# ===========================================================================================================================================#
def total_conc_fields_main_distr(source_params_all: list[list[float]], dispersion_params: list[float], domain_params: list[float], stability_class: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute cumulative concentration values for Central heat source and for combination of all distributed heat sources
    (i.e. returns two separate matrices - concentration fields)

    Args:
        source_params_all:       list of list of all sources parameters
        dispersion_params:       dispersion-meteo parameters as a list
        domain_params:           "space" characteristics of the domain (dimension in each axis, number of nodes in each dimension)
        stability_class          stability class specification

    Returns:
        tot_conc_field_main:      cumulative, average yearly concentration values for Central heat source running at nominal output
        tot_conc_field_distrib:    cumulative, average yearly concentration values for all distributed heat sources running at nominal output
    """
    tot_conc_field_main = gauss_disp_eq_total_conc_field(
        source_params_all[0], dispersion_params, domain_params, stability_class)
    tot_conc_field_distrib = gauss_disp_eq_total_conc_field(
        source_params_all[1], dispersion_params, domain_params, stability_class)
    for indx in range(2, len(source_params_all)):
        tot_conc_field_distrib += gauss_disp_eq_total_conc_field(
            source_params_all[indx], dispersion_params, domain_params, stability_class)

    return tot_conc_field_main, tot_conc_field_distrib
# ===========================================================================================================================================#
# =================================/CONCENTRATIONS=MAIN=SOURCE=DISTRIBUTED=SOURCES=NOMINAL=OUTPUT============================================#
# ===========================================================================================================================================#


# ===========================================================================================================================================#
# =======================================CONCENTRATION=IN=ONE=POINT=FROM=ONE=SOURCE==========================================================#
# ===========================================================================================================================================#

# Compute yearly concetration in point x for one source and all wind directions, for one stability class
# Compute concentrations corresponding to the real power-output of relevant source,
# defined as ratio (number from <0,1> interval) of nominal power output
# source_params at the input is already changed proportianally to this real power output
def conc_one_source(x_coor: float, y_coor: float, z_coor: float,
                    source_params: list[float],
                    dispersion_params: list[float],
                    domain_params: list[float],
                    stability_class: str) -> float:
    """
    Compute average yearly concetration in point x for one source and all wind directions, for one stability class
    corresponding to the real power-output of relevant source, defined as ratio (number from <0,1> interval) of nominal power output.

    Args:
        x_coor:              recpetion point distance downwind from source in km, y = lateral distance from downwind direction through the source, in km
        y_coor:              reception point lateral distance from downwind direction through the source, in km
        z_coor:              reception point horizontal distance from the terrain, in m
        source_params_all:   list of list of all sources parameters
        dispersion_params:   dispersion-meteo parameters as a list
        domain_params:       "space" characteristics of the domain (dimension in each axis, number of nodes in each dimension)
        stability_class      stability class specification

    Returns:
        point_conc:     average yearly concetration in point x for one source and all wind directions, for one stability class
                                corresponding to the real power-output of relevant source
    """
    # set initial concentration
    point_conc = 0
    # ransform x_coor and y_coor from domain-coordinate system into source coordinate system
    wind_directions = ["N", "NW", "W", "SW", "S", "SE", "E", "NE"]
    for indx in range(len(wind_directions)):
        point_x_coor_source, point_y_coor_source = domain.domain_to_source_coor(
            x_coor, y_coor, source_params, domain_params, wind_directions[indx])
        point_conc += dispersion_params[indx+2] * gauss_disp_eq(
            point_x_coor_source, point_y_coor_source, z_coor, source_params, dispersion_params, stability_class)

    return point_conc

# ===========================================================================================================================================#
# =======================================/CONCENTRATION=IN=ONE=POINT=FROM=ONE=SOURCE=========================================================#
# ===========================================================================================================================================#

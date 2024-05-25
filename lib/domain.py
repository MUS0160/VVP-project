import numpy as np
from numpy.typing import NDArray


def create_domain_matrix(n: int) -> NDArray[np.float64]:
    """
    Create concentration field matrix, according to domain parameters

    Args:
        n: number of domain nodes in one dimensions

    Returns:
        domain matrix filled with zeros
    """
    return np.zeros(shape=(n, n))


def transform_coor_rot_transl(x: float, y: float, x_source: float, y_source: float, phi: float) -> tuple[float, float]:
    '''
    Auxiliary function, transform coordinates x, y of a given point
    in rotating coordinate system (with angle of rotation = phi)
    together with the translation according to source coordinates

    Args:
        x:          x coordinate in the original system
        y:          y coordinate in the original system
        x_source:   x coordinate of the source in domain
        y_source:   y coordinate of the source in domain  
        phi:        angle of rotation of coordinates system

    Returns:
        x_transformed:  new x coordinate in rotated system
        y_transformed:  new y coordinate in rotated system
    '''
    x_transformed = np.cos(phi)*(x-x_source) + np.sin(phi)*(y-y_source)
    y_transformed = abs((-1)*np.sin(phi)*(x-x_source) +
                        np.cos(phi)*(y-y_source))
    return x_transformed, y_transformed


def domain_to_source_coor(x_point_domain_coor: float, y_point_domain_coor: float, source_params: list, domain_params: list, wind_direction: str) -> tuple[float, float]:
    """
    Transforms domain coordinates into source-based coordinate system.  
    Coordinates in the domain system are defined as number from <0,1>, where 0 is left (upper) edge of a domain
    and 1 is right (lower) edge of the domain. 
    Wind direction is defined as one of the following strings "N", "NW", "W", "SW", "S", "SE", "E" and "NE", 
    ach string represents corresponding direction of standard wind rose (i.e. north, north-west, west and so on) 

    Args:
        x_point_domain_coor:     x coordinates in domain system
        y_point_domain_coor:     y coordinates in domain system
        source_params:           source parameters as a list
        domain_params:           "space" characteristics of the domain (dimension in each axis, number of nodes in each dimension)
        wind_direction:      wind direction specification according to standard wind rose (N, NE, E, SE, S, SW, W, NW)  

    Returns:
        x_coor:      x coordinates in source-based system, distance downwind from source in m, 
        y_coor:      y coordinates in source-based system, lateral distance from downwind direction through the source, in m
    """
    match wind_direction:
        case "N":
            x_source_coor, y_source_coor = transform_coor_rot_transl(
                x_point_domain_coor, y_point_domain_coor, source_params[0], source_params[1], (1/2)*np.pi)
        case "NW":
            x_source_coor, y_source_coor = transform_coor_rot_transl(
                x_point_domain_coor, y_point_domain_coor, source_params[0], source_params[1], (3/4)*np.pi)
        case "W":
            x_source_coor, y_source_coor = transform_coor_rot_transl(
                x_point_domain_coor, y_point_domain_coor, source_params[0], source_params[1],  (1)*np.pi)
        case "SW":
            x_source_coor, y_source_coor = transform_coor_rot_transl(
                x_point_domain_coor, y_point_domain_coor, source_params[0], source_params[1],  (5/4)*np.pi)
        case "S":
            x_source_coor, y_source_coor = transform_coor_rot_transl(
                x_point_domain_coor, y_point_domain_coor, source_params[0], source_params[1],  (3/2)*np.pi)
        case "SE":
            x_source_coor, y_source_coor = transform_coor_rot_transl(
                x_point_domain_coor, y_point_domain_coor, source_params[0], source_params[1],  (7/4)*np.pi)
        case "E":
            x_source_coor, y_source_coor = transform_coor_rot_transl(
                x_point_domain_coor, y_point_domain_coor, source_params[0], source_params[1],  (0)*np.pi)
        case "NE":
            x_source_coor, y_source_coor = transform_coor_rot_transl(
                x_point_domain_coor, y_point_domain_coor, source_params[0], source_params[1],  (1/4)*np.pi)

    return 1000*domain_params[0]*x_source_coor, 1000*domain_params[1]*y_source_coor

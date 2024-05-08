import numpy as np
from numpy.typing import NDArray


def createDomainMatrix(n: int) -> NDArray[np.float64]:
    #create concentration field matrix, according to domain parameters
    return np.zeros(shape = (n,n))

def domainToSourceCoor(x_point_domainCoor: float, y_point_domainCoor: float, sourceParams: list, domainParams: list, windDirection: str ) ->tuple[float, float]:
    #coordinates of point and source are defined as number from <0,1>, where 0 is left (lower) edge of a domain
    #and 1 is right (upper) edge
    #windDirection is defined as one of the following strings "N", "NW", "W", "SW", "S", "SE", "E" and "NE"
    #each string represents corresponding direction of standard wind rose (i.e. north, north-west, west and so on) 
    #Return x and y coordinates in source-wind coordinate system 
    # (x = distance im km down-wind from the source, y = lateral distance from the down-wind line passing through source)

    match windDirection:
        case "N":
            x_sourceCoor = y_point_domainCoor - sourceParams[1]
            y_sourceCoor = abs(x_point_domainCoor - sourceParams[0])
        case "NW":
            x_sourceCoor = (-1)*(np.sqrt(2)/2) * (x_point_domainCoor-sourceParams[0]) + (np.sqrt(2)/2) * (y_point_domainCoor-sourceParams[1])
            y_sourceCoor = (-1)*(np.sqrt(2)/2) * (x_point_domainCoor-sourceParams[0]) + (-1)*(np.sqrt(2)/2) * (y_point_domainCoor-sourceParams[1]) 
        case "W":
            x_sourceCoor = sourceParams[0] - x_point_domainCoor
            y_sourceCoor = abs(y_point_domainCoor - sourceParams[1])
        case "SW":
            x_sourceCoor = (-1)*(np.sqrt(2)/2) * (x_point_domainCoor-sourceParams[0]) + (-1)*(np.sqrt(2)/2) * (y_point_domainCoor-sourceParams[1])
            y_sourceCoor = -(-1)*(np.sqrt(2)/2) * (x_point_domainCoor-sourceParams[0]) + (-1)*(np.sqrt(2)/2) * (y_point_domainCoor-sourceParams[1])
        case "S":
            x_sourceCoor = sourceParams[1] - y_point_domainCoor
            y_sourceCoor = abs(x_point_domainCoor - sourceParams[0])
        case "SE":
            x_sourceCoor = (np.sqrt(2)/2) * (x_point_domainCoor-sourceParams[0]) + (-1)*(np.sqrt(2)/2) * (y_point_domainCoor-sourceParams[1])
            y_sourceCoor = -(-1)*(np.sqrt(2)/2) * (x_point_domainCoor-sourceParams[0]) + (np.sqrt(2)/2) * (y_point_domainCoor-sourceParams[1])
        case "E":
            x_sourceCoor = x_point_domainCoor - sourceParams[0]
            y_sourceCoor = abs(y_point_domainCoor - sourceParams[1])     
        case "NE":
            x_sourceCoor = (np.sqrt(2)/2) * (x_point_domainCoor-sourceParams[0]) + (np.sqrt(2)/2) * (y_point_domainCoor-sourceParams[1])
            y_sourceCoor = (-1)*(np.sqrt(2)/2) * (x_point_domainCoor-sourceParams[0]) + (np.sqrt(2)/2) * (y_point_domainCoor-sourceParams[1])
        
    return 1000*domainParams[0]*x_sourceCoor, 1000*domainParams[1]*y_sourceCoor
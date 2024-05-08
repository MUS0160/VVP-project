# Module: ge.py
import numpy as np
import scipy.optimize as spopt
import matplotlib.pyplot as plt
from numpy.typing import NDArray


from lib import domain


def gaussDispEq(xCoor: float, yCoor: float, zCoor: float, sourceParams: list[float], dispersionParams: list[float], stabilityClass: str) -> float:
    #Compute concentration of pollutant at x,y coordinates in source-centered coordinate system
    #x - distance downwind from source in km, y = lateral distance from downwind direction through the source, in km
    #Return one concentration value for specified point [xCoor, yCoor]

    #===================================DISPERSION COEFFIENTS===================================================#
    #Compute vertical and horizontal dispersion coefficients, 
    #using Gifford's urban dispersion coefficients equation [Baychok. M, Fundamentals of stack gas dispersion, 1979]
    #Gifford's urban dispersion coefficients: L,M,N parameters for A stability class
    #xCoor and yCoor are defined as downwind distance from the source (x) and croswind distance from downwind line passing through the source (y)
    sigma_z_AclassCoef = [240, 1.0, 0.5]
    sigma_z_BclassCoef = [240, 1.0, 0.5]
    sigma_z_CclassCoef = [200, 0.0, 0.0]
    sigma_z_DclassCoef = [140, 0.3, -0.5]
    sigma_z_EclassCoef = [80, 1.5, -0.5]
    sigma_z_FclassCoef = [80, 1.5, -0.5]

    sigma_y_AclassCoef = [320, 0.4, -0.5]
    sigma_y_BclassCoef = [320, 0.4, -0.5]
    sigma_y_CclassCoef = [220, 0.4, -0.5]
    sigma_y_DclassCoef = [160, 0.4, -0.5]
    sigma_y_EclassCoef = [110, 0.4, -0.5]
    sigma_y_FclassCoef = [110, 0.4, -0.5]

    #vertical and horizontal dispersion coeff.
    match stabilityClass:
        case "A":
            sigma_z = (sigma_z_AclassCoef[0]*(xCoor/1000))*(1+sigma_z_AclassCoef[1]*(xCoor/1000))**sigma_z_AclassCoef[2]
            sigma_y = (sigma_y_AclassCoef[0]*(xCoor/1000))*(1+sigma_y_AclassCoef[1]*(xCoor/1000))**sigma_y_AclassCoef[2]
        case "B":
            sigma_z = (sigma_z_BclassCoef[0]*(xCoor/1000))*(1+sigma_z_BclassCoef[1]*(xCoor/1000))**sigma_z_BclassCoef[2]
            sigma_y = (sigma_y_BclassCoef[0]*(xCoor/1000))*(1+sigma_y_BclassCoef[1]*(xCoor/1000))**sigma_y_BclassCoef[2]
        case "C":
            sigma_z = (sigma_z_CclassCoef[0]*(xCoor/1000))*(1+sigma_z_CclassCoef[1]*(xCoor/1000))**sigma_z_CclassCoef[2]
            sigma_y = (sigma_y_CclassCoef[0]*(xCoor/1000))*(1+sigma_y_CclassCoef[1]*(xCoor/1000))**sigma_y_CclassCoef[2]
        case "D":
            sigma_z = (sigma_z_DclassCoef[0]*(xCoor/1000))*(1+sigma_z_DclassCoef[1]*(xCoor/1000))**sigma_z_DclassCoef[2]
            sigma_y = (sigma_y_DclassCoef[0]*(xCoor/1000))*(1+sigma_y_DclassCoef[1]*(xCoor/1000))**sigma_y_DclassCoef[2]
        case "E":
            sigma_z = (sigma_z_EclassCoef[0]*(xCoor/1000))*(1+sigma_z_EclassCoef[1]*(xCoor/1000))**sigma_z_EclassCoef[2]
            sigma_y = (sigma_y_EclassCoef[0]*(xCoor/1000))*(1+sigma_y_EclassCoef[1]*(xCoor/1000))**sigma_y_EclassCoef[2]
        case "F":
            sigma_z = (sigma_z_FclassCoef[0]*(xCoor/1000))*(1+sigma_z_FclassCoef[1]*(xCoor/1000))**sigma_z_FclassCoef[2]
            sigma_y = (sigma_y_FclassCoef[0]*(xCoor/1000))*(1+sigma_y_FclassCoef[1]*(xCoor/1000))**sigma_y_FclassCoef[2]
    #===================================/DISPERSION COEFFIENTS===================================================#

    #===================================EFFECTIVE=STACK=HEIGHT===================================================#
    #Compute effective stack height
    #using Briggs equation for bent-over, buoyant plume [Baychok. M, Fundamentals of stack gas dispersion, 1979]
    F = 9.807 * sourceParams[4] * (sourceParams[3]**2) * ( (sourceParams[5] - dispersionParams[0])/sourceParams[5] )
    effPlumeHeight = 1.6 * np.power(F, 1/3) * np.power(xCoor,2/3) * (1/dispersionParams[1])
    #===================================/EFFECTIVE=STACK=HEIGHT===================================================#

    #===================================COMPUTE=CONCENTRATION========================================================#
    #Compute concentration in point with x,y coordinates = xCoor [km], yCoor [km] at the zCoor [m] height above the terrain
    if(xCoor > 0):
        horizontalDispTerm = np.power(np.e, (-(yCoor**2)/(2*(np.power(sigma_y,2)))))
        verticalDispTerm = ( np.e**(-((zCoor-effPlumeHeight)**2)/(2*(sigma_z**2))) ) + (np.e**(-((zCoor+effPlumeHeight)**2)/(2*(sigma_z**2))))
        C = (sourceParams[6]/(dispersionParams[1] * sigma_y * sigma_z * 2 * np.pi)) * horizontalDispTerm  * verticalDispTerm
    else:
        C = 0
    return C #*1000000 converting from g.m-3 to micrograms.m-3 (imission limit is formulated in micrograms.m-3)
    #===================================/COMPUTE=CONCENTRATION========================================================#

def gaussDispEqDomain(sourceParams: list[float], zCoor: float, dispersionParams: list[float], domainParams: list[float], windDirection: str, stabilityClass: str) -> NDArray[np.float64]:
    #Compute concetration values for whole domain, with given parameters (resolution)
    #Return matrix - concentration field for the given domaian, given wind direction and stability class
    n = domainParams[2]
    partialConcField = domain.createDomainMatrix(n) #create domain
    for i in range(n):
        for j in range(n):
            #get realative domain coord
            x_point_domainCoor = j/(n-1)
            y_point_domainCoor =((n-1)- i)/(n-1)
            #transform them into source coordinate system
            point_xCoorSource, point_ySource = domain.domainToSourceCoor(x_point_domainCoor, y_point_domainCoor, sourceParams, domainParams, windDirection)
            #compute actual concentration for the point
            partialConcField[i,j] = gaussDispEq(point_xCoorSource, point_ySource, zCoor, sourceParams, dispersionParams, stabilityClass)
    return partialConcField

def gaussDispEq_TotalConcField(sourceParams: list[float], dispersionParams: list[float], domainParams: list[float], stabilityClass: str) -> NDArray[np.float64]:
    #Compute cumulative concentration values for whole domain, for every wind direction and for specified stability class
    n = domainParams[2]
    totalConcField = domain.createDomainMatrix(n) #create domain
    #Add concetration contribution for each wind direction
    totalConcField += dispersionParams[2]*gaussDispEqDomain(sourceParams, 2, dispersionParams, domainParams, "N", stabilityClass)
    totalConcField += dispersionParams[3]*gaussDispEqDomain(sourceParams, 2, dispersionParams, domainParams, "NW", stabilityClass)
    totalConcField += dispersionParams[4]*gaussDispEqDomain(sourceParams, 2, dispersionParams, domainParams, "W", stabilityClass)
    totalConcField += dispersionParams[5]*gaussDispEqDomain(sourceParams, 2, dispersionParams, domainParams, "SW", stabilityClass)
    totalConcField += dispersionParams[6]*gaussDispEqDomain(sourceParams, 2, dispersionParams, domainParams, "S", stabilityClass)
    totalConcField += dispersionParams[7]*gaussDispEqDomain(sourceParams, 2, dispersionParams, domainParams, "SE", stabilityClass)
    totalConcField += dispersionParams[8]*gaussDispEqDomain(sourceParams, 2, dispersionParams, domainParams, "E", stabilityClass)
    totalConcField += dispersionParams[9]*gaussDispEqDomain(sourceParams, 2, dispersionParams, domainParams, "NE", stabilityClass)
    
    return totalConcField


def totalConcFields_MainSmall(sourceParams_all: list[list[float]], dispersionParams: list[float], domainParams: list[float], stabilityClass: str ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    #Compute cumulative concentration values for Central heat source and for combination of all distributed heat sources
    #(i.e. return two separate matrices - concentration fields)
    totalConcField_MainSource = gaussDispEq_TotalConcField(sourceParams_all[0], dispersionParams, domainParams, stabilityClass)
    totalConcField_SmallSources = gaussDispEq_TotalConcField(sourceParams_all[1], dispersionParams, domainParams, stabilityClass)
    for indx in range(2,len(sourceParams_all)):
        totalConcField_SmallSources += gaussDispEq_TotalConcField(sourceParams_all[indx], dispersionParams, domainParams, stabilityClass)
    
    return totalConcField_MainSource, totalConcField_SmallSources


def computeCumulativeImission(sourceParams: list[float], dispersionParams: list[float], domainParams: list[float], stabilityClass: str) -> float:
    #Compute cumulative imission concentration (through whole domain) for givenh stability class and for given source (at nominal power) 
    #Cumulative imission means summ of all computed concentrationf for each point in the domain. 
    #Represent the overal imission polution of computed domain.
    # Return cumulative imission as one value, which represents given source and stability class.
    n = domainParams[2]
    concField = gaussDispEq_TotalConcField(sourceParams, dispersionParams, domainParams, stabilityClass)
    imissionCum = 0
    for i in range(n):
        for j in range(n):
            imissionCum += concField[i,j] #TO DO, solve case when actual concentration is higher than imission limit
    return imissionCum

def minimizeImissions(imissionCums: NDArray[np.float64]) -> list[float]:
    #Minimalization of total cumulative imissions for all sources and every stability class, with respect to power output of each source
    #Key constraint is, that combined power output of all sources must be constant (in order to supply necessary amount of heat)
    '''
    There are still some (probably major) problems. Change in resolution of the domain, without any changes in other parameters
    causes changes in optimized solution.  Solution however appears to converge with the higher resolution.
    '''
    #dimension xinit must  be changed when the number of sources is changed.
    #in automated version, number of elements and their values should be initiated according to number of sources and constraints on x
    #CHANGE OF SOURCE COUNT: change dimension of xinit according to source count
    lst = [0.4, 0.6, 0.6, 0.6, 0.6, 0.6]
    #lst = [0.5, 0.5, 0.5]
    xinit = np.array(lst)

    minimizingFunction = lambda x: (np.linalg.norm((imissionCums@x), ord=1))

    #Constraints
    #TO DO: need to rewrite for automatic idettification of number of sources and corresponding number of constraints
    #CHANGE OF SOURCE COUNT: delete or add constraints for each element in optimized x (one element for each source)
    con0 = lambda x: x[0]
    cons0 = spopt.NonlinearConstraint(con0, 0, 1)
    con1 = lambda x: x[1]
    cons1 = spopt.NonlinearConstraint(con1, 0, 1)
    con2 = lambda x: x[2]
    cons2 = spopt.NonlinearConstraint(con2, 0, 1)
    con3 = lambda x: x[3]
    cons3 = spopt.NonlinearConstraint(con3, 0, 1)
    con4 = lambda x: x[4]
    cons4 = spopt.NonlinearConstraint(con4, 0, 1)
    con5 = lambda x: x[5]
    cons5 = spopt.NonlinearConstraint(con5, 0, 1)
    conNonLin = lambda x: 1*x[0]+0.2*x[1]+0.2*x[2]+0.2*x[3]+0.2*x[4]+0.2*x[5]
    #conNonLin = lambda x: 1*x[0]+0.5*x[1]+0.5*x[2]
    consNonLin = spopt.NonlinearConstraint(conNonLin, 1, 1)
    #CHANGE OF SOURCE COUNT: delete or add constraints
    cons =[cons0, cons1, cons2, cons3, cons4, cons5, consNonLin]

    #optimize vector of optimal power outputs for all sources, with respect to minimal cumulative imissions in the domain
    res = spopt.minimize(minimizingFunction, x0=xinit, constraints=cons, method = "trust-constr")

    return res.x

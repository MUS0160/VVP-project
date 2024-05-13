# Module: ge.py
import numpy as np
import scipy.optimize as spopt
import matplotlib.pyplot as plt
from numpy.typing import NDArray


from lib import domain
#===========================================================================================================================================#
#===========================================CONCENTRATION=IN=ONE=POINT======================================================================#
#===========================================================================================================================================#
#Compute concentration of pollutant at x,y coordinates in source-centered coordinate system
#x - distance downwind from source in km, y = lateral distance from downwind direction through the source, in km
#Return one concentration value for specified point [xCoor, yCoor]
#Compute concentrations corresponding to the real power-output of relevant source,
# defined as ratio (number from <0,1> interval) of nominal power output
#sourceParams at the input is already changed proportianally to this real power output
def gaussDispEq(powerOutputRatio:float,
                xCoor: float, yCoor: float, zCoor: float, 
                sourceParams: list[float],
                dispersionParams: list[float], 
                stabilityClass: str) -> float:
    
    #===================================DISPERSION COEFFIENTS===================================================#
    #Compute vertical and horizontal dispersion coefficients, 
    #using Gifford's urban dispersion coefficients equation [Baychok. M, Fundamentals of stack gas dispersion, 1979]
    #Gifford's urban dispersion coefficients: L,M,N parameters for A stability class
    #xCoor and yCoor are defined as downwind distance from the source (x) and croswind distance from downwind line passing through the source (y)
    sigma_z, sigma_y = computeDispersionCoeff(xCoor, stabilityClass)
    
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
        C = ((powerOutputRatio*sourceParams[6])/(dispersionParams[1] * sigma_y * sigma_z * 2 * np.pi)) * horizontalDispTerm  * verticalDispTerm
    else:
        C = 0
    return C #*1000000 #converting from g.m-3 to micrograms.m-3 (imission limit is formulated in micrograms.m-3)
    #===================================/COMPUTE=CONCENTRATION========================================================#

#============================================================================================================================================#
#===========================================/CONCENTRATION=IN=ONE=POINT======================================================================#
#============================================================================================================================================#

#===========================================================================================================================================#
#==============================================CONCENTRATIONS=ONE=SOURCE=ONE=WIND=DIRECTION=================================================#
#===========================================================================================================================================#
def gaussDispEqDomain(sourceParams: list[float], 
                      zCoor: float, 
                      dispersionParams: list[float], 
                      domainParams: list[float], 
                      windDirection: str, 
                      stabilityClass: str) -> NDArray[np.float64]:
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
#===========================================================================================================================================#
#==============================================/CONCENTRATIONS=ONE=SOURCE=ONE=WIND=DIRECTION================================================#
#===========================================================================================================================================#


#===========================================================================================================================================#
#==============================================CONCENTRATIONS=ONE=SOURCE====================================================================#
#===========================================================================================================================================#
def gaussDispEq_TotalConcField(sourceParams: list[float], 
                               dispersionParams: list[float], 
                               domainParams: list[float], 
                               stabilityClass: str) -> NDArray[np.float64]:
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
#===========================================================================================================================================#
#==============================================/CONCENTRATIONS=ONE=SOURCE===================================================================#
#===========================================================================================================================================#



#===========================================================================================================================================#
#=================================CONCENTRATIONS=MAIN=SOURCE=DISTRIBUTED=SOURCES=NOMINAL=OUTPUT=============================================#
#===========================================================================================================================================#
def totalConcFields_MainSmall(sourceParams_all: list[list[float]], dispersionParams: list[float], domainParams: list[float], stabilityClass: str ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    #Compute cumulative concentration values for Central heat source and for combination of all distributed heat sources
    #(i.e. return two separate matrices - concentration fields)
    totalConcField_MainSource = gaussDispEq_TotalConcField(sourceParams_all[0], dispersionParams, domainParams, stabilityClass)
    totalConcField_SmallSources = gaussDispEq_TotalConcField(sourceParams_all[1], dispersionParams, domainParams, stabilityClass)
    for indx in range(2,len(sourceParams_all)):
        totalConcField_SmallSources += gaussDispEq_TotalConcField(sourceParams_all[indx], dispersionParams, domainParams, stabilityClass)
    
    return totalConcField_MainSource, totalConcField_SmallSources
#===========================================================================================================================================#
#=================================/CONCENTRATIONS=MAIN=SOURCE=DISTRIBUTED=SOURCES=NOMINAL=OUTPUT============================================#
#===========================================================================================================================================#




#===========================================================================================================================================#
#=======================================CONCENTRATION=IN=ONE=POINT=FROM=ONE=SOURCE==========================================================#
#===========================================================================================================================================#

#Compute yearly concetration in point x for one source and all wind directions, for one stability class
#Compute concentrations corresponding to the real power-output of relevant source,
# defined as ratio (number from <0,1> interval) of nominal power output
#sourceParams at the input is already changed proportianally to this real power output
def conc_OneSource(powerOutputRatio: float, 
                   xCoor: float, yCoor: float, zCoor: float, 
                   sourceParams: list[float], 
                   dispersionParams: list[float], 
                   domainParams: list[float], 
                   stabilityClass: str)-> float:
    #set initial concentration
    pointConcentration = 0
    #tcompute concentration contribution from source for whole year
    #ransform xCoor and yCoor from domain-coordinate system into source coordinate system
    windDirections = ["N", "NW", "W", "SW", "S", "SE", "E", "NE"]
    for indx in range(len(windDirections)):
        point_xCoorSource, point_ySource = domain.domainToSourceCoor(xCoor, yCoor, sourceParams, domainParams, windDirections[indx])
        pointConcentration += dispersionParams[indx+2] * gaussDispEq(powerOutputRatio, point_xCoorSource, point_ySource, zCoor, sourceParams, dispersionParams, stabilityClass)
        
    return pointConcentration

#===========================================================================================================================================#
#=======================================/CONCENTRATION=IN=ONE=POINT=FROM=ONE=SOURCE=========================================================#
#===========================================================================================================================================#



#===========================================================================================================================================#
#=======================================CONCENTRATION=IN=ONE=POINT=FROM=ALL=SOURCES=========================================================#
#===========================================================================================================================================#
#Compute yearly concetration in point x for all sources and all wind directions, for one stability class, based on real power outout
#of all sources defined as ratio (number from <0,1> interval) of nominal power output
def concAllSourcesOnePoint(powerOutputRatios: list[float],
                  xCoor: float, yCoor: float, zCoor: float,
                  sourceParams_all: list[list[float]], 
                  dispersionParams: list[float], 
                  domainParams: list[float], 
                  stabilityClass: str) -> float:

    '''
    sourceParams_all_copy = []
    for source in sourceParams_all:
        sourceParams_all_copy.append(source.copy())
    '''
    pointConcentration_AllSources = 0 #total concentration at point x,y,z due to the imission contribution of all sources running at real power outputs
    
    ##add imission contribution of each source
    for indx in range(len(sourceParams_all)):
        #sourceParams_Copy = (sourceParams_all[indx]).copy()
        sourceParams = sourceParams_all[indx]
        #Change pollutant mass flow at the stack outlet proportianaly to real power output ratio of each source
        #sourceParams[6] = powerOutputRatios[indx]*sourceParams[6]
        powerOutputRatio = powerOutputRatios[indx].copy()
        if(powerOutputRatio != 0):
            pointConcentration_AllSources += conc_OneSource(powerOutputRatio, xCoor, yCoor, zCoor, sourceParams, dispersionParams, domainParams, stabilityClass)
           
    return pointConcentration_AllSources
#===========================================================================================================================================#
#=======================================/CONCENTRATION=IN=ONE=POINT=FROM=ALL=SOURCES========================================================#
#===========================================================================================================================================#


#===========================================================================================================================================#
#=======================================CONCENTRATION=IN=ALL=POINTS=FROM=ALL=SOURCES========================================================#
#===========================================================================================================================================#
#compute sum of yerly concentrations in all points in <points> for all sources and all wind directions, for one stability class,
# based on real power outout of all sources defined as ratio (number from <0,1> interval) of nominal power output
def concAllSourcesAllPoint(powerOutputRatios: list[float],
                  points: list[list[float]],
                  sourceParams_all: list[list[float]], 
                  dispersionParams: list[float], 
                  domainParams: list[float], 
                  stabilityClass: str) -> float:
    '''
    sourceParams_all_copy = []
    for source in sourceParams_all:
        sourceParams_all_copy.append(source.copy())
    '''

    allPointsConcentration_AllSources = 0
    for point in points:
        xCoor, yCoor, zCoor = point
        allPointsConcentration_AllSources += concAllSourcesOnePoint(powerOutputRatios, 
                                                                    xCoor, yCoor, zCoor, 
                                                                    sourceParams_all, 
                                                                    dispersionParams, 
                                                                    domainParams, 
                                                                    stabilityClass)
    
    return allPointsConcentration_AllSources

#===========================================================================================================================================#
#=======================================/CONCENTRATION=IN=ALL=POINTs=FROM=ALL=SOURCES=======================================================#
#===========================================================================================================================================#



#===========================================================================================================================================#
#=======================================================MINIMIZE=FOR=ONE=POINT==============================================================#
#===========================================================================================================================================#
#find real power output combination [q1, q2, q3 .... qn] which minimize concentration at point x,y,z
def minimizeConcAtOnePoint(xCoor: float, yCoor: float, zCoor: float,
                        sourceParams_all: list[list[float]],
                        powerRatiosNominal: list[float], 
                        dispersionParams: list[float], 
                        domainParams: list[float], 
                        stabilityClass: str) -> float:
    '''
    print("\n")
    for source in sourceParams_all:
        print("Source params:", source[0], source[1], source[2], source[3], source[4], source[5], source[6])
    '''

    #define initial guess for qRatios
    #lst = [0.4, 0.6, 0.6, 0.6, 0.6, 0.6]
    lst = [1, 0, 0, 0, 0, 0]
    xinit = np.array(lst)

    #constraint that ensures constant total power output
    sumaryPowerOutput = lambda x: np.dot(x, powerRatiosNominal)
    qRatiosConstraint = spopt.NonlinearConstraint(sumaryPowerOutput, 1, 1)

    #minimize output from concAllSourcesOnePoint() for point X[xCoor, yCoor, zCoor] with respect to first parameter - powerOutputRatios
    #(i.e. find combination of real power-outpus of all sources, for which total concentration in point X is minimal)
    #few important facts: in case of no method defined, solution does not move from xinit,
    #                       only method for which (probably aproximate) solution is found is trust-constr, all other methods does not work at all
    #                       however even this method returns slightly different solution for diferent initial seeds xinit
    res = spopt.minimize(concAllSourcesOnePoint, x0 = xinit, 
                         args = (xCoor, yCoor, zCoor, sourceParams_all, dispersionParams, domainParams, stabilityClass),  
                         bounds = [(0,1), (0,1), (0,1), (0,1), (0,1), (0,1) ], 
                         constraints=qRatiosConstraint,
                         method = "trust-constr")
    
    return res.x
#===========================================================================================================================================#
#=======================================================/MINIMIZE=FOR=ONE=POINT=============================================================#
#===========================================================================================================================================#


#===========================================================================================================================================#
#=======================================================MINIMIZE=FOR=ALL=POINT==============================================================#
#===========================================================================================================================================#
#find real power output combination [q1, q2, q3 .... qn] which minimize concentration at point x,y,z
def minimizeConcAtAllPoints(points: list[list[float]],
                        sourceParams_all: list[list[float]],
                        powerRatiosNominal: list[float], 
                        dispersionParams: list[float], 
                        domainParams: list[float], 
                        stabilityClass: str) -> float:
    '''
    print("\n")
    for source in sourceParams_all:
        print("Source params:", source[0], source[1], source[2], source[3], source[4], source[5], source[6])
    '''

    '''
    sourceParams_all_copy = []
    for source in sourceParams_all:
        sourceParams_all_copy.append(source.copy())
    '''
        
    #define initial guess for qRatios
    #lst = [0.4, 0.6, 0.6, 0.6, 0.6, 0.6]
    #lst = [1, 0, 0, 0, 0, 0]
    lst = [0, 1, 1, 1, 1, 1]
    xinit = np.array(lst)

    #constraint that ensures constant total power output
    sumaryPowerOutput = lambda x: np.dot(x, powerRatiosNominal)
    qRatiosConstraint = spopt.NonlinearConstraint(sumaryPowerOutput, 1, 1)

    #minimizingFunction = lambda x: concAllSourcesAllPoint(x, points, sourceParams_all, dispersionParams, domainParams, stabilityClass)

    #minimize output from concAllSourcesAllPoint() for all points define in <points with respect to first parameter - powerOutputRatios
    #(i.e. find combination of real power-outpus of all sources, for which sum of total concentration in all points is minimal)
    #few important facts: in case of no method defined, solution does not move from xinit,
    #                       only method for which (probably aproximate) solution is found is trust-constr, all other methods does not work at all
    #                       however even this method returns slightly different solution for diferent initial seeds xinit

    '''
    res = spopt.minimize(minimizingFunction, x0 = xinit, 
                         bounds = [(0,1), (0,1), (0,1), (0,1), (0,1), (0,1) ], 
                         constraints=qRatiosConstraint,
                         method = "trust-constr")
    '''

    res = spopt.minimize(concAllSourcesAllPoint, x0 = xinit, 
                         args = (points, sourceParams_all, dispersionParams, domainParams, stabilityClass),  
                         bounds = [(0,1), (0,1), (0,1), (0,1), (0,1), (0,1) ], 
                         constraints=qRatiosConstraint,
                         method = "trust-constr")
    

    '''
    print("\n")
    for source in sourceParams_all:
        print("Source params:", source[0], source[1], source[2], source[3], source[4], source[5], source[6])
    '''
    return res.x
#===========================================================================================================================================#
#=======================================================/MINIMIZE=FOR=ALL=POINT=============================================================#
#===========================================================================================================================================#



#===========================================================================================================================================#
#=================================================DISPERSION=COEFFICIENTS===================================================================#
#===========================================================================================================================================#
def computeDispersionCoeff(xCoor, stabilityClass: str) -> tuple[float, float]:
    #after consideration, I think that having these coefficient in "plain sigth" directly in a code is better for readibility and understanding 
    # of code. However, I moved them into stand-alone function
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
    
    return sigma_z, sigma_y
#===========================================================================================================================================#
#=================================================/DISPERSION=COEFFICIENTS==================================================================#
#===========================================================================================================================================#




#===========================================================================================================================================#
#=========================================!!!!!!CUMULATIVE=IMISSIONS=NOT=USED!!!!===========================================================#
#===========================================================================================================================================#
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
#===========================================================================================================================================#
#===========================================/!!!!!!CUMULATIVE=IMISSIONS=NOT=USED!!!!========================================================#
#===========================================================================================================================================#



#===========================================================================================================================================#
#=========================================!!!!!!MINIMIZE=IMISSIONS=NOT=USED!!!!=============================================================#
#===========================================================================================================================================#
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
    #TO DO try simpler version, imissionCums * x(1, 0.2, 0.2, 0.2, 0.2, 0.2), where x is a simple scalar

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
#===========================================================================================================================================#
#=========================================/!!!!!!MINIMIZE=IMISSIONS=NOT=USED!!!!============================================================#
#===========================================================================================================================================#

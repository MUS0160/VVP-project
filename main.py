import numpy as np
import scipy.optimize as spopt
import matplotlib.pyplot as plt


from lib import getinput
from lib import output
from lib import domain
from lib import ge


#==================================================INPUT==============================================================================#
#For the purpose of this program equivalency between source power-output and source emission rate is assumed. 
#Optimal power output computed with ge.minimizeImissions() is directly used as a coefficient for optimal emission rate of each source.
#As power output and emission are for standard combustion proces directly related it's not an incorrect assumption.
#However, some kind of numerical relation between power output and corresponding emission rate should be formulated 
#(for example using emission factors for respective kind of fuel, fuel calorific value and fuel consuption)

sourceFolder = "./input/v01/"
domainParams, dispersionParams = getinput.getInputData(sourceFolder + "domain.txt", 
                                                       sourceFolder + "dispersion.txt")

#get source characteristics
sourceFilesNames = [sourceFolder +"sourceMain.txt",
                    sourceFolder +"sourceDistributed_01.txt", 
                    sourceFolder +"sourceDistributed_02.txt", 
                    sourceFolder +"sourceDistributed_03.txt",
                    sourceFolder +"sourceDistributed_04.txt",
                    sourceFolder +"sourceDistributed_05.txt"]

sourceParams_all = []
for file in sourceFilesNames:
    sourceParams_all.append(getinput.getSourceData(file))

#get nominal power output of all sources
powerRatiosNominal = getinput.getPowerRatios(sourceFolder + "powerRatiosNominal.txt")

#==================================================/INPUT==============================================================================#

#==================================================PARAMETERS==============================================================================#
#define stability classes 
stabilityClass = ["A", "B", "C", "D", "E", "F"]
#==================================================/PARAMETERS==============================================================================#


if __name__ == "__main__":
    #=================================DEFINE=POINTS=IN=DOMAIN=TO=MINIMIZE=IMISSIONS===================================================#
    #Define all points in domain, for which total imission concentrations must as low as possible (for given total power-output of all sources)
    x1 = [0.2, 0.6, 2]
    x2 = [0.6, 0.7, 2]
    points = [x1, x2]
    #=================================/DEFINE=POINTS=IN=DOMAIN=TO=MINIMIZE=IMISSIONS===================================================#


    #==============================CREATE=MATRIX=MAX=IMISSION=FROM=EACH=SOURCE=IN=EACH=POINT============================================#
    #Compute total imission  in each point of interest as computed concentration in this point from one source running at
    #nominal power output. Compute such total imission  for each point and for each source. 
    #Result is m x n numbers (where m is a number of points of interest and n is a number of sources), i.e. m x n matrix.
    #Represent the overal imission polution in each point from all sources  domain.
    sourceCount = len(sourceParams_all)
    pointsCount = len(points)

    #create matrix for total imission concentrations
    totalImissionConcentrations = np.zeros(shape = (pointsCount,sourceCount))
    #and fill it with computed total concentrations
    for i in range(pointsCount):
        for j in range(sourceCount):
            xCoor = points[i][0]
            yCoor = points[i][1]
            zCoor = points[i][2]
            totalImissionConcentrations[i,j] = ge.conc_OneSource(xCoor, yCoor, zCoor, sourceParams_all[j], dispersionParams, domainParams, "A")

    #==============================/CREATE=MATRIX=MAX=IMISSION=FROM=each=SOURCE=IN=EACH=POINT============================================#


    #==================================OPTIMAL=COMBINATION=OF=POWER=OUTPUTS============================================================#
    #compute optimal combinations of power-outputs of all sources 
    #(optimal combination is the one, for which is sum of all total imission concentrations in all points minimal) 
    #(optimal power output is defined as ratio of nominal power-output of each source, i.e. as number from <0,1 interval>)
    #Key constraint is, that combined optimal power output of all sources must be constant (in order to supply necessary amount of heat)

    #define initial guess for power output ratios
    #dimension xinit must  be changed manually when the number of sources is changed.
    #in automated version, number of elements and their values should be initiated according to number of sources and constraints on x
    lst = [0.4, 0.6, 0.6, 0.6, 0.6, 0.6]
    #lst = [1, 0, 0, 0, 0, 0]
    #lst = [0, 1, 1, 1, 1, 1]
    xinit = np.array(lst)

    #constraint that ensures constant total power output
    sumaryPowerOutput = lambda x: np.dot(x, powerRatiosNominal)
    powerRatiosConstraint = spopt.NonlinearConstraint(sumaryPowerOutput, 1, 1)

    #minimizing function
    minimizingFunction = lambda x: (np.linalg.norm((totalImissionConcentrations@x), ord=1))

    #find optimal combination
    res = spopt.minimize(minimizingFunction, x0 = xinit, 
                         bounds = [(0,1), (0,1), (0,1), (0,1), (0,1), (0,1) ], 
                         constraints=powerRatiosConstraint)

    optimalPowerOutputs = res.x

    np. set_printoptions(precision=3)
    print("Optimal combination of power outputs for all sources: ", optimalPowerOutputs)
    #==================================/OPTIMAL=COMBINATION=OF=POWER=OUTPUTS============================================================#


    #===============================CREATE=CVS's=AND=GRAPH=FOR=OPTIMAL=POWER=OUTPUT=========================================================#
    #adjust source power output in each source to optimal values
    sourceParams_optimal_all = []
    for source in range(0,len(optimalPowerOutputs)):
        sourceParams_optimal_all.append(sourceParams_all[source])
        sourceParams_optimal_all[source][6] = optimalPowerOutputs[source]*sourceParams_all[source][6]

    #and compute total imission concentration with all sources running at this power output (example with stability class A)
    totalConcField_Optimal = ge.gaussDispEq_TotalConcField(sourceParams_optimal_all[0], dispersionParams, domainParams, "A")
    for source in range(1, len(sourceParams_optimal_all)):
        totalConcField_Optimal += ge.gaussDispEq_TotalConcField(sourceParams_optimal_all[source], dispersionParams, domainParams, "A")
    
    np.savetxt("./output/imissionConc_optimal.csv", totalConcField_Optimal, delimiter=",")

    #create graph with optimal imission concentration and save it
    fileName = 'plot_optimal'
    title = 'Concentrations for optimal power combination'
    output.createGraphs(totalConcField_Optimal, fileName, title, domainParams)
    #===============================/CREATE=CVS's=AND=GRAPH=FOR=OPTIMAL=POWER=OUTPUT=========================================================#



    #===============================CREATE=CVS's=AND=GRAPH=FOR=JUST=MAIN=SOURCE=IN=OPERATION================================================#
    #For comparison print imission concentration for just central heat source in full operation
    totalConcField_MainSource, totalConcField_SmallSources = ge.totalConcFields_MainSmall(sourceParams_all, dispersionParams, domainParams, "A" )
    np.savetxt("./output/imissionConc_main.csv", totalConcField_MainSource, delimiter=",")
    np.savetxt("./output/imissionConc_distributed.csv", totalConcField_SmallSources, delimiter=",")
    #and create graph and save it
    fileName = 'plot_main'
    title = 'Concentrations for main source in operation'
    output.createGraphs(totalConcField_MainSource, fileName, title, domainParams)
    #===============================/CREATE=CVS's=AND=GRAPH=FOR=JUST=MAIN=SOURCE=IN=OPERATION================================================#


    #===============================CREATE=CVS's=AND=GRAPH=FOR=JUST=DISTRIBUTED=SOURCES=IN=OPERATION==========================================#
    #and for distributed heat sources in full opeation (without central source) 
    fileName = 'plot_distributed'
    title = 'Concentrations for distributed sources in operation'
    output.createGraphs(totalConcField_SmallSources, fileName, title, domainParams)
    #===============================/CREATE=CVS's=AND=GRAPH=FOR=JUST=DISTRIBUTED=SOURCES=IN=OPERATION==========================================#
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

sourceFilesNames = [sourceFolder +"sourceMain.txt",
                    sourceFolder +"sourceDistributed_01.txt", 
                    sourceFolder +"sourceDistributed_02.txt", 
                    sourceFolder +"sourceDistributed_03.txt",
                    sourceFolder +"sourceDistributed_04.txt",
                    sourceFolder +"sourceDistributed_05.txt"]

sourceParams_all = []
for file in sourceFilesNames:
    sourceParams_all.append(getinput.getSourceData(file))

powerRatiosNominal = getinput.getPowerRatios(sourceFolder + "powerRatiosNominal.txt")

#==================================================/INPUT==============================================================================#

#==================================================PARAMETERS==============================================================================#
stabilityClass = ["A", "B", "C", "D", "E", "F"]
#==================================================/PARAMETERS==============================================================================#


if __name__ == "__main__":
    #=================================DEFINE=POINTS=IN=DOMAIN=TO=MINIMIZE=IMISSIONS===================================================#
    #Define all points in domain, for which total imission concentrations must as low as possible (for given total power-output of all sources)
    x1 = [0.5, 0.35, 2]
    x2 = [0.8, 0.4, 2]
    points = [x1, x2]
    #=================================/DEFINE=POINTS=IN=DOMAIN=TO=MINIMIZE=IMISSIONS===================================================#



    #==================================OPTIMAL=COMBINATION=OF=POWER=OUTPUTS============================================================#
    #compute optimal combinations of power-outputs of all sources 
    #(defined as ratio of nominal power-output of each source, i.e. as number from <0,1 interval>)
    #for which sum of all imission concetration in all points of interest is minimal.
    #Function minimizeConcAtAllPoints() identify combination of power-outputs of all sources, for which the level of imission concentrations 
    #in all points of interest is as low as possible.

    optimalPowerOutputs = ge.minimizeConcAtAllPoints(points, sourceParams_all, powerRatiosNominal, dispersionParams, domainParams, "A") 


    #c = ge.concAllSourcesAllPoint(optimalPowerOutputs, points, sourceParams_all, dispersionParams, domainParams, "A")
    #print(np.dot(optimalPowerOutputs,powerRatiosNominal))
    
    print("Optimal combination of power outputs for all sources: ", optimalPowerOutputs)
    #print(c)
    #==================================/OPTIMAL=COMBINATION=OF=POWER=OUTPUTS============================================================#
    # compute total imission concentration with all sources running at this power output (example with stability class A)
    #DEBUG
    
    for source in sourceParams_all:
        print("Source params:", source[0], source[1], source[2], source[3], source[4], source[5], source[6])
    
    #TO DO, change source params to optimal power output
    totalConcField_Optimal = ge.gaussDispEq_TotalConcField(sourceParams_all[0], dispersionParams, domainParams, "A")
    for source in range(1, len(sourceParams_all)):
        totalConcField_Optimal += ge.gaussDispEq_TotalConcField(sourceParams_all[source], dispersionParams, domainParams, "A")
    np.savetxt("./output/imissionConc_optimal.csv", totalConcField_Optimal, delimiter=",")
    
    #create graph with optimal imission concentration and save it
    fileName = 'plot_optimal'
    title = 'Concentrations for optimal power combination'
    output.createGraphs(totalConcField_Optimal, fileName, title, domainParams)

    #=================================SHOW=CUMULATIVE=IMISSION=FOR=OPTIMAL=COMBINATION======================================================#

    #=================================/SHOW=CUMULATIVE=IMISSION=FOR=OPTIMAL=COMBINATION======================================================#


    #======================SHOW=CUMULATIVE=IMISSION=FOR=MAIN=SOURCE=ONLY=AND=FOR=DISTRIBUTED=SOURCES=ONLY=================================#
    #Create imission concentration in whole domain for just central heat source in full operation
    '''
    sourceParams_all = []
    for file in sourceFilesNames:
        sourceParams_all.append(getinput.getSourceData(file))
    
    #DEBUG
    for source in sourceParams_all:
        print("Source params:", source[0], source[1], source[2], source[3], source[4], source[5], source[6])
    '''
    totalConcField_MainSource, totalConcField_SmallSources = ge.totalConcFields_MainSmall(sourceParams_all, dispersionParams, domainParams, "A" )
    np.savetxt("./output/imissionConc_main.csv", totalConcField_MainSource, delimiter=",")
    np.savetxt("./output/imissionConc_distributed.csv", totalConcField_SmallSources, delimiter=",")
    #and create graph and save it
    fileName = 'plot_main'
    title = 'Concentrations for main source in operation'
    output.createGraphs(totalConcField_MainSource, fileName, title, domainParams)

    #and for distributed heat sources in full opeation (without central source) 
    fileName = 'plot_distributed'
    title = 'Concentrations for distributed sources in operation'
    output.createGraphs(totalConcField_SmallSources, fileName, title, domainParams)


    #======================/SHOW=CUMULATIVE=IMISSION=FOR=MAIN=SOURCE=ONLY=AND=FOR=DISTRIBUTED=SOURCES=ONLY================================#




    #=================================SHOW=CUMULATIVE=IMISSION=FOR=DISTRIBUTED=SOURCES=ONLY======================================================#

    #=================================/SHOW=UMULATIVE=IMISSION=FOR=DISTRIBUTED=SOURCES=ONLY======================================================#






'''
if __name__ == "__main__":
    #=================================CUMULATIVE=IMISSION=PER=SOURCE=AND=STABILITY=CLASS===================================================#
    #Compute cumulative imission concentration (through whole domain) for each stability class and for each source (at nominal power) 
    #Cumulative imission means summ of all computed concentrationf for each point in the domain. 
    #Represent the overal imission polution of computed domain.
    # Store each cumulative value in matrix 
    #(each matrix element represent total imission polution in the domain for one source and one stability class)
    sourceCount = len(sourceParams_all)
    stabClassCount = len(stabilityClass)
    cumulativeImission = np.zeros(shape = (stabClassCount,sourceCount))
    for i in range(stabClassCount):
        for j in range(sourceCount):
            cumulativeImission[i,j] = ge.computeCumulativeImission(sourceParams_all[j], dispersionParams, domainParams, stabilityClass[i])
        #=================================/CUMULATIVE=IMISSION=PER=SOURCE=AND=STABILITY=CLASS===================================================#

    
    #=================================MINIMIZE=CUMULATIVE=IMISSIONS=OF=ALL=SOURCES==========================================================#
    #compute power output of each source, for which the combined cumulative imissions of all sources for each classes are minimal
    sourcePowerOutputs = ge.minimizeImissions(cumulativeImission)
    np.set_printoptions(precision=3)
    print("\n\nOptimal power output for main source: ", sourcePowerOutputs[0])
    for source in range(1,len(sourcePowerOutputs)):
        print("Optimal power output for distributed source no.", source, ": ",  sourcePowerOutputs[source])
    #=================================/MINIMIZE=CUMULATIVE=IMISSIONS=OF=ALL=SOURCES==========================================================#

    
    #===============================CREATE=CVS's=AND=GRAPH=FOR=OPTIMAL=POWER=OUTPUT=========================================================#
    #adjust source power output to optimal values
    sourceParams_optimal_all = []
    for source in range(0,len(sourcePowerOutputs)):
        sourceParams_optimal_all.append(sourceParams_all[source])
        sourceParams_optimal_all[source][6] = sourcePowerOutputs[source]*sourceParams_all[source][6]

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
'''


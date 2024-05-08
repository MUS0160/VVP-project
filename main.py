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

#TO DO: sources are imported one by one, should be automated (automatic identification of number of source files, without need to explicitly import them)
#CHANGE OF SOURCE COUNT: delete or add new source files
sourceFolder = "./input/v01/"
domainParams, dispersionParams = getinput.getInputData(sourceFolder + "domain.txt", sourceFolder + "dispersion.txt")
sourceParams_main = getinput.getSourceData(sourceFolder +"sourceMain.txt")
sourceParams_01 = getinput.getSourceData(sourceFolder +"sourceDistributed_01.txt")
sourceParams_02 = getinput.getSourceData(sourceFolder +"sourceDistributed_02.txt")
sourceParams_03 = getinput.getSourceData(sourceFolder +"sourceDistributed_03.txt")
sourceParams_04 = getinput.getSourceData(sourceFolder +"sourceDistributed_04.txt")
sourceParams_05 = getinput.getSourceData(sourceFolder +"sourceDistributed_05.txt")
#CHANGE OF SOURCE COUNT: delete or add sourceParams elements
sourceParams_all = [sourceParams_main, sourceParams_01, sourceParams_02, sourceParams_03, sourceParams_04, sourceParams_05]
#==================================================/INPUT==============================================================================#

#==================================================PARAMETERS==============================================================================#
stabilityClass = ["A", "B", "C", "D", "E", "F"]
#imissionLimit = 40/1000000 # microgram.m-3 to g.m-3
#==================================================/PARAMETERS==============================================================================#


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



   
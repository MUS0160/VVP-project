

def getInputData(domainFile: str, dispersionFile: str) -> tuple[list[float], list[float]]:
    #get domainParams
    #Get domain characteristis from input file.
    #Input file name: domain.txt
    #Structure of domain.txt
    # txt file with following lines (each line consist of single number):
    #x dimension in km
    #y dimension in km (for simplicity and easy coordinate transformation only square domain is considered)
    #number of nodes in one dimension (resolution of the domain)
    domainParams = []
    f = open(domainFile, "r")

    #get x dimension of a domain
    xdim = f.readline() 
    domainParams.append(int(xdim))
    #get y dimension of a domain
    ydim = f.readline()
    domainParams.append(int(ydim))
    #get resolution of a domain
    resolution = f.readline()
    domainParams.append(int(resolution))
    f.close()

    #get dispersionParams
    #Input file name: dispersion.txt
    #txt file with following lines (each line consist of single number):
    #atmospheric temperature [°K] (format: double)
    #average wind velocity [m/s] (format:double)
    #wind rose on next 8 lines- percentage share of wind directions per year (format:double)
    dispersionParams = []
    f = open(dispersionFile, "r")
    #get atmospheric temperature
    atmTemp = f.readline()
    dispersionParams.append(float(atmTemp))
    #get average wind velocity
    windVel = f.readline()
    dispersionParams.append(float(windVel))
    #get percentage share of wind from N direction
    N_wind = f.readline()
    dispersionParams.append(float(N_wind))
    #get percentage share of wind from NE direction
    NE_wind = f.readline()
    dispersionParams.append(float(NE_wind))
    #get percentage share of wind from E direction
    E_wind = f.readline()
    dispersionParams.append(float(E_wind))
    #get percentage share of wind from SE direction
    SE_wind = f.readline()
    dispersionParams.append(float(SE_wind))
    #get percentage share of wind from S direction
    S_wind = f.readline()
    dispersionParams.append(float(S_wind))
    #get percentage share of wind from SW direction
    SW_wind = f.readline()
    dispersionParams.append(float(SW_wind))
    #get percentage share of wind from W direction
    W_wind = f.readline()
    dispersionParams.append(float(W_wind))
    #get percentage share of wind from NW direction
    NW_wind = f.readline()
    dispersionParams.append(float(NW_wind))

    f.close()
    
    return domainParams, dispersionParams


def getSourceData(sourceFile: str) -> list[float]:
    #Get source parameters
    #Input file name: sourceMain.txt, sourceDistributed_XX.txt
    #txt file with following lines (each line consist of single number):
    #source_xCoor: relative x coordinate of source in domain (number from <0,1> interval)
    #source_yCoor: relative y coordinate of source in domain (number from <0,1> interval)
    #source stack height [m] (format: double)
    #source stack internal diameter (outlet) [m] (format: double)
    #source stack flue gas velocity at the outlet [m/s] (format: double)
    #source stack flue gas temperature at the outlet [°K] (format: double)
    #source emission rate [g/s] (format:double)
    sourceParams = []
    f = open(sourceFile, "r")
    
    #get x coordinate of the source in the domain (relative coordinate, number from <0,1> interval)
    source_xCoor = f.readline()
    sourceParams.append(float(source_xCoor))
    #get y coordinate of the source in the domain (relative coordinate, number from <0,1> interval)
    source_yCoor = f.readline()
    sourceParams.append(float(source_yCoor))
    #get source stack height above the terrain [in m]
    source_height = f.readline()
    sourceParams.append(float(source_height))
    #get source stack internal diameter at the outlet [in m]
    source_intDiam = f.readline()
    sourceParams.append(float(source_intDiam))
    #get flue gas output velocity [in m/s]
    source_flueGasVel = f.readline()
    sourceParams.append(float(source_flueGasVel))
    #get flue gas temperature at the outlet [in K]
    source_flueGasTemp = f.readline()
    sourceParams.append(float(source_flueGasTemp))
    #get emission rate from the source for computed pollutant [in g/s]
    source_emissionRate = f.readline()
    sourceParams.append(float(source_emissionRate))

    f.close()

    return sourceParams
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

def createGraphs(concField: NDArray[np.float64] , fileName: str, title: str, domainParams: list[float]) -> None:
    #for concentration field in computed domain create scalar and contour graph, 
    #save them under the specified names
    #print  imission concentrations
    plt.imshow(concField, cmap='viridis', origin='lower', aspect='auto')
    plt.title(title)
    fileName1 = './output/' + fileName + '_scalar.png'
    #fileName1 = fileName + '_scalar.png'
    plt.savefig(fileName1)
    #plt.show()
    plt.clf()
    #print  imission concentrations - contour version
    x = np.linspace(0, domainParams[0], domainParams[2])
    y = np.linspace(0, domainParams[1], domainParams[2])
    X, Y = np.meshgrid(x, y)
    cnt = plt.contour(X, Y, concField, 10, cmap="jet")
    plt.colorbar()
    plt.title(title)
    fileName2 = './output/' + fileName + '_contour.png'
    #fileName2 = fileName + '_contour.png'
    plt.savefig(fileName2)
    #plt.show()
    plt.clf()


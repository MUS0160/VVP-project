import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


def create_graphs(concField: NDArray[np.float64], file_name: str, title: str, domain_params: list[float]) -> None:
    """
    For concentration field in computed domain create scalar and contour graph,
    save them under the specified names and save imission concentrations.
    If file_name == "print" graphs are printed instead of saved

    Args: 
        concField:      ND array of float values, concentrations of pollutant in the domain
        file_name:      file name for the output graph to save
        domain_params:  "space" charakcteristics of the domain (dimensions in each axis)

    Returns: 
        image of imission concentratios saved in ./output/file_name_scalar.png
        or printed in case of file_name == "print"
    """
    # print  imission concentrations - scalar version
    plt.imshow(concField, cmap='viridis', origin='lower', aspect='auto')
    plt.title(title)
    if (file_name == "print"):
        plt.show()
    else:
        file_name1 = './output/' + file_name + '_scalar.png'
        plt.savefig(file_name1)
    plt.clf()
    # print  imission concentrations - contour version
    x = np.linspace(0, domain_params[0], domain_params[2])
    y = np.linspace(0, domain_params[1], domain_params[2])
    X, Y = np.meshgrid(x, y)
    cnt = plt.contour(X, Y, concField, 10, cmap="jet")
    plt.colorbar()
    plt.title(title)
    if (file_name == "print"):
        plt.show()
    else:
        file_name2 = './output/' + file_name + '_contour.png'
        plt.savefig(file_name2)
    plt.clf()

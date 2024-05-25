import numpy as np
import scipy.optimize as spopt

from lib import getinput
from lib import output
from lib import ge


# ==================================================INPUT==============================================================================#
"""
For the purpose of this program equivalency between source power-output and
source emission rate is assumed.  Optimal power output computed with
ge.minimizeImissions() is directly used as a coefficient for optimal emission
rate of each source.
As power output and emission are for standard combustion proces directly
related it's not an incorrect assumption. However, some kind of numerical
relation between power output and corresponding emission rate should be
formulated (for example using emission factors for respective kind of fuel,
fuel calorific value and fuel consuption)
"""

# Get all necessary input data
domain_params, dispersion_params, source_params_all, power_ratios_nominal = getinput.get_input(
    "./input/v01/", 5)
# ==================================================/INPUT==============================================================================#

# ==================================================PARAMETERS==============================================================================#
# define stability classes
stability_class = ["A", "B", "C", "D", "E", "F"]
# ==================================================/PARAMETERS==============================================================================#


if __name__ == "__main__":
    # =================================DEFINE=POINTS=IN=DOMAIN=TO=MINIMIZE=IMISSIONS===================================================#
    # Define all points in domain, for which total imission concentrations must as low as possible (for given total power-output of all sources)
    x1 = [0.2, 0.6, 2]
    x2 = [0.6, 0.7, 2]
    points = [x1, x2]
    # =================================/DEFINE=POINTS=IN=DOMAIN=TO=MINIMIZE=IMISSIONS===================================================#

    # ==============================CREATE=MATRIX=MAX=IMISSION=FROM=EACH=SOURCE=IN=EACH=POINT============================================#
    """
    Compute total imission  in each point of interest as computed
    concentration in this point from one source running at nominal power
    output. Compute such total imission  for each point and for each source.
    Result is m x n numbers (where m is a number of points of interest and n
    is a number of sources), i.e. m x n matrix.     Represent the overal
    imission polution in each point from all sources domain.
    """
    source_count = len(source_params_all)
    points_count = len(points)

    # create matrix for total imission concentrations
    tot_imission_conc = np.zeros(shape=(points_count, source_count))
    # and fill it with computed total concentrations
    for i in range(points_count):
        for j in range(source_count):
            x_coor = points[i][0]
            y_coor = points[i][1]
            z_coor = points[i][2]
            tot_imission_conc[i, j] = ge.conc_one_source(
                x_coor, y_coor, z_coor, source_params_all[j], dispersion_params, domain_params, "A")

    # ==============================/CREATE=MATRIX=MAX=IMISSION=FROM=each=SOURCE=IN=EACH=POINT============================================#

    # ==================================OPTIMAL=COMBINATION=OF=POWER=OUTPUTS============================================================#
    """
    Compute optimal combinations of power-outputs of all sources
    (optimal combination is the one, for which is sum of all total imission
    concentrations in all points minimal)
    (optimal power output is defined as ratio of nominal power-output of each
    source, i.e. as number from <0,1 interval>)
    Key constraint is, that combined optimal power output of all sources must
    be constant (in order to supply necessary amount of heat)
    """

    """
    define initial guess for power output ratios
    dimension xinit must  be changed manually when the number of sources is
    changed. In automated version, number of elements and their values should
    be initiated according to number of sources and constraints on x
    """
    lst = [0.4, 0.6, 0.6, 0.6, 0.6, 0.6]
    # lst = [1, 0, 0, 0, 0, 0]
    # lst = [0, 1, 1, 1, 1, 1]
    xinit = np.array(lst)

    # constraint that ensures constant total power output
    def sum_power_out(x): return np.dot(x, power_ratios_nominal)
    pow_ratio_constraint = spopt.NonlinearConstraint(sum_power_out, 1, 1)

    # minimizing function
    def minimizing_function(x): return (
        np.linalg.norm((tot_imission_conc@x), ord=1))

    # find optimal combination
    res = spopt.minimize(minimizing_function, x0=xinit,
                         bounds=[(0, 1), (0, 1), (0, 1),
                                 (0, 1), (0, 1), (0, 1)],
                         constraints=pow_ratio_constraint)

    opt_power_out = res.x

    np. set_printoptions(precision=3)
    print("Optimal combination of power outputs for all sources: ", opt_power_out)
    # ==================================/OPTIMAL=COMBINATION=OF=POWER=OUTPUTS============================================================#

    # ===============================CREATE=CVS's=AND=GRAPH=FOR=OPTIMAL=POWER=OUTPUT=========================================================#
    # adjust source power output in each source to optimal values
    source_params_opt_all = []
    for source in range(0, len(opt_power_out)):
        source_params_opt_all.append(source_params_all[source])
        source_params_opt_all[source][6] = opt_power_out[source] * \
            source_params_all[source][6]

    # and compute total imission concentration with all sources running at this power output (example with stability class A)
    tot_conc_field_opt = ge.gauss_disp_eq_total_conc_field(
        source_params_opt_all[0], dispersion_params, domain_params, "A")
    for source in range(1, len(source_params_opt_all)):
        tot_conc_field_opt += ge.gauss_disp_eq_total_conc_field(
            source_params_opt_all[source], dispersion_params, domain_params, "A")

    np.savetxt("./output/imissionConc_optimal.csv",
               tot_conc_field_opt, delimiter=",")

    # create graph with optimal imission concentration and save it
    file_name = 'plot_optimal'
    title = 'Concentrations for optimal power combination'
    output.create_graphs(tot_conc_field_opt, file_name, title, domain_params)
    # ===============================/CREATE=CVS's=AND=GRAPH=FOR=OPTIMAL=POWER=OUTPUT=========================================================#

    # ===============================CREATE=CVS's=AND=GRAPH=FOR=JUST=MAIN=SOURCE=IN=OPERATION================================================#
    # For comparison print imission concentration for just central heat source in full operation
    tot_conc_field_main, tot_conc_field_distrib = ge.total_conc_fields_main_distr(
        source_params_all, dispersion_params, domain_params, "A")
    np.savetxt("./output/imissionConc_main.csv",
               tot_conc_field_main, delimiter=",")
    np.savetxt("./output/imissionConc_distributed.csv",
               tot_conc_field_distrib, delimiter=",")
    # and create graph and save it
    file_name = 'plot_main'
    title = 'Concentrations for main source in operation'
    output.create_graphs(tot_conc_field_main, file_name, title, domain_params)
    # ===============================/CREATE=CVS's=AND=GRAPH=FOR=JUST=MAIN=SOURCE=IN=OPERATION================================================#

    # ===============================CREATE=CVS's=AND=GRAPH=FOR=JUST=DISTRIBUTED=SOURCES=IN=OPERATION==========================================#
    # and for distributed heat sources in full opeation (without central source)
    file_name = 'plot_distributed'
    title = 'Concentrations for distributed sources in operation'
    output.create_graphs(tot_conc_field_distrib,
                         file_name, title, domain_params)
    # ===============================/CREATE=CVS's=AND=GRAPH=FOR=JUST=DISTRIBUTED=SOURCES=IN=OPERATION==========================================#

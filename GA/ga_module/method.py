import numpy as np
import GA.ga_module.igrow_geatpy as igrow_geatpy
from GA.ga_module.custom import TenSimSearch_v0


def blackbox_opt_mask(opt, f, x_inits, x_fixed, x_mask, params):
    # x_inits are a set of initialization of input x
    # params are all the parameters of opt
    # opt is some opimization method like geatpy2_maximize
    # x_mask an numpy array with 1 means corresponding value can be changed
    # x_fixed defines the values that cannot be changed. len(x_mask) == len(x_fised)
    # return the best top n (ordered by f value, descending) values and corresponding x
    # such as [1.5, 1.46, 1.3], [x1, x2, x3] meaning f(x1) = 1.5 and so on
    idx_fixed = (1-x_mask).nonzero()[0]
    idx_var = x_mask.nonzero()[0]
    n = len(x_inits[0])

    x_inits_mask = [xx[idx_var] for xx in x_inits]

    x_full = np.zeros((n, ))
    x_full[idx_fixed] = x_fixed[idx_fixed]
    x_full[idx_var] = x_inits_mask[0]

    x_bests_mask, v_bests = opt(
        f, x_inits, idx_var,  params)

    x_bests = []
    for xx in x_bests_mask:
        x_full = np.zeros((n, ))
        x_full[idx_var] = xx
        x_full[idx_fixed] = x_fixed[idx_fixed]

        x_bests.append(x_full)
    x_bests = np.array(x_bests)

    return x_bests, v_bests


def geatpy2_maximize_global_psy_v0(f, x_inits, idx_var, params):
    # def geatpy2_maximize_day_v0(x_inits, params):
    # x_inits are a set of initialization of input x
    # params are all the parameters, in a dictionary
    # return the best top n (ordered by f value, descending) values and corresponding x
    # such as [1.5, 1.46, 1.3], [x1, x2, x3] meaning f(x1) = 1.5 and so on
    # this function should be as simple as possible and everything should be put in dependencies

    problem = TenSimSearch_v0(f, idx_var, params)

    population = igrow_geatpy.PsyPopulation(
        problem.Encodings, problem.Fields, params.NIND)

    myAlgorithm = igrow_geatpy.soea_psy_SEGA_templet(
        problem, population, params.XOVR)

    population = myAlgorithm.run(individual=x_inits)

    bestIdx = np.argmax(population.ObjV)
    bestIdiv = population.Phen[bestIdx]

    return bestIdiv[np.newaxis, :], population.FitnV[bestIdx]

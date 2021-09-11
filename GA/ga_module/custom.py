import argparse
import numpy as np
import time
import geatpy as ea
import multiprocessing as mp
from tqdm import tqdm

from GA.ga_module.igrow_geatpy import Problem


def init_params(params, settings, bound,
                vars_dict,
                day_dims,
                plant_periods,
                env, save_dir):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--seed', default=params['seed'], help='random seed')
    parser.add_argument(
        '--NIND', default=params['NIND'], help='population size')
    parser.add_argument(
        '--MAXGEN', default=params['MAXGEN'], help='maximum generation')
    parser.add_argument(
        '--LINKDAY', default=params['LINKDAY'], help='copy days')
    parser.add_argument(
        '--XOVR', default=params['XOVR'], help='crossover probability')
    parser.add_argument(
        '--parallel', default=settings['parallel'], help='parallel calculate')
    parser.add_argument('--env', default=env,
                        help='Ten-sim')
    parser.add_argument('--vars_dict', default=vars_dict,
                        help='control variables name and dims')
    parser.add_argument('--bound', default=bound,
                        help='bound value of variables')
    parser.add_argument('--day_dims', default=day_dims,
                        type=int, help='one day search dims')
    parser.add_argument('--plant_periods', default=plant_periods,
                        type=int, help='plant periods')
    parser.add_argument('--X_best_sofar_path', default=save_dir+'/ga_train/policy/',
                        help='TenSim version')
    parser.add_argument('--log_path', default=save_dir+'/ga_train/log/',
                        help='TenSim version')
    parser.add_argument('--save_interval', default=100,
                        help='save_interval')
    args = parser.parse_args()
    args.seed = int(args.seed)
    args.NIND = int(args.NIND)
    args.MAXGEN = int(args.MAXGEN)
    args.LINKDAY = int(args.LINKDAY)
    if float(args.XOVR) >= 1:
        args.XOVR = round(float(args.XOVR) / 10, 1)
    else:
        args.XOVR = float(args.XOVR)
    return args


class TenSimSearch_v0(Problem):
    def __init__(self, f, idx_var, args):
        self.idx_var = idx_var
        self.f = f

        self.parallel = args.parallel

        self.bound = args.bound

        self.vars_dict = args.vars_dict
        self.MAXGEN = args.MAXGEN
        self.NIND = args.NIND
        self.day_dims = args.day_dims
        self.plant_periods = args.plant_periods
        self.X_best_sofar_path = args.X_best_sofar_path
        self.save_interval = args.save_interval

        self.sim_count = 0

        name = 'MyProblem'
        M = 1
        maxormins = [-1]
        Dim = len(idx_var)

        varTypes = [1] * Dim
        lb_list, ub_list = self.bound
        lb_np, ub_np = np.array(lb_list), np.array(ub_list)
        lb, ub = list(lb_np[idx_var]), list(ub_np[idx_var])
        assert len(lb) == Dim, len(ub) == Dim
        lbin = [1] * Dim
        ubin = [1] * Dim

        Problem.__init__(self, name, M, maxormins,
                         Dim, varTypes,
                         lb, ub, lbin, ubin)

        ranges = np.array([lb, ub])
        borders = np.array([lbin, ubin])

        Encoding = 'BG'
        self.Fields = []
        idx = 0
        for _, sets in self.vars_dict.items():
            dim = self.plant_periods * sets[0]
            self.Fields.append(ea.crtfld(Encoding=Encoding,
                                         varTypes=np.array(
                                             varTypes[idx: idx+dim]),
                                         ranges=ranges[:, idx: idx+dim],
                                         borders=borders[:, idx: idx+dim]))
            idx += dim
        self.Encodings = [Encoding] * len(self.vars_dict.keys())

    def aimFunc(self, pop):
        start = time.time()
        policy_Mat = pop.Phen.copy()  # get matrix of decision variables

        if self.parallel == "true":

            cores = mp.cpu_count()
            pool = mp.Pool(processes=cores)
            pool = mp.Pool(processes=self.NIND)

            pool_list = []
            for NIND_i in range(self.NIND):
                pool_list.append(
                    pool.apply_async(
                        self.eval_policy, (policy_Mat[NIND_i, :],)))
            pool.close()
            pool.join()
            x_y = [xx.get() for xx in pool_list]

        else:
            x_y = []
            pool_list = []
            bar = tqdm(range(self.NIND))
            bar.set_description(" GEN: %d" % (self.sim_count))
            for NIND_i in bar:
                res = self.eval_policy(policy_Mat[NIND_i, :])
                x_y.append(res)

        profit = []
        for NIND_i in range(self.NIND):
            profit.append([x_y[NIND_i][1]])
            print('profit: ', profit[NIND_i])

        pop.ObjV = np.array(profit)  # update objective

        self.sim_count += 1
        print("iters %d, use time: %.2f sec" % (self.sim_count,
                                                time.time() - start))

    def eval_policy(self, x_i):
        x_i = self.recover_var(x_i)  # recover var
        action_num = len(self.vars_dict.keys())
        x = np.zeros_like(x_i)
        x_i = x_i.reshape(action_num, (len(x_i)//action_num), order='C')

        Idx = 0
        for d in range(self.plant_periods):
            var_i = 0
            for _, sets in self.vars_dict.items():
                x[Idx:Idx+sets[2]] = x_i[var_i, d*sets[2]: (d+1)*sets[2]]

                Idx += sets[2]
                var_i += 1

        # st = time.time()
        y = self.f(x)  # eval
        # print("sim time:%.2f" % (time.time()-st))
        # y = 1

        return (x, y)

    def print_policy(self, x):
        startIdx = 0
        for var, attr in self.vars_dict.items():
            print(var)
            endIdx = startIdx + attr[2]
            print(list(x[startIdx:endIdx]))
            startIdx = endIdx

    def recover_var(self, x):
        startIdx = 0
        var_i = 0
        for _, attr in self.vars_dict.items():
            endIdx = self.Fields[var_i].shape[1] + startIdx
            index = self.idx_var[np.where(
                (startIdx <= self.idx_var) & (self.idx_var < endIdx))]
            x[index] *= attr[1]

            startIdx = endIdx
            var_i += 1

        return x

    def genearte_policy_array(self, x_subsection):
        """
        :param x_subsection:
        :return:
        """
        def var_extend(section, seg_num, day_dim):
            var = []
            for i in range(seg_num):
                var += [section[i]] * (day_dim//seg_num)

            return var

        day_x = []
        idx = 0
        for _, length in self.vars_dict.items():
            day_x += var_extend(section=x_subsection[idx:idx+length[0]],
                                seg_num=length[0], day_dim=length[2])
            idx += length[0]
        return np.array(day_x)

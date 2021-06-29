import datetime
import argparse
import os
import pandas as pd
import numpy as np
import warnings
from scipy import stats

from utils.common import mkdir

warnings.filterwarnings("ignore")
os.environ['NLS_LANG'] = 'AMERICAN_AMERICA.AL32UTF8'


def CalculateCost(Table4_df, harvest_file_dir):
    harvest_price_dir = os.path.join(harvest_file_dir, 'overall_cost.xlsx')

    # energy
    energy_df = pd.read_excel(harvest_price_dir, sheet_name='Energy')
    ctrl_energy = energy_df.values[-1, :2]
    expr_energy = energy_df.values[-1, 2:]
    record = get_record(ctrl_energy, expr_energy, col='Energy Cost', minus='-')
    Table4_df = add_to_table(Table4_df, record, 0)

    # labour
    labour_df = pd.read_excel(harvest_price_dir, sheet_name='Labour')
    ctrl_labour = labour_df.values[-1, :2]
    expr_labour = labour_df.values[-1, 2:]
    record = get_record(ctrl_labour, expr_labour, col='Crop Maintenance Cost')
    Table4_df = add_to_table(Table4_df, record, 1)

    # fixed
    fixed_df = pd.read_excel(harvest_price_dir, sheet_name='Fixed')
    ctrl_fiexd = fixed_df.values[-1, :2]
    expr_fiexd = fixed_df.values[-1, 2:]
    record = get_record(ctrl_fiexd, expr_fiexd, col='Equipment Emortization')
    Table4_df = add_to_table(Table4_df, record, 2)

    # total cost
    ctrl_cost = ctrl_energy+ctrl_labour+ctrl_fiexd
    expr_cost = expr_energy+expr_labour+expr_fiexd
    record = get_record(ctrl_cost, expr_cost, col='Total Cost')
    Table4_df = add_to_table(Table4_df, record, 3)

    return ctrl_cost, expr_cost


def CalculateHarvest(Table4_df, harvest_file_dir):
    # Price
    harvest_price_dir = os.path.join(harvest_file_dir, 'price.csv')
    df = pd.read_csv(harvest_price_dir)
    ctrl_price = df.values[:, 1:3]
    expr_price = df.values[:, 3:]
    expr_price = expr_price.astype(np.float32) * args.rmb2euro
    ctrl_price = ctrl_price.astype(np.float32) * args.rmb2euro

    expr_price[expr_price == 0] = np.nan
    ctrl_price[ctrl_price == 0] = np.nan
    ctrl_avg = np.nanmean(ctrl_price, axis=0)
    expr_avg = np.nanmean(expr_price, axis=0)
    record = get_record(ctrl_avg, expr_avg, col='Price')
    Table4_df = add_to_table(Table4_df, record, 4)

    expr_harvest, ctrl_harvest = get_harvest(args)
    # Production
    ctrl_prod = ctrl_harvest['production'][-1, :]
    expr_prod = expr_harvest['production'][-1, :]
    record = get_record(ctrl_prod, expr_prod, col='Production')
    Table4_df = add_to_table(Table4_df, record, 5)

    # gains
    m2_to_Mu = 667
    ctrl_gains = ctrl_harvest['gains'][-1, :]*m2_to_Mu
    expr_gains = expr_harvest['gains'][-1, :]*m2_to_Mu
    record = get_record(ctrl_gains, expr_gains, col='Gains')
    Table4_df = add_to_table(Table4_df, record, 6)

    return ctrl_gains, expr_gains


def CalculateBalance(Table4_df, ctrl_economic, expr_economic):
    ctrl_balance = ctrl_economic['gains']-ctrl_economic['cost']
    expr_balance = expr_economic['gains']-expr_economic['cost']

    record = get_record(ctrl_balance, expr_balance, col='Net Profit')
    Table4_df = add_to_table(Table4_df, record, 7)

    # save
    save_path = args.base_tmp_folder + '/table4/'
    mkdir(save_path)
    Table4_df.to_csv(save_path+'Overall_economic.csv', index=False)


def Table4(args):
    print("=============Table4===============")

    # file
    harvest_file = os.path.join(args.base_input_path, args.harvest_files)
    with open(harvest_file, 'r') as f:
        harvest_file_dir = f.readlines()
    harvest_file_dir = harvest_file_dir[0].replace("\n", '')

    columns = ['Economic',
               'Control Group',
               'Experimental Group',
               'RI*',
               'T-test']
    Table4_df = pd.DataFrame(np.full((8, 5), np.nan), columns=columns)

    ctrl_cost, expr_cost = CalculateCost(Table4_df, harvest_file_dir)
    ctrl_gains, expr_gains = CalculateHarvest(Table4_df, harvest_file_dir)

    ctrl_economic = {"cost": ctrl_cost,
                     "gains": ctrl_gains}
    expr_economic = {"cost": expr_cost,
                     "gains": expr_gains}
    CalculateBalance(Table4_df, ctrl_economic, expr_economic)


def add_to_table(Table4_df, record, rowIdx):
    columns = Table4_df.columns
    for i in range(len(record.keys())):
        Table4_df[columns[i]].iloc[rowIdx] = record[i]

    return Table4_df


def get_record(ctrl, expr, col, minus=''):
    ctrl_mean = np.mean(ctrl)
    expr_mean = np.mean(expr)
    ctrl_std = np.std(ctrl)
    expr_std = np.std(expr)
    t, p_ = stats.ttest_1samp(expr, ctrl.mean())
    record = {0: col,
              1: '+-'.join([str(np.around(ctrl_mean, 2)), str(np.around(ctrl_std, 2))]),
              2: "+-".join([str(np.around(expr_mean, 2)), str(np.around(expr_std, 2))]),
              3: f"{minus}{np.around(abs(ctrl_mean-expr_mean) / ctrl_mean*100, 2)} %",
              4: str(p_)}
    return record


def get_harvest(args):
    harvest_file = os.path.join(args.base_input_path, args.harvest_files)
    with open(harvest_file, 'r') as f:
        harvest_file_dir = f.readlines()
    harvest_file_dir = harvest_file_dir[0].replace("\n", '')

    expr_harvest, ctrl_harvest = harvest_analysis(args=args,
                                                  harvest_dir=harvest_file_dir)

    return expr_harvest, ctrl_harvest


def harvest_analysis(args, harvest_dir):
    # 结果初始化
    startDate = datetime.datetime.strptime(args.startDate, "%Y-%m-%d")
    endDate = datetime.datetime.strptime(args.endDate, "%Y-%m-%d")
    days = (endDate-startDate).days + 1
    expr_prod = np.zeros((days, len(args.experiment_gh)))
    ctrl_prod = np.zeros((days, len(args.control_group)))
    expr_gains = np.zeros((days, len(args.experiment_gh)))
    ctrl_gains = np.zeros((days, len(args.control_group)))
    # 读取收成数据文件
    m2_to_Mu = 667
    production = pd.read_csv(harvest_dir + 'production.csv')
    production = production.values[:, 1:] / m2_to_Mu
    Income = pd.read_csv(harvest_dir + 'Income.csv')
    Income = Income.values[:, 1:] / m2_to_Mu * args.rmb2euro

    ctrl_prod[-len(production):, :] = np.nancumsum(production[:, :2], axis=0)
    expr_prod[-len(production):, :] = np.nancumsum(production[:, 2:], axis=0)

    ctrl_gains[-len(Income):, :] = np.nancumsum(Income[:, :2], axis=0)
    expr_gains[-len(Income):, :] = np.nancumsum(Income[:, 2:], axis=0)

    expr_harvest = {"production": expr_prod,
                    "gains": expr_gains}
    ctrl_harvest = {"production": ctrl_prod,
                    "gains": ctrl_gains}

    return expr_harvest, ctrl_harvest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--startDate', default="2020-03-15",
                        help='start date of planting.')
    parser.add_argument('--endDate', default="2020-07-13",
                        help='end date of planting.')
    parser.add_argument('--control_group', type=list, default=[1,2],
                        help='ids of all green house.')
    parser.add_argument('--experiment_gh', type=list, default=[3,4,5,6,7],
                        help='ids of all green house.')
    parser.add_argument('--rmb2euro', type=float, default=0.1276,
                        help="rate of rmb to euro")
    parser.add_argument("--base_input_path", default="./input", type=str)
    parser.add_argument("--base_tmp_folder", default="./result", type=str)
    parser.add_argument("--harvest_files", default='harvest.txt', type=str)
    args = parser.parse_args()

    Table4(args)

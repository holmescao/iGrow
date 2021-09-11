import math
import os
import time
import numpy as np
import pandas as pd
import scipy.io as scio
import geatpy as ea
import warnings


class Problem:
    """
Problem : Class - 问题类

描述:
    问题类是用来存储与待求解问题相关信息的一个类。

属性:
    name      : str   - 问题名称（可以自由设置名称）。

    M         : int   - 目标维数，即有多少个优化目标。

    maxormins : array - 目标函数最小最大化标记的行向量，1表示最小化，-1表示最大化，例如：
                        array([1,1,-1,-1])，表示前2个目标是最小化，后2个目标是最大化。

    Dim       : int   - 决策变量维数，即有多少个决策变量。

    varTypes  : array - 连续或离散标记，是Numpy array类型的行向量，
                        0表示对应的决策变量是连续的；1表示对应的变量是离散的。

    ranges    : array - 决策变量范围矩阵，第一行对应决策变量的下界，第二行对应决策变量的上界。

    borders   : array - 决策变量范围的边界矩阵，第一行对应决策变量的下边界，第二行对应决策变量的上边界，
                        0表示范围中不含边界，1表示范围包含边界。

函数:
    aimFunc(pop) : 目标函数，需要在继承类即自定义的问题类中实现，或是传入已实现的函数。
                   其中pop为Population类的对象，代表一个种群，
                   pop对象的Phen属性（即种群染色体的表现型）等价于种群所有个体的决策变量组成的矩阵，
                   该函数根据该Phen计算得到种群所有个体的目标函数值组成的矩阵，并将其赋值给pop对象的ObjV属性。
                   若有约束条件，则在计算违反约束程度矩阵CV后赋值给pop对象的CV属性（详见Geatpy数据结构）。
                   该函数不返回任何的返回值，求得的目标函数值保存在种群对象的ObjV属性中。
                   例如：population为一个种群对象，则调用aimFunc(population)即可完成目标函数值的计算，
                   此时可通过population.ObjV得到求得的目标函数值，population.CV得到违反约束程度矩阵。

    calReferObjV()   : 计算目标函数参考值，需要在继承类中实现，或是传入已实现的函数。

    getReferObjV()   : 获取目标函数参考值。

"""

    def __init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin, aimFunc=None, calReferObjV=None):
        self.name = name
        self.M = M
        self.maxormins = np.array(maxormins)
        self.Dim = Dim
        self.varTypes = np.array(varTypes)
        self.ranges = np.array([lb, ub])  # 初始化ranges（决策变量范围矩阵）
        self.borders = np.array([lbin, ubin])  # 初始化borders（决策变量范围边界矩阵）
        self.aimFunc = aimFunc if aimFunc is not None else self.aimFunc  # 初始化目标函数接口
        # 初始化理论最优值计算函数接口
        self.calReferObjV = calReferObjV if calReferObjV is not None else self.calReferObjV

    def aimFunc(self, pop):
        raise RuntimeError(
            'error in Problem: aimFunc has not been initialized. (未在问题子类中设置目标函数！)')

    def calReferObjV(self):
        return None

    def getReferObjV(self, reCalculate=False):
        """
        描述: 该函数用于读取/计算问题的目标函数参考值，这个参考值可以是理论上的全局最优解的目标函数值，也可以是人为设定的非最优的目标函数参考值。
        reCalculate是一个bool变量，用于判断是否需要调用calReferObjV()来重新计算目标函数参考值。
        默认情况下reCalculate是False，此时将先尝试读取理论全局最优解的数据，
        若读取不到，则尝试调用calReferObjV()来计算理论全局最优解。
        在计算理论全局最优解后，
        将结果按照“问题名称_目标维数_决策变量个数.csv”的文件命名把数据保存到referenceObjV文件夹内。

        """

        if os.path.exists('referenceObjV') == False:
            os.makedirs('referenceObjV')
        if reCalculate == False:
            # 尝试读取数据
            if os.path.exists('referenceObjV/' + self.name + '_M' + str(self.M) + '_D' + str(self.Dim) + '.csv'):
                return np.loadtxt('referenceObjV/' + self.name + '_M' + str(self.M) + '_D' + str(self.Dim) + '.csv',
                                  delimiter=',')
        # 若找不到数据，则调用calReferObjV()计算目标函数参考值
        referenceObjV = self.calReferObjV()
        if referenceObjV is not None:
            # 保存数据
            np.savetxt('referenceObjV/' + self.name + '_M' + str(self.M) + '_D' + str(self.Dim) + '.csv', referenceObjV,
                       delimiter=',')
        else:
            print('未找到目标函数参考值数据！')
        return referenceObjV


class Population:
    """
Population : class - 种群类

描述:
    种群类是用来存储种群相关信息的一个类。

属性:
    sizes    : int   - 种群规模，即种群的个体数目。

    ChromNum : int   - 染色体的数目，即每个个体有多少条染色体。

    Encoding : str   - 染色体编码方式，
                       'BG':二进制/格雷编码；
                       'RI':实整数编码，即实数和整数的混合编码；
                       'P':排列编码。
                       相关概念：术语“实值编码”包含实整数编码和排列编码，
                       它们共同的特点是染色体不需要解码即可直接表示对应的决策变量。
                       "实整数"指的是种群染色体既包含实数的小数，也包含实数的整数。
                       特殊用法：
                       设置Encoding=None，此时种群类的Field,Chrom成员属性将被设置为None，
                       种群将不携带与染色体直接相关的信息，可以减少不必要的数据存储，
                       这种用法可以在只想统计非染色体直接相关的信息时使用，
                       尤其可以在多种群进化优化过程中对个体进行统一的适应度评价时使用。

    Field    : array - 译码矩阵，可以是FieldD或FieldDR（详见Geatpy数据结构）。

    Chrom    : array - 种群染色体矩阵，每一行对应一个个体的一条染色体。

    Lind     : int   - 种群染色体长度。

    ObjV     : array - 种群目标函数值矩阵，每一行对应一个个体的目标函数值，每一列对应一个目标。

    FitnV    : array - 种群个体适应度列向量，每个元素对应一个个体的适应度，最小适应度为0。

    CV       : array - CV(Constraint Violation Value)是用来定量描述违反约束条件程度的矩阵，每行对应一个个体，每列对应一个约束。
                       注意：当没有设置约束条件时，CV设置为None。

    Phen     : array - 种群表现型矩阵（即种群各染色体解码后所代表的决策变量所组成的矩阵）。

函数:
    详见源码。

"""

    def __init__(self, Encoding, Field, NIND, Chrom=None, ObjV=None, FitnV=None, CV=None, Phen=None):
        """
        描述: 种群类的构造方法，用于实例化种群对象，例如：
             import geatpy as ea
             population = ea.Population(Encoding, Field, NIND)，
             NIND为所需要的个体数，
             此时得到的population还没被真正初始化，仅仅是完成种群对象的实例化。
             该构造方法必须传入Chrom，才算是完成种群真正的初始化。
             一开始可以只传入Encoding, Field以及NIND来完成种群对象的实例化，
             其他属性可以后面再通过计算进行赋值。

        """

        if type(NIND) is int and NIND >= 0:
            self.sizes = NIND
        else:
            raise RuntimeError(
                'error in Population: Size error. (种群规模设置有误，必须为非负整数。)')
        self.ChromNum = 1
        self.Encoding = Encoding
        if Encoding is None:
            self.Field = None
            self.Chrom = None
        else:
            self.Field = Field.copy()
            self.Chrom = Chrom.copy() if Chrom is not None else None
        self.Lind = Chrom.shape[1] if Chrom is not None else 0
        self.ObjV = ObjV.copy() if ObjV is not None else None
        self.FitnV = FitnV.copy() if FitnV is not None else None
        self.CV = CV.copy() if CV is not None else None
        self.Phen = Phen.copy() if Phen is not None else None

    def initChrom(self, NIND=None):
        """
        描述: 初始化种群染色体矩阵，NIND为所需要的个体数。
        NIND可缺省，不缺省时，种群在初始化染色体矩阵前会把种群规模调整为NIND。

        """

        if NIND is not None:
            self.sizes = NIND  # 重新设置种群规模
        self.Chrom = ea.crtpc(self.Encoding, self.sizes, self.Field)  # 生成染色体矩阵
        self.Lind = self.Chrom.shape[1]  # 计算染色体的长度
        self.ObjV = None
        self.FitnV = None
        self.CV = None

    def decoding(self):
        """
        描述: 种群染色体解码。

        """

        if self.Encoding == 'BG':  # 此时Field实际上为FieldD
            # Phen = ea.bs2ri(self.Chrom, self.Field)  # 把二进制/格雷码转化为实整数
            Phen = self.bi2dec(self.Chrom, self.Field)
        elif self.Encoding == 'RI' or self.Encoding == 'P':
            Phen = self.Chrom.copy()
        else:
            raise RuntimeError(
                'error in Population.decoding: Encoding must be ''BG'' or ''RI'' or ''P''. (编码设置有误，解码时Encoding必须为''BG'', ''RI'' 或 ''P''。)')
        return Phen

    def copy(self):
        """
        copy : function - 种群的复制
        用法:
            假设pop是一个种群矩阵，那么：pop1 = pop.copy()即可完成对pop种群的复制。

        """

        return Population(self.Encoding,
                          self.Field,
                          self.sizes,
                          self.Chrom,
                          self.ObjV,
                          self.FitnV,
                          self.CV,
                          self.Phen)

    def __getitem__(self, index):
        """
        描述: 种群的切片，即根据index下标向量选出种群中相应的个体组成一个新的种群。
        用法: 假设pop是一个包含多于2个个体的种群矩阵，那么：
             pop1 = pop[[0,1]]即可得到由pop种群的第1、2个个体组成的种群。
        注意: index必须为一个slice或者为一个Numpy array类型的行向量或者为一个list类型的列表，
             该函数不对传入的index参数的合法性进行详细检查。

        """

        if self.Encoding is None:
            NewChrom = None
        else:
            if self.Chrom is None:
                raise RuntimeError(
                    'error in Population: Chrom is None. (种群染色体矩阵未初始化。)')
            NewChrom = self.Chrom[index]
        # 计算切片后的长度
        if type(index) != slice and type(index) != np.ndarray and type(index) != list:
            raise RuntimeError(
                'error in Population: index must be a 1-D array. (index必须是一个一维的向量。)')
        if type(index) == slice:
            NIND = (index.stop - (index.start if index.start is not None else 0)) // (
                index.step if index.step is not None else 1)
        else:
            index_array = np.array(index)
            if index_array.dtype == bool:
                NIND = int(np.sum(index_array))
            else:
                NIND = len(index_array)
        return Population(self.Encoding,
                          self.Field,
                          NIND,
                          NewChrom,
                          self.ObjV[index] if self.ObjV is not None else None,
                          self.FitnV[index] if self.FitnV is not None else None,
                          self.CV[index] if self.CV is not None else None,
                          self.Phen[index] if self.Phen is not None else None)

    def shuffle(self):
        """
        shuffle : function - 打乱种群个体的个体顺序
        用法: 假设pop是一个种群矩阵，那么，pop.shuffle()即可完成对pop种群个体顺序的打乱。

        """

        shuff = np.argsort(np.random.rand(self.sizes))
        if self.Encoding is None:
            self.Chrom = None
        else:
            if self.Chrom is None:
                raise RuntimeError(
                    'error in Population: Chrom is None. (种群染色体矩阵未初始化。)')
            self.Chrom = self.Chrom[shuff, :]
        self.ObjV = self.ObjV[shuff, :] if self.ObjV is not None else None
        self.FitnV = self.FitnV[shuff] if self.FitnV is not None else None
        self.CV = self.CV[shuff, :] if self.CV is not None else None
        self.Phen = self.Phen[shuff, :] if self.Phen is not None else None

    def __setitem__(self, index, pop):  # 种群个体赋值（种群个体替换）
        """
        描述: 种群个体的赋值
        用法: 假设pop是一个包含多于2个个体的种群矩阵，pop1是另一个包含2个个体的种群矩阵，那么
             pop[[0,1]] = pop1，即可完成将pop种群的第1、2个个体赋值为pop1种群的个体。
        注意: index必须是一个Numpy array类型的行向量，该函数不会对传入的index的合法性进行检查。
             此外，进行种群个体替换后，该函数不会对适应度进行主动重置，
             如果因个体替换而需要重新对所有个体的适应度进行评价，则需要手写代码更新种群的适应度。

        """

        if self.Encoding is not None:
            if self.Encoding != pop.Encoding:
                raise RuntimeError(
                    'error in Population: Encoding disagree. (两种群染色体的编码方式必须一致。)')
            if np.all(self.Field == pop.Field) == False:
                raise RuntimeError(
                    'error in Population: Field disagree. (两者的译码矩阵必须一致。)')
            if self.Chrom is None:
                raise RuntimeError(
                    'error in Population: Chrom is None. (种群染色体矩阵未初始化。)')
            self.Chrom[index] = pop.Chrom
        if (self.ObjV is None) ^ (pop.ObjV is None):
            raise RuntimeError(
                'error in Population: ObjV disagree. (两者的目标函数值矩阵必须要么同时为None要么同时不为None。)')
        if (self.FitnV is None) ^ (pop.FitnV is None):
            raise RuntimeError(
                'error in Population: FitnV disagree. (两者的适应度列向量必须要么同时为None要么同时不为None。)')
        if (self.CV is None) ^ (pop.CV is None):
            raise RuntimeError(
                'error in Population: CV disagree. (两者的违反约束程度矩阵必须要么同时为None要么同时不为None。)')
        if (self.Phen is None) ^ (pop.Phen is None):
            raise RuntimeError(
                'error in Population: Phen disagree. (两者的表现型矩阵必须要么同时为None要么同时不为None。)')
        if self.ObjV is not None:
            self.ObjV[index] = pop.ObjV
        if self.FitnV is not None:
            self.FitnV[index] = pop.FitnV
        if self.CV is not None:
            self.CV[index] = pop.CV
        if self.Phen is not None:
            self.Phen[index] = pop.Phen
        self.sizes = self.Phen.shape[0]  # 更新种群规模

    def __add__(self, pop):
        """
        描述: 种群个体合并
        用法: 假设pop1, pop2是两个种群，它们的个体数可以相等也可以不相等，此时
             pop = pop1 + pop2，即可完成对pop1和pop2两个种群个体的合并。
        注意：
            进行种群合并后，该函数不会对适应度进行主动重置，
            如果因种群合并而需要重新对所有个体的适应度进行评价，则需要手写代码更新种群的适应度。

        """

        if self.Encoding is None:
            NewChrom = None
        else:
            if self.Encoding != pop.Encoding:
                raise RuntimeError(
                    'error in Population: Encoding disagree. (两种群染色体的编码方式必须一致。)')
            if self.Chrom is None or pop.Chrom is None:
                raise RuntimeError(
                    'error in Population: Chrom is None. (种群染色体矩阵未初始化。)')
            if np.all(self.Field == pop.Field) == False:
                raise RuntimeError(
                    'error in Population: Field disagree. (两者的译码矩阵必须一致。)')
            NewChrom = np.vstack([self.Chrom, pop.Chrom])
        if (self.ObjV is None) ^ (pop.ObjV is None):
            raise RuntimeError(
                'error in Population: ObjV disagree. (两者的目标函数值矩阵必须要么同时为None要么同时不为None。)')
        if (self.CV is None) ^ (pop.CV is None):
            raise RuntimeError(
                'error in Population: CV disagree. (两者的违反约束程度矩阵必须要么同时为None要么同时不为None。)')
        if (self.Phen is None) ^ (pop.Phen is None):
            raise RuntimeError(
                'error in Population: Phen disagree. (两者的表现型矩阵必须要么同时为None要么同时不为None。)')
        NIND = self.sizes + pop.sizes  # 得到合并种群的个体数
        return Population(self.Encoding,
                          self.Field,
                          NIND,
                          NewChrom,
                          np.vstack([self.ObjV, pop.ObjV]
                                    ) if self.ObjV is not None else None,
                          np.vstack(
                              [self.FitnV, pop.FitnV]) if self.FitnV is not None and pop.FitnV is not None else None,
                          np.vstack([self.CV, pop.CV]
                                    ) if self.CV is not None else None,
                          np.vstack([self.Phen, pop.Phen]) if self.Phen is not None else None)

    def __len__(self):
        """
        描述: 计算种群规模
        用法: 假设pop是一个种群，那么len(pop)即可得到该种群的个体数。
             实际上，种群规模也可以通过pop.sizes得到。

        """

        return self.sizes

    def save(self):
        """
        描述: 把种群的信息保存到文件中。
        该函数将在"Result"文件夹下保存种群的信息，其中：
        "Encoding.txt"保存种群的染色体编码；
        "Field.csv"保存种群染色体的译码矩阵；
        "Chrom.csv"保存种群的染色体矩阵；
        "ObjV.csv"保存种群的目标函数矩阵；
        "FitnV.csv"保存种群个体的适应度列向量；
        "CV.csv"保存种群个体的违反约束程度矩阵；
        "Phen.csv"保存种群染色体表现型矩阵；
        注意：该函数不会对种群的合法性进行检查。

        """
        return None

        # if os.path.exists('Result') == False:
        #     os.makedirs('Result')
        # with open('Result/Encoding.txt', 'w') as file:
        #     file.write(str(self.Encoding))
        #     file.close()
        # if self.Encoding is not None:
        #     np.savetxt('Result/Field.csv', self.Field, delimiter=',')
        #     np.savetxt('Result/Chrom.csv', self.Chrom, delimiter=',')
        # if self.ObjV is not None:
        #     np.savetxt('Result/ObjV.csv', self.ObjV, delimiter=',')
        # if self.FitnV is not None:
        #     np.savetxt('Result/FitnV.csv', self.FitnV, delimiter=',')
        # if self.CV is not None:
        #     np.savetxt('Result/CV.csv', self.CV, delimiter=',')
        # if self.Phen is not None:
        #     np.savetxt('Result/Phen.csv', self.Phen, delimiter=',')
        # print('种群信息导出完毕。')

    def load_latest_population(self, population):
        """
        描述: 把种群的信息从文件提取，并返回给population对象。
        该函数将在"Result"文件夹下保存种群的信息，其中：
        #"Encodingsi.txt"保存种群的染色体编码，i为0,1,2,3...；
        #"Fieldsi.csv"保存种群染色体的译码矩阵，i为0,1,2,3...；
        "Chromsi.csv"保存种群的染色体矩阵，i为0,1,2,3...；
        "ObjV.csv"保存种群的目标函数矩阵；
        "FitnV.csv"保存种群个体的适应度列向量；
        "CV.csv"保存种群个体的违反约束程度矩阵；
        "Phen.csv"保存种群染色体表现型矩阵；
        #'ChromNum'：根据Chroms决定
        'Linds'：根据Chroms决定，是一个list,eg: [32, 12]
        #'size'： 种群规模，根据输入参数决定
        注意：该函数不会对种群的合法性进行检查。

        """

        # Chroms & Linds
        for i in range(population.ChromNum):
            Chroms = pd.read_csv('GA_Search/Result/Chroms' +
                                 str(i) + '.csv', header=None)
            population.Linds[i] = Chroms.shape[1]
            population.Chroms[i] = np.array(Chroms)

        # ObjV
        if os.path.exists('GA_Search/Result/ObjV.csv'):
            ObjV = pd.read_csv('GA_Search/Result/ObjV.csv', header=None)
            population.ObjV = np.array(ObjV)
        # FitnV
        if os.path.exists('GA_Search/Result/FitnV.csv'):
            FitnV = pd.read_csv('GA_Search/Result/FitnV.csv', header=None)
            population.FitnV = np.array(FitnV)
        # CV
        if os.path.exists('GA_Search/Result/CV.csv'):
            CV = pd.read_csv('GA_Search/Result/CV.csv', header=None)
            population.CV = np.array(CV)
        # Phen
        if os.path.exists('GA_Search/Result/Phen.csv'):
            Phen = pd.read_csv('GA_Search/Result/Phen.csv', header=None)
            population.Phen = np.array(Phen)

        print('种群信息加载完毕。')

        return population

    def warmup_Chrom(self, individual, NIND):
        """
        描述: 初始化种群染色体矩阵，NIND为所需要的个体数。
        NIND可缺省，不缺省时，种群在初始化染色体矩阵前会把种群规模调整为NIND。

        """
        individual = np.array(individual)
        if individual.ndim == 1:
            Phen = np.expand_dims(individual, 0).repeat(
                NIND, axis=0)  # 复制个体给整个种群
        elif individual.ndim == 2 and individual.shape[0] == NIND:
            Phen = individual
        else:
            print("个体的数量与设定的种群大小不一致，请查看individual的维度")
        self.sizes = NIND  # 重新设置种群规模
        self.Chrom = self.dec2bi(Phen, self.Field)
        self.Lind = self.Chrom.shape[1]  # 计算染色体的长度
        self.ObjV = None
        self.FitnV = None
        self.CV = None

    def bi2dec(self, Chrom, Field):
        lengthVars = np.cumsum(Field[0])
        posVars = np.concatenate(([0], lengthVars[:-1]))  # 染色体中每个变量的起始位置
        numVar = Field.shape[1]
        temporary = np.zeros((Chrom.shape[0], numVar))
        for i in range(Chrom.shape[0]):
            phen_i = np.zeros(numVar)
            for var in range(numVar):
                total = 0
                var_s = int(posVars[var])
                var_e = int(posVars[var] + Field[0][var])
                for j in range(var_s, var_e):
                    total += Chrom[i][j] * int((math.pow(2, var_e - j - 1)))
                # 判断是否溢出
                if total + Field[1, var] <= Field[2, var]:
                    phen_i[var] = total + Field[1, var]
                else:
                    phen_i[var] = Field[2, var]  # 溢出则采用上界
            temporary[i] = phen_i

        temporary = temporary.astype(int)

        return temporary

    def dec2bi(self, Phen, Field):
        lengthVars = Field[0]  # 每个基因的长度
        posVars = np.concatenate(
            ([0], np.cumsum(lengthVars)[:-1]))  # 染色体中每个变量的起始位置
        pop_size = Phen.shape[0]
        var_num = Phen.shape[1]
        Chrom = np.zeros((pop_size, int(np.sum(lengthVars))))  # 染色体矩阵
        for i in range(pop_size):
            Chrom_i = []
            for var in range(var_num):
                num = Phen[i][var] - Field[1, var]
                arry = []
                while True:
                    arry.append(num % 2)
                    num = num // 2
                    if num == 0:
                        break
                bi = np.array(arry[::-1])
                zero_need = int(lengthVars[var] - len(bi))
                bi_full = np.concatenate(([0] * zero_need, bi))
                Chrom_i += list(bi_full.astype(int))
            Chrom[i] = np.array(Chrom_i)

        Chrom = Chrom.astype(int)

        return Chrom


class PsyPopulation(ea.Population):
    """
PsyPopulation : class - 多染色体种群类(Popysomy Population)

描述:
    多染色体种群类是用来存储每个个体包含多条染色体的种群相关信息的一个类。
    该类和种群类Population似，不同之处是可以包含多条染色体，因此支持复杂的混合编码。

属性:
    sizes     : int   - 种群规模，即种群的个体数目。

    ChromNum  : int   - 染色体的数目，即每个个体有多少条染色体。

    Encodings : list  - 存储各染色体编码方式的列表。

    Fields    : list  - 存储各染色体对应的译码矩阵的列表。

    Chroms    : list  - 存储种群各染色体矩阵的列表。

    Linds     : list  - 存储种群各染色体长度的列表。

    ObjV      : array - 种群目标函数值矩阵，每一行对应一个个体的目标函数值，每一列对应一个目标。

    FitnV     : array - 种群个体适应度列向量，每个元素对应一个个体的适应度，最小适应度为0。

    CV        : array - CV(Constraint Violation Value)是用来定量描述违反约束条件程度的矩阵，每行对应一个个体，每列对应一个约束。
                        注意：当没有设置约束条件时，CV设置为None。

    Phen      : array - 种群表现型矩阵（即染色体解码后所代表的决策变量所组成的矩阵）。

函数:
    详见源码。

"""

    def __init__(self, Encodings, Fields, NIND, Chroms=None, ObjV=None, FitnV=None, CV=None, Phen=None):
        """
        描述: 种群类的构造方法，用于实例化种群对象，例如：
             import geatpy as ea
             population = ea.PsyPopulation(Encodings, Fields, NIND)，
             NIND为所需要的个体数，
             此时得到的population还没被真正初始化，仅仅是完成种群对象的实例化。
             该构造方法必须传入Chroms，才算是完成种群真正的初始化。
             一开始可以只传入Encodings, Fields以及NIND来完成种群对象的实例化，
             其他属性可以后面再通过计算进行赋值。

        """

        if type(NIND) is int and NIND >= 0:
            self.sizes = NIND
        else:
            raise RuntimeError(
                'error in PysPopulation: Size error. (种群规模设置有误，必须为非负整数。)')
        self.ChromNum = len(Encodings)
        if self.ChromNum == 1:
            raise RuntimeError(
                'error in PysPopulation: ChromNum must be bigger than 1. (使用PysPopulation类时，染色体数目必须大于1，否则应该使用Population类。)')
        self.Encodings = Encodings
        self.Fields = Fields.copy()
        self.Chroms = [None] * self.ChromNum  # 初始化Chroms为元素是None的列表
        self.Linds = []
        if Chroms is None:
            self.Linds = [0] * self.ChromNum
        else:
            for i in range(self.ChromNum):
                if Chroms[i] is not None:
                    self.Linds.append(Chroms[i].shape[1])
                    self.Chroms[i] = Chroms[i].copy(
                    ) if Chroms[i] is not None else None
                else:
                    self.Linds.append(0)
        self.ObjV = ObjV.copy() if ObjV is not None else None
        self.FitnV = FitnV.copy() if FitnV is not None else None
        self.CV = CV.copy() if CV is not None else None
        self.Phen = Phen.copy() if Phen is not None else None

    def initChrom(self, NIND=None):
        """
        描述: 初始化种群染色体矩阵，NIND为所需要的个体数。
        NIND可缺省，不缺省时，种群在初始化染色体矩阵前会把种群规模调整为NIND。

        """

        if NIND is not None:
            self.sizes = NIND  # 重新设置种群规模
        # 遍历各染色体矩阵进行初始化
        for i in range(self.ChromNum):
            self.Chroms[i] = ea.crtpc(
                self.Encodings[i], self.sizes, self.Fields[i])  # 生成染色体矩阵
            self.Linds.append(self.Chroms[i].shape[1])  # 计算染色体的长度
        self.ObjV = None
        self.FitnV = np.ones((self.sizes, 1))  # 默认适应度全为1
        self.CV = None

    def warmup_Chroms(self, individual, NIND):
        """
        描述: 初始化种群染色体矩阵，NIND为所需要的个体数。
        NIND可缺省，不缺省时，种群在初始化染色体矩阵前会把种群规模调整为NIND。

        """
        individual = np.array(individual)
        if individual.ndim == 1:
            Phen = np.expand_dims(individual, 0).repeat(
                NIND, axis=0)  # 复制个体给整个种群
        elif individual.ndim == 2 and individual.shape[0] == NIND:
            Phen = individual
        else:
            print("个体的数量与设定的种群大小不一致，请查看individual的维度")
        # 遍历各染色体矩阵进行初始化
        idx = 0
        for i in range(self.ChromNum):
            self.Chroms[i] = self.dec2bi(
                Phen[:, idx: idx+self.Fields[i].shape[1]], self.Fields[i])  # 生成染色体矩阵
            self.Linds.append(self.Chroms[i].shape[1])  # 计算染色体的长度

            idx += self.Fields[i].shape[1]

        self.sizes = NIND
        self.ObjV = None
        self.FitnV = None
        self.CV = None

    def bi2dec(self, Chrom, Field):
        lengthVars = np.cumsum(Field[0])
        posVars = np.concatenate(([0], lengthVars[:-1]))  # 染色体中每个变量的起始位置
        numVar = Field.shape[1]
        temporary = np.zeros((Chrom.shape[0], numVar))
        for i in range(Chrom.shape[0]):
            phen_i = np.zeros(numVar)
            for var in range(numVar):
                total = 0
                var_s = int(posVars[var])
                var_e = int(posVars[var] + Field[0][var])
                for j in range(var_s, var_e):
                    total += Chrom[i][j] * int((math.pow(2, var_e - j - 1)))
                # 判断是否溢出
                if total + Field[1, var] <= Field[2, var]:
                    phen_i[var] = total + Field[1, var]
                else:
                    phen_i[var] = Field[2, var]  # 溢出则采用上界
            temporary[i] = phen_i

        temporary = temporary.astype(int)

        return temporary

    def dec2bi(self, Phen, Field):
        lengthVars = Field[0]  # 每个基因的长度
        posVars = np.concatenate(
            ([0], np.cumsum(lengthVars)[:-1]))  # 染色体中每个变量的起始位置
        pop_size = Phen.shape[0]
        var_num = Phen.shape[1]
        Chrom = np.zeros((pop_size, int(np.sum(lengthVars))))  # 染色体矩阵
        for i in range(pop_size):
            Chrom_i = []
            for var in range(var_num):
                num = Phen[i][var] - Field[1, var]
                arry = []
                while True:
                    arry.append(num % 2)
                    num = num // 2
                    if num == 0:
                        break
                bi = np.array(arry[::-1])
                zero_need = int(lengthVars[var] - len(bi))
                bi_full = np.concatenate(([0] * zero_need, bi))
                Chrom_i += list(bi_full.astype(int))
            Chrom[i] = np.array(Chrom_i)

        Chrom = Chrom.astype(int)

        return Chrom

    def decoding(self):
        """
        描述: 种群染色体解码。

        """

        Phen = np.ones((self.sizes, 0))  # 初始化一个空的矩阵
        # 遍历各染色体矩阵进行解码
        for i in range(self.ChromNum):
            if self.Encodings[i] == 'BG':  # 此时Field实际上为FieldD
                tempPhen = self.bi2dec(self.Chroms[i], self.Fields[i])
                # tempPhen = ea.bs2ri(
                #     self.Chroms[i], self.Fields[i])  # 把二进制/格雷码转化为实整数
            elif self.Encodings[i] == 'RI' or self.Encodings[i] == 'P':
                tempPhen = self.Chroms[i].copy()
            else:
                raise RuntimeError(
                    'error in PsyPopulation.decoding: Encoding must be ''BG'' or ''RI'' or ''P''. (编码设置有误，Encoding必须为''BG'', ''RI'' 或 ''P''。)')
            Phen = np.hstack([Phen, tempPhen])

        return Phen

    def copy(self):
        """
        copy : function - 种群的复制
        用法:
            假设pop是一个种群矩阵，那么：pop1 = pop.copy()即可完成对pop种群的复制。

        """

        return PsyPopulation(self.Encodings,
                             self.Fields,
                             self.sizes,
                             self.Chroms,
                             self.ObjV,
                             self.FitnV,
                             self.CV,
                             self.Phen)

    def __getitem__(self, index):
        """
        描述: 种群的切片，即根据index下标向量选出种群中相应的个体组成一个新的种群。
        用法: 假设pop是一个包含多于2个个体的种群矩阵，那么：
             pop1 = pop[[0,1]]即可得到由pop种群的第1、2个个体组成的种群。
        注意: 这里不对index的合法性进行检查。

        """

        NewChroms = []
        for i in range(self.ChromNum):
            if self.Chroms[i] is None:
                raise RuntimeError(
                    'error in PsyPopulation: Chrom[i] is None. (种群染色体矩阵未初始化。)')
            NewChroms.append(self.Chroms[i][index])
        NIND = NewChroms[0].shape[0]
        return PsyPopulation(self.Encodings,
                             self.Fields,
                             NIND,
                             NewChroms,
                             self.ObjV[index] if self.ObjV is not None else None,
                             self.FitnV[index] if self.FitnV is not None else None,
                             self.CV[index] if self.CV is not None else None,
                             self.Phen[index] if self.Phen is not None else None)

    def shuffle(self):
        """
        shuffle : function - 打乱种群个体的个体顺序
        用法: 假设pop是一个种群矩阵，那么，pop.shuffle()即可完成对pop种群个体顺序的打乱。

        """

        shuff = np.argsort(np.random.rand(self.sizes))
        for i in range(self.ChromNum):
            if self.Chroms[i] is None:
                raise RuntimeError(
                    'error in PsyPopulation: Chrom[i] is None. (种群染色体矩阵未初始化。)')
            self.Chroms[i] = self.Chroms[i][shuff, :]
        self.ObjV = self.ObjV[shuff, :] if self.ObjV is not None else None
        self.FitnV = self.FitnV[shuff] if self.FitnV is not None else None
        self.CV = self.CV[shuff, :] if self.CV is not None else None
        self.Phen = self.Phen[shuff, :] if self.Phen is not None else None

    def __setitem__(self, index, pop):  # 种群个体赋值（种群个体替换）
        """
        描述: 种群个体的赋值
        用法: 假设pop是一个包含多于2个个体的种群矩阵，pop1是另一个包含2个个体的种群矩阵，那么
             pop[[0,1]] = pop1，即可完成将pop种群的第1、2个个体赋值为pop1种群的个体。
        注意: index必须是一个Numpy array类型的行向量，该函数不会对传入的index的合法性进行检查。
             此外，进行种群个体替换后，该函数不会对适应度进行主动重置，
             如果因个体替换而需要重新对所有个体的适应度进行评价，则需要手写代码更新种群的适应度。

        """

        for i in range(self.ChromNum):
            if self.Encodings[i] != pop.Encodings[i]:
                raise RuntimeError(
                    'error in PsyPopulation: Encoding disagree. (两种群染色体的编码方式必须一致。)')
            if np.all(self.Fields[i] == pop.Fields[i]) == False:
                raise RuntimeError(
                    'error in PsyPopulation: Field disagree. (两者的译码矩阵必须一致。)')
            if self.Chroms[i] is None:
                raise RuntimeError(
                    'error in PsyPopulation: Chrom[i] is None. (种群染色体矩阵未初始化。)')
            self.Chroms[i][index] = pop.Chroms[i]
        if (self.ObjV is None) ^ (pop.ObjV is None):
            raise RuntimeError(
                'error in PsyPopulation: ObjV disagree. (两者的目标函数值矩阵必须要么同时为None要么同时不为None。)')
        if (self.FitnV is None) ^ (pop.FitnV is None):
            raise RuntimeError(
                'error in PsyPopulation: FitnV disagree. (两者的适应度列向量必须要么同时为None要么同时不为None。)')
        if (self.CV is None) ^ (pop.CV is None):
            raise RuntimeError(
                'error in PsyPopulation: CV disagree. (两者的违反约束程度矩阵必须要么同时为None要么同时不为None。)')
        if (self.Phen is None) ^ (pop.Phen is None):
            raise RuntimeError(
                'error in PsyPopulation: Phen disagree. (两者的表现型矩阵必须要么同时为None要么同时不为None。)')
        if self.ObjV is not None:
            self.ObjV[index] = pop.ObjV
        if self.FitnV is not None:
            self.FitnV[index] = pop.FitnV
        if self.CV is not None:
            self.CV[index] = pop.CV
        if self.Phen is not None:
            self.Phen[index] = pop.Phen
        self.sizes = self.Phen.shape[0]  # 更新种群规模

    def __add__(self, pop):
        """
        描述: 种群个体合并
        用法: 假设pop1, pop2是两个种群，它们的个体数可以相等也可以不相等，此时
             pop = pop1 + pop2，即可完成对pop1和pop2两个种群个体的合并。
        注意：
            进行种群合并后，该函数不会对适应度进行主动重置，
            如果因种群合并而需要重新对所有个体的适应度进行评价，则需要手写代码更新种群的适应度。

        """

        NIND = self.sizes + pop.sizes  # 得到合并种群的个体数
        NewChroms = self.Chroms
        for i in range(self.ChromNum):
            if self.Encodings[i] != pop.Encodings[i]:
                raise RuntimeError(
                    'error in PsyPopulation: Encoding disagree. (两种群染色体的编码方式必须一致。)')
            if np.all(self.Fields[i] == pop.Fields[i]) == False:
                raise RuntimeError(
                    'error in PsyPopulation: Field disagree. (两者的译码矩阵必须一致。)')
            if self.Chroms[i] is None or pop.Chroms[i] is None:
                raise RuntimeError(
                    'error in PsyPopulation: Chrom is None. (种群染色体矩阵未初始化。)')
            NewChroms[i] = np.vstack([NewChroms[i], pop.Chroms[i]])
        if (self.ObjV is None) ^ (pop.ObjV is None):
            raise RuntimeError(
                'error in PsyPopulation: ObjV disagree. (两者的目标函数值矩阵必须要么同时为None要么同时不为None。)')
        if (self.CV is None) ^ (pop.CV is None):
            raise RuntimeError(
                'error in PsyPopulation: CV disagree. (两者的违反约束程度矩阵必须要么同时为None要么同时不为None。)')
        if (self.Phen is None) ^ (pop.Phen is None):
            raise RuntimeError(
                'error in PsyPopulation: Phen disagree. (两者的表现型矩阵必须要么同时为None要么同时不为None。)')
        return PsyPopulation(self.Encodings,
                             self.Fields,
                             NIND,
                             NewChroms,
                             np.vstack([self.ObjV, pop.ObjV]
                                       ) if self.ObjV is not None else None,
                             np.vstack(
                                 [self.FitnV, pop.FitnV]) if self.FitnV is not None and pop.FitnV is not None else None,
                             np.vstack([self.CV, pop.CV]
                                       ) if self.CV is not None else None,
                             np.vstack([self.Phen, pop.Phen]) if self.Phen is not None else None)

    def __len__(self):
        """
        描述: 计算种群规模
        用法: 假设pop是一个种群，那么len(pop)即可得到该种群的个体数。
             实际上，种群规模也可以通过pop.sizes得到。

        """

        return self.sizes

    def save(self):
        """
        描述: 把种群的信息保存到文件中。
        该函数将在"Result"文件夹下保存种群的信息，其中：
        "Encodingsi.txt"保存种群的染色体编码，i为0,1,2,3...；
        "Fieldsi.csv"保存种群染色体的译码矩阵，i为0,1,2,3...；
        "Chromsi.csv"保存种群的染色体矩阵，i为0,1,2,3...；
        "ObjV.csv"保存种群的目标函数矩阵；
        "FitnV.csv"保存种群个体的适应度列向量；
        "CV.csv"保存种群个体的违反约束程度矩阵；
        "Phen.csv"保存种群染色体表现型矩阵；
        注意：该函数不会对种群的合法性进行检查。

        """

        if os.path.exists('Result') == False:
            os.makedirs('Result')
        for i in range(self.ChromNum):
            with open('Result/Encodings' + str(i) + '.txt', 'w') as file:
                file.write(str(self.Encodings[i]))
                file.close()
            np.savetxt('Result/Fields' + str(i) + '.csv',
                       self.Fields[i], delimiter=',')
            np.savetxt('Result/Chroms' + str(i) + '.csv',
                       self.Chroms[i], delimiter=',')
        if self.ObjV is not None:
            np.savetxt('Result/ObjV.csv', self.ObjV, delimiter=',')
        if self.FitnV is not None:
            np.savetxt('Result/FitnV.csv', self.FitnV, delimiter=',')
        if self.CV is not None:
            np.savetxt('Result/CV.csv', self.CV, delimiter=',')
        if self.Phen is not None:
            np.savetxt('Result/Phen.csv', self.Phen, delimiter=',')
        print('种群信息导出完毕。')


class Algorithm:
    """
Algorithm : class - 算法模板顶级父类

描述:
    算法设置类是用来存储与算法运行参数设置相关信息的一个类。

属性:
    name            : str      - 算法名称（可以自由设置名称）。

    problem         : class <Problem> - 问题类的对象。

    MAXGEN          : int      - 最大进化代数。

    currentGen      : int      - 当前进化的代数。

    MAXTIME         : float    - 时间限制（单位：秒）。

    timeSlot        : float    - 时间戳（单位：秒）。

    passTime        : float    - 已用时间（单位：秒）。

    MAXEVALS        : int      - 最大评价次数。

    evalsNum        : int      - 当前评价次数。

    MAXSIZE         : int      - 最优解的最大数目。

    population      : class <Population> - 种群对象。

    drawing         : int      - 绘图方式的参数，
                                 0表示不绘图，
                                 1表示绘制结果图，
                                 2表示实时绘制目标空间动态图，
                                 3表示实时绘制决策空间动态图。

函数:
    terminated()    : 计算是否需要终止进化，具体功能需要在继承类即算法模板中实现。

    run()           : 执行函数，需要在继承类即算法模板中实现。

    check()         : 用于检查种群对象的ObjV和CV的数据是否有误。

    call_aimFunc()  : 用于调用问题类中的aimFunc()进行计算ObjV和CV(若有约束)。

"""

    def __init__(self):
        self.name = 'Algorithm'
        self.problem = None
        self.MAXGEN = None
        self.currentGen = None
        self.MAXTIME = None
        self.timeSlot = None
        self.passTime = None
        self.MAXEVALS = None
        self.evalsNum = None
        self.MAXSIZE = None
        self.population = None
        self.drawing = None

    def terminated(self):
        pass

    def run(self):
        pass

    def check(self, pop):
        """
        用于检查种群对象的ObjV和CV的数据是否有误。

        """

        # 检测数据非法值
        if np.any(np.isnan(pop.ObjV)):
            warnings.warn(
                "Warning: Some elements of ObjV are NAN, please check the calculation of ObjV.(ObjV的部分元素为NAN，请检查目标函数的计算。)",
                RuntimeWarning)
        elif np.any(np.isinf(pop.ObjV)):
            warnings.warn(
                "Warning: Some elements of ObjV are Inf, please check the calculation of ObjV.(ObjV的部分元素为Inf，请检查目标函数的计算。)",
                RuntimeWarning)
        if pop.CV is not None:
            if np.any(np.isnan(pop.CV)):
                warnings.warn(
                    "Warning: Some elements of CV are NAN, please check the calculation of CV.(CV的部分元素为NAN，请检查CV的计算。)",
                    RuntimeWarning)
            elif np.any(np.isinf(pop.CV)):
                warnings.warn(
                    "Warning: Some elements of CV are Inf, please check the calculation of CV.(CV的部分元素为Inf，请检查CV的计算。)",
                    RuntimeWarning)

    def call_aimFunc(self, pop):
        """
        使用注意:
        本函数调用的目标函数形如：aimFunc(pop), (在自定义问题类中实现)。
        其中pop为种群类的对象，代表一个种群，
        pop对象的Phen属性（即种群染色体的表现型）等价于种群所有个体的决策变量组成的矩阵，
        该函数根据该Phen计算得到种群所有个体的目标函数值组成的矩阵，并将其赋值给pop对象的ObjV属性。
        若有约束条件，则在计算违反约束程度矩阵CV后赋值给pop对象的CV属性（详见Geatpy数据结构）。
        该函数不返回任何的返回值，求得的目标函数值保存在种群对象的ObjV属性中，
                              违反约束程度矩阵保存在种群对象的CV属性中。
        例如：population为一个种群对象，则调用call_aimFunc(population)即可完成目标函数值的计算。
             之后可通过population.ObjV得到求得的目标函数值，population.CV得到违反约束程度矩阵。
        若不符合上述规范，则请修改算法模板或自定义新算法模板。

        """

        pop.Phen = pop.decoding()  # 染色体解码
        if self.problem is None:
            raise RuntimeError(
                'error: problem has not been initialized. (算法模板中的问题对象未被初始化。)')
        self.problem.aimFunc(pop)  # 调用问题类的aimFunc()
        self.evalsNum = self.evalsNum + \
            pop.sizes if self.evalsNum is not None else pop.sizes  # 更新评价次数
        if type(pop.ObjV) != np.ndarray or pop.ObjV.ndim != 2 or pop.ObjV.shape[0] != pop.sizes or pop.ObjV.shape[
                1] != self.problem.M:
            raise RuntimeError(
                'error: ObjV is illegal. (目标函数值矩阵ObjV的数据格式不合法，请检查目标函数的计算。)')
        if pop.CV is not None:
            if type(pop.CV) != np.ndarray or pop.CV.ndim != 2 or pop.CV.shape[0] != pop.sizes:
                raise RuntimeError(
                    'error: CV is illegal. (违反约束程度矩阵CV的数据格式不合法，请检查CV的计算。)')


class soea_SEGA_templet(ea.SoeaAlgorithm):
    """
soea_SEGA_templet : class - Strengthen Elitist GA templet(增强精英保留的遗传算法模板)

算法描述:
    本模板实现的是增强精英保留的遗传算法。算法流程如下：
    1) 根据编码规则初始化N个个体的种群。
    2) 若满足停止条件则停止，否则继续执行。
    3) 对当前种群进行统计分析，比如记录其最优个体、平均适应度等等。
    4) 独立地从当前种群中选取N个母体。
    5) 独立地对这N个母体进行交叉操作。
    6) 独立地对这N个交叉后的个体进行变异。
    7) 将父代种群和交叉变异得到的种群进行合并，得到规模为2N的种群。
    8) 从合并的种群中根据选择算法选择出N个个体，得到新一代种群。
    9) 回到第2步。
    该算法宜设置较大的交叉和变异概率，否则生成的新一代种群中会有越来越多的重复个体。

"""

    def __init__(self, problem, population):
        ea.SoeaAlgorithm.__init__(self, problem, population)  # 先调用父类构造方法
        if population.ChromNum != 1:
            raise RuntimeError('传入的种群对象必须是单染色体的种群类型。')
        self.name = 'SEGA'
        self.selFunc = 'tour'  # 锦标赛选择算子
        self.MAXGEN = problem.MAXGEN
        if population.Encoding == 'P':
            self.recOper = ea.Xovpmx(XOVR=0.7)  # 生成部分匹配交叉算子对象
            self.mutOper = ea.Mutinv(Pm=0.5)  # 生成逆转变异算子对象
        else:
            self.recOper = ea.Xovdp(XOVR=0.7)  # 生成两点交叉算子对象
            if population.Encoding == 'BG':
                # 生成二进制变异算子对象，Pm设置为None时，具体数值取变异算子中Pm的默认值
                self.mutOper = ea.Mutbin(Pm=None)
            elif population.Encoding == 'RI':
                self.mutOper = ea.Mutbga(
                    Pm=1 / self.problem.Dim, MutShrink=0.5, Gradient=20)  # 生成breeder GA变异算子对象
            else:
                raise RuntimeError('编码方式必须为''BG''、''RI''或''P''.')

    def run(self, individual, prophetPop=None):  # prophetPop为先知种群（即包含先验知识的种群）
        # ==========================初始化配置===========================
        population = self.population
        NIND = population.sizes
        self.initialization()  # 初始化算法模板的一些动态参数
        # ===========================准备进化============================
        # 使用输入来初始化
        population.warmup_Chrom(individual, NIND)
        self.call_aimFunc(population)  # 计算种群的目标函数值
        # 插入先验知识（注意：这里不会对先知种群prophetPop的合法性进行检查，故应确保prophetPop是一个种群类且拥有合法的Chrom、ObjV、Phen等属性）
        if prophetPop is not None:
            population = (prophetPop + population)[:NIND]  # 插入先知种群
        population.FitnV = ea.scaling(
            population.ObjV, population.CV, self.problem.maxormins)  # 计算适应度
        # ===========================开始进化============================
        while self.terminated(population) == False:
            # 选择
            offspring = population[ea.selecting(
                self.selFunc, population.FitnV, NIND)]
            # 进行进化操作
            # 交叉
            offspring.Chrom = self.recOper.do(offspring.Chrom)
            # 变异
            offspring.Chrom = self.mutOper.do(
                offspring.Encoding, offspring.Chrom, offspring.Field)
            # 计算目标函数值
            self.call_aimFunc(offspring)
            # 父子合并
            population = population + offspring
            # 计算适应度
            population.FitnV = ea.scaling(
                population.ObjV, population.CV, self.problem.maxormins)
            # 采用基于适应度排序的直接复制选择生成新一代种群
            population = population[ea.selecting(
                'dup', population.FitnV, NIND)]

        return self.finishing(population)  # 调用finishing完成后续工作并返回结果


class MoeaAlgorithm(Algorithm):  # 多目标优化算法模板父类

    """
    描述:
        此为多目标进化优化算法模板的父类，所有多目标优化算法模板均继承自该父类。
        为了使算法也能很好地求解约束优化问题，本算法模板稍作修改，增添“遗忘策略”，
        当某一代没有可行个体时，让进化记录器忽略这一代，不对这一代的个体进行记录，但不影响进化。

    """

    def __init__(self, problem, population):  # 构造方法，这里只初始化静态参数以及对动态参数进行定义
        Algorithm.__init__(self)  # 先调用父类构造方法
        self.problem = problem
        self.population = population
        self.drawing = 0  # 绘图
        self.ax = None  # 用于存储动态图
        self.forgetCount = None  # “遗忘策略”计数器，用于记录连续若干代出现种群所有个体都不是可行个体的次数
        self.maxForgetCount = None  # “遗忘策略”计数器最大上限值
        self.pop_trace = None  # 种群记录器

    def initialization(self):
        """
        描述: 该函数用于在进化前对算法模板的参数进行初始化操作。
        该函数需要在执行算法模板的run()方法的一开始被调用，同时开始计时，
        以确保所有这些参数能够被正确初始化。

        """
        self.ax = None  # 重置ax
        self.passTime = 0  # 初始化计时器
        self.forgetCount = 0  # 初始化“遗忘策略”计数器
        self.maxForgetCount = 100000  # 初始化“遗忘策略”计数器最大上限值，当超过这个上限时将终止进化
        self.pop_trace = []  # 初始化种群记录器
        self.currentGen = 0  # 设置初始为第0代
        self.evalsNum = 0  # 设置评价次数为0
        self.timeSlot = time.time()  # 开始计时

    def stat(self, population):  # 分析记录，更新进化记录器，population为传入的种群对象
        feasible = np.where(np.all(population.CV <= 0, 1))[0] if population.CV is not None else np.array(
            range(population.sizes))  # 找到可行解个体的下标
        if len(feasible) > 0:
            tempPop = population[feasible]  # 获取可行解个体
            self.pop_trace.append(tempPop)  # 添加记录（只添加可行解个体到种群记录器中）
            self.forgetCount = 0  # “遗忘策略”计数器清零
            self.passTime += time.time() - self.timeSlot  # 更新用时记录
            if self.drawing == 2:
                # 绘制目标空间动态图
                self.ax = ea.moeaplot(
                    tempPop.ObjV, 'objective values', False, self.ax, self.currentGen, gridFlag=True)
            elif self.drawing == 3:
                # 绘制决策空间动态图
                self.ax = ea.varplot(tempPop.Phen, 'decision variables', False, self.ax, self.currentGen,
                                     gridFlag=False)
            self.timeSlot = time.time()  # 更新时间戳
        else:
            self.currentGen -= 1  # 忽略这一代
            self.forgetCount += 1  # “遗忘策略”计数器加1

    def terminated(self, population):
        """
        描述:
            该函数用于判断是否应该终止进化，population为传入的种群。

        """

        self.check(population)  # 检查种群对象的关键属性是否有误
        self.stat(population)  # 进行统计分析，更新进化记录器
        # 判断是否终止进化，由于代数是从0数起，因此在比较currentGen和MAXGEN时需要对currentGen加1
        if self.currentGen + 1 >= self.MAXGEN or self.forgetCount >= self.maxForgetCount:
            return True
        else:
            self.currentGen += 1  # 进化代数+1
            return False

    def finishing(self, population):
        """
        进化完成后调用的函数。

        """

        # 得到非支配种群
        [levels, criLevel] = ea.ndsortDED(
            population.ObjV, None, 1, population.CV, self.problem.maxormins)  # 非支配分层
        NDSet = population[np.where(levels == 1)[0]]  # 只保留种群中的非支配个体，形成一个非支配种群
        if NDSet.CV is not None:  # CV不为None说明有设置约束条件
            NDSet = NDSet[np.where(np.all(NDSet.CV <= 0, 1))[0]]  # 最后要彻底排除非可行解
        self.passTime += time.time() - self.timeSlot  # 更新用时记录
        # # 绘图
        # if self.drawing != 0:
        #     if NDSet.ObjV.shape[1] == 2 or NDSet.ObjV.shape[1] == 3:
        #         ea.moeaplot(NDSet.ObjV, 'Pareto Front', saveFlag=True, gridFlag=True)
        #     else:
        #         ea.moeaplot(NDSet.ObjV, 'Value Path', saveFlag=True, gridFlag=False)
        # 返回帕累托最优集
        return NDSet


class SoeaAlgorithm(Algorithm):  # 单目标优化算法模板父类

    """
    描述:
        此为单目标进化优化算法模板的父类，所有单目标优化算法模板均继承自该父类。
        为了使算法也能很好地求解约束优化问题，本算法模板稍作修改，增添“遗忘策略”，
        当某一代没有可行个体时，让进化记录器忽略这一代，不对这一代的个体进行记录，但不影响进化。
    """

    def __init__(self, problem, population):  # 构造方法，这里只初始化静态参数以及对动态参数进行定义
        Algorithm.__init__(self)  # 先调用父类构造方法
        self.problem = problem
        self.population = population
        self.drawing = 0  # 绘图
        self.forgetCount = None  # “遗忘策略”计数器，用于记录连续若干代出现种群所有个体都不是可行个体的次数
        self.maxForgetCount = 100000  # “遗忘策略”计数器最大上限值，当超过这个上限时将终止进化
        self.trappedCount = 0  # “进化停滞”计数器
        # 进化算法陷入停滞的判断阈值，当abs(最优目标函数值-上代的目标函数值) < trappedValue时，对trappedCount加1
        self.trappedValue = 0
        self.maxTrappedCount = 10000000  # 进化停滞计数器最大上限值，当超过这个上限时将终止进化
        self.preObjV = np.nan  # “前代最优目标函数值记录器”，用于记录上一代的最优目标函数值
        self.ax = None  # 存储上一桢动画

    def initialization(self):
        """
        描述: 该函数用于在进化前对算法模板的一些动态参数进行初始化操作
        该函数需要在执行算法模板的run()方法的一开始被调用，同时开始计时，
        以确保所有这些参数能够被正确初始化。

        """

        self.ax = None  # 重置ax
        self.passTime = 0  # 记录用时
        self.forgetCount = 0  # “遗忘策略”计数器，用于记录连续若干代出现种群所有个体都不是可行个体的次数
        self.preObjV = np.nan  # 重置“前代最优目标函数值记录器”
        self.trappedCount = 0  # 重置“进化停滞”计数器
        self.obj_trace = np.zeros((self.MAXGEN, 2)) * \
            np.nan  # 定义目标函数值记录器，初始值为nan
        # 定义变量记录器，记录决策变量值，初始值为nan
        ''' 节省空间
        self.var_trace = np.zeros((self.MAXGEN, self.problem.Dim)) * np.nan
        '''
        self.currentGen = 0  # 设置初始为第0代
        self.evalsNum = 0  # 设置评价次数为0
        self.timeSlot = time.time()  # 开始计时

    def stat(self, population):  # 分析记录，更新进化记录器，population为传入的种群对象
        # 进行进化记录
        feasible = np.where(np.all(population.CV <= 0, 1))[0] if population.CV is not None else np.array(
            range(population.sizes))  # 找到可行解个体的下标
        if len(feasible) > 0:
            tempPop = population[feasible]
            bestIdx = np.argmax(tempPop.FitnV)  # 获取最优个体的下标
            self.obj_trace[self.currentGen, 0] = np.sum(
                tempPop.ObjV) / tempPop.sizes  # 记录种群个体平均目标函数值
            self.obj_trace[self.currentGen,
                           1] = tempPop.ObjV[bestIdx]  # 记录当代目标函数的最优值
            '''节省空间
            self.var_trace[self.currentGen,
                           :] = tempPop.Phen[bestIdx, :]  # 记录当代最优的决策变量值
            '''
            self.forgetCount = 0  # “遗忘策略”计数器清零
            if np.abs(self.preObjV - self.obj_trace[self.currentGen, 1]) < self.trappedValue:
                self.trappedCount += 1
            else:
                self.trappedCount = 0  # 重置进化停滞计数器
            self.passTime += time.time() - self.timeSlot  # 更新用时记录
            if self.drawing == 2:
                self.ax = ea.soeaplot(self.obj_trace[:, [1]], Label='Objective Value', saveFlag=False, ax=self.ax,
                                      gen=self.currentGen, gridFlag=False)  # 绘制动态图
            elif self.drawing == 3:
                self.ax = ea.varplot(tempPop.Phen, Label='decision variables', saveFlag=False, ax=self.ax,
                                     gen=self.currentGen, gridFlag=False)
            self.timeSlot = time.time()  # 更新时间戳
        else:
            self.currentGen -= 1  # 忽略这一代
            self.forgetCount += 1  # “遗忘策略”计数器加1

    def terminated(self, population):
        """
        描述:
            该函数用于判断是否应该终止进化，population为传入的种群。

        """

        self.check(population)  # 检查种群对象的关键属性是否有误
        self.stat(population)  # 分析记录当代种群的数据
        # 判断是否终止进化，由于代数是从0数起，因此在比较currentGen和MAXGEN时需要对currentGen加1
        if self.currentGen + 1 >= self.MAXGEN or self.forgetCount >= self.maxForgetCount or self.trappedCount >= self.maxTrappedCount:
            return True
        else:
            # 更新“前代最优目标函数值记录器”
            self.preObjV = self.obj_trace[self.currentGen, 1]
            self.currentGen += 1  # 进化代数+1
            return False

    def finishing(self, population):
        """
        进化完成后调用的函数。

        """

        # 处理进化记录器
        delIdx = np.where(np.isnan(self.obj_trace))[0]
        self.obj_trace = np.delete(self.obj_trace, delIdx, 0)
        self.var_trace = np.delete(self.var_trace, delIdx, 0)
        if self.obj_trace.shape[0] == 0:
            raise RuntimeError(
                'error: No feasible solution. (有效进化代数为0，没找到可行解。)')
        self.passTime += time.time() - self.timeSlot  # 更新用时记录
        # 绘图
        # if self.drawing != 0:
        #     ea.trcplot(self.obj_trace, [['种群个体平均目标函数值', '种群最优个体目标函数值']], xlabels=[['Number of Generation']],
        #                ylabels=[['Value']], gridFlags=[[False]])
        # 返回最后一代种群、进化记录器、变量记录器以及执行时间

        return [population, self.obj_trace, self.var_trace]

    def save_profit(self):
        # # 处理进化记录器
        # delIdx = np.where(np.isnan(self.obj_trace))[0]
        # self.obj_trace = np.delete(self.obj_trace, delIdx, 0)
        # self.var_trace = np.delete(self.var_trace, delIdx, 0)
        # if self.obj_trace.shape[0] == 0:
        #     raise RuntimeError('error: No feasible solution. (有效进化代数为0，没找到可行解。)')
        # self.passTime += time.time() - self.timeSlot  # 更新用时记录
        # # 绘图
        # if self.drawing != 0:
        #     ea.trcplot(self.obj_trace, [['种群个体平均目标函数值', '种群最优个体目标函数值']], xlabels=[['Number of Generation']],
        #                ylabels=[['Value']], gridFlags=[[False]])
        # 返回最后一代种群、进化记录器、变量记录器以及执行时间
        return self.obj_trace


class soea_psy_SEGA_templet(ea.SoeaAlgorithm):
    """
soea_psy_SEGA_templet : class - Polysomy Strengthen Elitist GA templet(增强精英保留的多染色体遗传算法模板)

模板说明:
    该模板是内置算法模板soea_SEGA_templet的多染色体版本，
    因此里面的种群对象为支持混合编码的多染色体种群类PsyPopulation类的对象。

算法描述:
    本模板实现的是增强精英保留的遗传算法。算法流程如下：
    1) 根据编码规则初始化N个个体的种群。
    2) 若满足停止条件则停止，否则继续执行。
    3) 对当前种群进行统计分析，比如记录其最优个体、平均适应度等等。
    4) 独立地从当前种群中选取N个母体。
    5) 独立地对这N个母体进行交叉操作。
    6) 独立地对这N个交叉后的个体进行变异。
    7) 将父代种群和交叉变异得到的种群进行合并，得到规模为2N的种群。
    8) 从合并的种群中根据选择算法选择出N个个体，得到新一代种群。
    9) 回到第2步。
    该算法宜设置较大的交叉和变异概率，否则生成的新一代种群中会有越来越多的重复个体。

"""

    def __init__(self, problem, population, XOVR):
        ea.SoeaAlgorithm.__init__(self, problem, population)  # 先调用父类构造方法
        if population.ChromNum == 1:
            raise RuntimeError('传入的种群对象必须是多染色体的种群类型。')
        self.name = 'psy-SEGA'
        self.selFunc = 'tour'  # 锦标赛选择算子
        self.MAXGEN = problem.MAXGEN
        # 由于有多个染色体，因此需要用多个重组和变异算子
        self.recOpers = []
        self.mutOpers = []
        for i in range(population.ChromNum):
            if population.Encodings[i] == 'P':
                recOper = ea.Xovpmx(XOVR=XOVR)  # 生成部分匹配交叉算子对象
                mutOper = ea.Mutinv(Pm=0.5)  # 生成逆转变异算子对象
            else:
                recOper = ea.Xovdp(XOVR=XOVR)  # 生成两点交叉算子对象
                if population.Encodings[i] == 'BG':
                    # 生成二进制变异算子对象，Pm设置为None时，具体数值取变异算子中Pm的默认值
                    mutOper = ea.Mutbin(Pm=None)
                elif population.Encodings[i] == 'RI':
                    # 生成breeder GA变异算子对象
                    mutOper = ea.Mutbga(
                        Pm=1 / self.problem.Dim, MutShrink=0.5, Gradient=20)
                else:
                    raise RuntimeError('编码方式必须为''BG''、''RI''或''P''.')
            self.recOpers.append(recOper)
            self.mutOpers.append(mutOper)

    def save_policy(self, population):
        bestIdx = np.argmax(population.ObjV)  # 获取最优个体的下标
        bestIdiv = population.Phen[bestIdx]
        raw_policy = bestIdiv

        ''' 标准化policy'''
        policy = np.zeros_like(raw_policy)

        action_num = len(self.problem.vars_dict.keys())
        raw_policy = raw_policy.reshape((action_num, -1), order='C')

        Idx = 0
        for d in range(self.problem.plant_periods):
            for i, tup in enumerate(self.problem.vars_dict.items()):
                dims = tup[1][2]
                multi = tup[1][1]
                policy[Idx:Idx+dims] = raw_policy[i,
                                                  d*dims: (d+1)*dims] * multi

                Idx += dims
        ''''''

        # # 更新最新的最优策略
        X_best_sofar_path = self.problem.X_best_sofar_path.replace(
            ".mat", "iter@%d.mat" % self.currentGen)
        scio.savemat(X_best_sofar_path,
                     {'policy': list(policy)},
                     do_compression=True)

    def run(self, individual, prophetPop=None):  # prophetPop为先知种群（即包含先验知识的种群）
        # ==========================初始化配置===========================
        population = self.population
        NIND = population.sizes
        self.initialization()  # 初始化算法模板的一些动态参数
        # ===========================准备进化============================
        # 使用输入来初始化
        population.warmup_Chroms(individual, NIND)
        self.call_aimFunc(population)  # 计算种群的目标函数值

        # 插入先验知识（注意：这里不会对先知种群prophetPop的合法性进行检查，故应确保prophetPop是一个种群类且拥有合法的Chrom、ObjV、Phen等属性）
        if prophetPop is not None:
            population = (prophetPop + population)[:NIND]  # 插入先知种群

        population.FitnV = ea.scaling(
            population.ObjV, population.CV, self.problem.maxormins)  # 计算适应度
        # ===========================开始进化============================
        while self.terminated(population) == False:
            # 选择
            offspring = population[ea.selecting(
                self.selFunc, population.FitnV, NIND)]
            # 进行进化操作，分别对各种编码的染色体进行重组和变异
            for i in range(population.ChromNum):
                offspring.Chroms[i] = self.recOpers[i].do(
                    offspring.Chroms[i])  # 重组
                offspring.Chroms[i] = self.mutOpers[i].do(offspring.Encodings[i], offspring.Chroms[i],
                                                          offspring.Fields[i])  # 变异
            self.call_aimFunc(offspring)  # 计算目标函数值
            population = population + offspring  # 父子合并
            population.FitnV = ea.scaling(
                population.ObjV, population.CV, self.problem.maxormins)  # 计算适应度
            # 得到新一代种群
            # 采用基于适应度排序的直接复制选择生成新一代种群
            population = population[ea.selecting(
                'dup', population.FitnV, NIND)]

            # 每隔一定迭代次数保存一次
            if self.currentGen % self.problem.save_interval == 0:
                self.save_policy(population)
        # return self.finishing(population)  # 调用finishing完成后续工作并返回结果
        return population  # 直接返回最后一代

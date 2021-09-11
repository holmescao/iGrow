import math
import os
import time
import numpy as np
import pandas as pd
import scipy.io as scio
import geatpy as ea
import warnings


class Problem:

    def __init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin, aimFunc=None, calReferObjV=None):
        self.name = name
        self.M = M
        self.maxormins = np.array(maxormins)
        self.Dim = Dim
        self.varTypes = np.array(varTypes)
        self.ranges = np.array([lb, ub])
        self.borders = np.array([lbin, ubin])
        self.aimFunc = aimFunc if aimFunc is not None else self.aimFunc

        self.calReferObjV = calReferObjV if calReferObjV is not None else self.calReferObjV

    def aimFunc(self, pop):
        raise RuntimeError(
            'error in Problem: aimFunc has not been initialized. ')

    def calReferObjV(self):
        return None

    def getReferObjV(self, reCalculate=False):
        """
        Description: This function is used to read/calculate the objective function reference value of the problem, which can be either the theoretical global optimal solution's objective function value or an artificially set non-optimal objective function reference value.
        reCalculate is a bool variable used to determine if calReferObjV() needs to be called to recalculate the objective function reference value.
        By default reCalculate is False, in which case it will first try to read the data of the theoretical global optimal solution.
        If it cannot be read, then calReferObjV() will be called to compute the theoretical global optimal solution.
        After calculating the theoretical global optimal solution
        After calculating the theoretical global optimal solution, the result will be saved to the referenceObjV folder with the file name "problem_target_dimension_number_of_decision_variables.csv".

        """

        if os.path.exists('referenceObjV') == False:
            os.makedirs('referenceObjV')
        if reCalculate == False:

            if os.path.exists('referenceObjV/' + self.name + '_M' + str(self.M) + '_D' + str(self.Dim) + '.csv'):
                return np.loadtxt('referenceObjV/' + self.name + '_M' + str(self.M) + '_D' + str(self.Dim) + '.csv',
                                  delimiter=',')

        referenceObjV = self.calReferObjV()
        if referenceObjV is not None:

            np.savetxt('referenceObjV/' + self.name + '_M' + str(self.M) + '_D' + str(self.Dim) + '.csv', referenceObjV,
                       delimiter=',')
        else:
            print('No data found for the reference value of the objective function')
        return referenceObjV


class Population:
    """
Population : class - population class

Description:
    The population class is a class used to store information about a population.

Properties:
    sizes : int - The size of the population, i.e. the number of individuals in the population.

    ChromNum : int - The number of chromosomes, i.e. how many chromosomes there are per individual.

    Encoding : str - the chromosome encoding method.
                       'BG':binary/Gray encoding.
                       'RI':real integer encoding, i.e. a mixture of real and integer encoding.
                       'P':Permutation encoding.
                       Related concept: The term 'real-valued encoding' includes both real integer encoding and permutation encoding.
                       Their common feature is that chromosomes can directly represent the corresponding decision variables without decoding.
                       The term "real integer" refers to a population chromosome that contains both real decimals and real integers.
                       Special usage.
                       If Encoding=None is set, the Field,Chrom member property of the population class will be set to None.
                       The population will not carry information directly related to the chromosome, which reduces unnecessary data storage.
                       This can be used when you only want to count information that is not directly related to chromosomes.
                       In particular, it can be used to evaluate the fitness of individuals in a uniform way during the evolutionary optimization of multiple populations.

    Field : array - The translation matrix, either FieldD or FieldDR (see Geatpy data structure for details).

    Chrom : array - population chromosome matrix, each row corresponds to one chromosome of an individual.

    Lind : int - The length of the population chromosome.

    ObjV : array - population objective function value matrix, each row corresponds to an individual's objective function value, each column corresponds to a target.

    FitnV : array - The vector of individual fitnesses of the population, each element corresponds to the fitness of one individual, and the minimum fitness is 0.

    CV : array - CV (Constraint Violation Value) is a matrix used to quantitatively describe the degree of constraint violation, each row corresponds to an individual and each column corresponds to a constraint.
                       Note: When no constraints are set, CV is set to None.

    Phen : array - the population expression matrix (i.e., the matrix of decision variables represented by each chromosome of the population after decoding).

Function :
    See the source code for details.

"""

    def __init__(self, Encoding, Field, NIND, Chrom=None, ObjV=None, FitnV=None, CV=None, Phen=None):
        """
        Description: Constructor for the population class, used to instantiate the population object, e.g.
             import geatpy as ea
             population = ea.Population(Encoding, Field, NIND), the
             NIND is the number of individuals needed.
             At this point the population is not really initialized, it is only the instantiation of the population object.
             This constructor must be passed in Chrom to complete the real initialization of the population.
             At first, you can pass only Encoding, Field and NIND to complete the instantiation of the population object.
             Other properties can be assigned later by calculation.

        """

        if type(NIND) is int and NIND >= 0:
            self.sizes = NIND
        else:
            raise RuntimeError(
                'error in Population: Size error.')
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
        Description: Initializes the population chromosome matrix with NIND as the desired number of individuals.
        NIND can be defaulted, if not, the population will be resized to NIND before initializing the chromosome matrix.

        """

        if NIND is not None:
            self.sizes = NIND
        self.Chrom = ea.crtpc(self.Encoding, self.sizes, self.Field)
        self.Lind = self.Chrom.shape[1]
        self.ObjV = None
        self.FitnV = None
        self.CV = None

    def decoding(self):
        """
        Description: Population chromosome decoding.

        """

        if self.Encoding == 'BG':

            Phen = self.bi2dec(self.Chrom, self.Field)
        elif self.Encoding == 'RI' or self.Encoding == 'P':
            Phen = self.Chrom.copy()
        else:
            raise RuntimeError(
                'error in Population.decoding: Encoding must be ''BG'' or ''RI'' or ''P''.')
        return Phen

    def copy(self):
        """
        copy : function - duplication of populations
        Usage:
            Suppose pop is a population matrix, then: pop1 = pop.copy() completes the copy of pop population.

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
        Description: Slicing of a population, i.e., selecting the corresponding individuals in the population according to the index subscript vector to form a new population.
        Usage: Suppose pop is a population matrix containing more than 2 individuals, then.
             pop1 = pop[[0,1]] to obtain the population consisting of the first and second individuals of the pop population.
        Note: index must be a slice or a row vector of type Numpy array or a list of type list.
             This function does not check the legality of the passed index parameter in detail.

        """

        if self.Encoding is None:
            NewChrom = None
        else:
            if self.Chrom is None:
                raise RuntimeError(
                    'error in Population: Chrom is None.')
            NewChrom = self.Chrom[index]

        if type(index) != slice and type(index) != np.ndarray and type(index) != list:
            raise RuntimeError(
                'error in Population: index must be a 1-D array.')
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
        shuffle : function - Shuffle the order of individuals in a population
        Usage: Assuming pop is a population matrix, then pop.shuffle() can be used to shuffle the order of individuals in the pop population.

        """

        shuff = np.argsort(np.random.rand(self.sizes))
        if self.Encoding is None:
            self.Chrom = None
        else:
            if self.Chrom is None:
                raise RuntimeError(
                    'error in Population: Chrom is None.')
            self.Chrom = self.Chrom[shuff, :]
        self.ObjV = self.ObjV[shuff, :] if self.ObjV is not None else None
        self.FitnV = self.FitnV[shuff] if self.FitnV is not None else None
        self.CV = self.CV[shuff, :] if self.CV is not None else None
        self.Phen = self.Phen[shuff, :] if self.Phen is not None else None

    def __setitem__(self, index, pop):
        """
        Description: Assignment of population individuals
        Usage: Suppose pop is a population matrix with more than 2 individuals, and pop1 is another population matrix with 2 individuals, then
             pop[[0,1]] = pop1 to assign the first and second individuals of pop population to the individuals of pop1 population.
        Note: index must be a row vector of type Numpy array, this function does not check the legitimacy of the index passed in.
             In addition, the function does not actively reset the fitness of the population after replacing individuals.
             If the fitness of all individuals needs to be re-evaluated due to individual replacement, the fitness of the population needs to be updated by handwritten code.

        """

        if self.Encoding is not None:
            if self.Encoding != pop.Encoding:
                raise RuntimeError(
                    'error in Population: Encoding disagree.')
            if np.all(self.Field == pop.Field) == False:
                raise RuntimeError(
                    'error in Population: Field disagree. ')
            if self.Chrom is None:
                raise RuntimeError(
                    'error in Population: Chrom is None. ')
            self.Chrom[index] = pop.Chrom
        if (self.ObjV is None) ^ (pop.ObjV is None):
            raise RuntimeError(
                'error in Population: ObjV disagree.')
        if (self.FitnV is None) ^ (pop.FitnV is None):
            raise RuntimeError(
                'error in Population: FitnV disagree. ')
        if (self.CV is None) ^ (pop.CV is None):
            raise RuntimeError(
                'error in Population: CV disagree. ')
        if (self.Phen is None) ^ (pop.Phen is None):
            raise RuntimeError(
                'error in Population: Phen disagree.')
        if self.ObjV is not None:
            self.ObjV[index] = pop.ObjV
        if self.FitnV is not None:
            self.FitnV[index] = pop.FitnV
        if self.CV is not None:
            self.CV[index] = pop.CV
        if self.Phen is not None:
            self.Phen[index] = pop.Phen
        self.sizes = self.Phen.shape[0]

    def __add__(self, pop):
        """
        Description: Merge populations of individuals
        Usage: Suppose pop1, pop2 are two populations, their number of individuals can be equal or unequal, then
             pop = pop1 + pop2 to merge the individuals of pop1 and pop2 populations.
        Note that.
            This function does not actively reset the fitness after performing a population merge.
            If you need to re-evaluate the fitness of all individuals due to population merging, you need to update the fitness of the population by handwriting the code.

        """

        if self.Encoding is None:
            NewChrom = None
        else:
            if self.Encoding != pop.Encoding:
                raise RuntimeError(
                    'error in Population: Encoding disagree. ')
            if self.Chrom is None or pop.Chrom is None:
                raise RuntimeError(
                    'error in Population: Chrom is None. ')
            if np.all(self.Field == pop.Field) == False:
                raise RuntimeError(
                    'error in Population: Field disagree.')
            NewChrom = np.vstack([self.Chrom, pop.Chrom])
        if (self.ObjV is None) ^ (pop.ObjV is None):
            raise RuntimeError(
                'error in Population: ObjV disagree.')
        if (self.CV is None) ^ (pop.CV is None):
            raise RuntimeError(
                'error in Population: CV disagree. ')
        if (self.Phen is None) ^ (pop.Phen is None):
            raise RuntimeError(
                'error in Population: Phen disagree.')
        NIND = self.sizes + pop.sizes
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
        Description: Calculate the population size
        Usage: Suppose pop is a population, then len(pop) gives the number of individuals in the population.
             In fact, the population size can also be obtained from pop.sizes.

        """

        return self.sizes

    def save(self):
        """
        Description: Saves the information about the population to a file.
        This function will save the information about the population in the "Result" folder, where.
        "Encoding.txt" holds the chromosome code of the population.
        "Field.csv" holds the decoding matrix of the population chromosomes.
        "Chrom.csv" holds the chromosome matrix of the population; "ObjV.csv" holds the chromosome matrix of the population.
        "ObjV.csv" holds the population's objective function matrix.
        "FitnV.csv" holds the fitness column vectors of the population individuals.
        "CV.csv" holds the matrix of constraint violations of the population individuals.
        "Phen.csv" holds the population chromosome phenotype matrix.
        Note: this function does not check the legality of the population.

        """
        return None

    def load_latest_population(self, population):
        """
        Description: Extracts the population information from the file and returns it to the population object.
        This function will save the population information in the "Result" folder, where.
        #"Encodingsi.txt" holds the chromosome code of the population, i is 0,1,2,3... i is 0,1,2,3...;.
        #"Fieldsi.csv" holds the decoding matrix of the population chromosomes, i is 0,1,2,3...; #"Fieldsi.csv" holds the decoding matrix of the population chromosomes, i is 0,1,2,3... ; # "Fieldsi.csv" holds the translation matrix of population chromosomes, i is 0,1,2,3...
        Chromsi.csv" holds the population chromosome matrix, i is 0,1,2,3...; # "Fieldsi.csv" holds the population chromosome matrix, i is 0,1,2,3... ;.
        "ObjV.csv" holds the matrix of the population's objective function.
        "FitnV.csv" holds the fitness column vectors of the population individuals.
        "CV.csv" holds the matrix of constraint violations of the population individuals.
        "Phen.csv" holds the population chromosome phenotype matrix.
        #'ChromNum': determined by Chroms
        'Linds': a list, eg: [32, 12], as determined by Chroms
        #'size': the size of the population, determined by the input parameters
        Note: This function does not check the legality of the population.

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

        return population

    def warmup_Chrom(self, individual, NIND):
        """
        Description: Initializes the population chromosome matrix with NIND as the desired number of individuals.
        NIND can be defaulted, if not, the population will be resized to NIND before initializing the chromosome matrix.

        """
        individual = np.array(individual)
        if individual.ndim == 1:
            Phen = np.expand_dims(individual, 0).repeat(
                NIND, axis=0)
        elif individual.ndim == 2 and individual.shape[0] == NIND:
            Phen = individual
        else:
            print("The number of individuals does not correspond to the set population size, please check the dimension of individual")
        self.sizes = NIND
        self.Chrom = self.dec2bi(Phen, self.Field)
        self.Lind = self.Chrom.shape[1]
        self.ObjV = None
        self.FitnV = None
        self.CV = None

    def bi2dec(self, Chrom, Field):
        lengthVars = np.cumsum(Field[0])
        posVars = np.concatenate(([0], lengthVars[:-1]))
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

                if total + Field[1, var] <= Field[2, var]:
                    phen_i[var] = total + Field[1, var]
                else:
                    phen_i[var] = Field[2, var]
            temporary[i] = phen_i

        temporary = temporary.astype(int)

        return temporary

    def dec2bi(self, Phen, Field):
        lengthVars = Field[0]
        posVars = np.concatenate(
            ([0], np.cumsum(lengthVars)[:-1]))
        pop_size = Phen.shape[0]
        var_num = Phen.shape[1]
        Chrom = np.zeros((pop_size, int(np.sum(lengthVars))))
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
PsyPopulation : class - Polychromosomal Population class (Popysomy Population)

Description:
    The class Popysomy Population is a class used to store information about populations that contain multiple chromosomes per individual.
    This class is similar to the population class Population, except that it can contain multiple chromosomes and therefore supports complex mixed coding.

Attributes:
    sizes : int - The population size, i.e. the number of individuals in the population.

    ChromNum : int - The number of chromosomes, i.e. how many chromosomes are present in each individual.

    Encodings : list - The list that stores the encoding method of each chromosome.

    Fields : list - A list of the corresponding decoding matrices for each chromosome.

    Chroms : list - stores the list of chromosome matrices for each population.

    Linds : list - List of chromosome lengths of the population.

    ObjV : array - Matrix of population objective function values, each row corresponds to an individual objective function value, each column corresponds to an objective.

    FitnV : array - The vector of individual fitnesses of the population, each element corresponds to the fitness of an individual, and the minimum fitness is 0.

    CV : array - CV (Constraint Violation Value) is a matrix used to quantitatively describe the degree of constraint violation, each row corresponds to an individual and each column corresponds to a constraint.
                        Note: When no constraints are set, CV is set to None.

    Phen : array - the population expression matrix (i.e., the matrix composed of the decision variables represented by the chromosomes after decoding).

Function :
    See the source code for details.

"""

    def __init__(self, Encodings, Fields, NIND, Chroms=None, ObjV=None, FitnV=None, CV=None, Phen=None):
        """
        Description: Constructor for the population class, used to instantiate the population object, e.g.
             import geatpy as ea
             population = ea.PsyPopulation(Encodings, Fields, NIND), the
             NIND is the number of individuals needed.
             At this point the population is not really initialized, it is only the instantiation of the population object.
             The constructor must be passed in Chroms to complete the real initialization of the population.
             At first, you can pass only Encodings, Fields and NIND to complete the instantiation of the population object.
             Other properties can be assigned later by calculation.

        """

        if type(NIND) is int and NIND >= 0:
            self.sizes = NIND
        else:
            raise RuntimeError(
                'error in PysPopulation: Size error.')
        self.ChromNum = len(Encodings)
        if self.ChromNum == 1:
            raise RuntimeError(
                'error in PysPopulation: ChromNum must be bigger than 1.')
        self.Encodings = Encodings
        self.Fields = Fields.copy()
        self.Chroms = [None] * self.ChromNum
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
        Description: Initializes the population chromosome matrix with NIND as the desired number of individuals.
        NIND can be defaulted, if not, the population will be resized to NIND before initializing the chromosome matrix.

        """

        if NIND is not None:
            self.sizes = NIND

        for i in range(self.ChromNum):
            self.Chroms[i] = ea.crtpc(
                self.Encodings[i], self.sizes, self.Fields[i])
            self.Linds.append(self.Chroms[i].shape[1])
        self.ObjV = None
        self.FitnV = np.ones((self.sizes, 1))
        self.CV = None

    def warmup_Chroms(self, individual, NIND):
        """
        Description: Initializes the population chromosome matrix with NIND as the desired number of individuals.
        NIND can be defaulted, if not, the population will be resized to NIND before initializing the chromosome matrix.

        """
        individual = np.array(individual)
        if individual.ndim == 1:
            Phen = np.expand_dims(individual, 0).repeat(
                NIND, axis=0)
        elif individual.ndim == 2 and individual.shape[0] == NIND:
            Phen = individual
        else:
            print("The number of individuals does not correspond to the set population size, please check the dimension of individual")

        idx = 0
        for i in range(self.ChromNum):
            self.Chroms[i] = self.dec2bi(
                Phen[:, idx: idx+self.Fields[i].shape[1]], self.Fields[i])
            self.Linds.append(self.Chroms[i].shape[1])

            idx += self.Fields[i].shape[1]

        self.sizes = NIND
        self.ObjV = None
        self.FitnV = None
        self.CV = None

    def bi2dec(self, Chrom, Field):
        lengthVars = np.cumsum(Field[0])
        posVars = np.concatenate(([0], lengthVars[:-1]))
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

                if total + Field[1, var] <= Field[2, var]:
                    phen_i[var] = total + Field[1, var]
                else:
                    phen_i[var] = Field[2, var]
            temporary[i] = phen_i

        temporary = temporary.astype(int)

        return temporary

    def dec2bi(self, Phen, Field):
        lengthVars = Field[0]
        posVars = np.concatenate(
            ([0], np.cumsum(lengthVars)[:-1]))
        pop_size = Phen.shape[0]
        var_num = Phen.shape[1]
        Chrom = np.zeros((pop_size, int(np.sum(lengthVars))))
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
        Description: Population chromosome decoding.

        """

        Phen = np.ones((self.sizes, 0))

        for i in range(self.ChromNum):
            if self.Encodings[i] == 'BG':
                tempPhen = self.bi2dec(self.Chroms[i], self.Fields[i])
                # tempPhen = ea.bs2ri(

            elif self.Encodings[i] == 'RI' or self.Encodings[i] == 'P':
                tempPhen = self.Chroms[i].copy()
            else:
                raise RuntimeError(
                    'error in PsyPopulation.decoding: Encoding must be ''BG'' or ''RI'' or ''P''.')
            Phen = np.hstack([Phen, tempPhen])

        return Phen

    def copy(self):
        """
        copy : function - duplication of populations
        Usage:
            Suppose pop is a population matrix, then: pop1 = pop.copy() completes the copy of pop population.

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
        Description: Slicing of a population, i.e., selecting the corresponding individuals in the population according to the index subscript vector to form a new population.
        Usage: Suppose pop is a population matrix containing more than 2 individuals, then.
             pop1 = pop[[0,1]] to obtain the population consisting of the 1st and 2nd individuals of the pop population.
        Note: The legality of index is not checked here.

        """

        NewChroms = []
        for i in range(self.ChromNum):
            if self.Chroms[i] is None:
                raise RuntimeError(
                    'error in PsyPopulation: Chrom[i] is None.')
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
        shuffle : function - Shuffle the order of individuals in a population
        Usage: Assuming pop is a population matrix, then pop.shuffle() can be used to shuffle the order of individuals in the pop population.

        """

        shuff = np.argsort(np.random.rand(self.sizes))
        for i in range(self.ChromNum):
            if self.Chroms[i] is None:
                raise RuntimeError(
                    'error in PsyPopulation: Chrom[i] is None. ')
            self.Chroms[i] = self.Chroms[i][shuff, :]
        self.ObjV = self.ObjV[shuff, :] if self.ObjV is not None else None
        self.FitnV = self.FitnV[shuff] if self.FitnV is not None else None
        self.CV = self.CV[shuff, :] if self.CV is not None else None
        self.Phen = self.Phen[shuff, :] if self.Phen is not None else None

    def __setitem__(self, index, pop):
        """
        Description: Assignment of population individuals
        Usage: Suppose pop is a population matrix with more than 2 individuals, and pop1 is another population matrix with 2 individuals, then
             pop[[0,1]] = pop1 to assign the first and second individuals of pop population to the individuals of pop1 population.
        Note: index must be a row vector of type Numpy array, this function does not check the legitimacy of the index passed in.
             In addition, the function does not actively reset the fitness of the population after replacing individuals.
             If the fitness of all individuals needs to be re-evaluated due to individual replacement, the fitness of the population needs to be updated by handwritten code.

        """

        for i in range(self.ChromNum):
            if self.Encodings[i] != pop.Encodings[i]:
                raise RuntimeError(
                    'error in PsyPopulation: Encoding disagree.')
            if np.all(self.Fields[i] == pop.Fields[i]) == False:
                raise RuntimeError(
                    'error in PsyPopulation: Field disagree. ')
            if self.Chroms[i] is None:
                raise RuntimeError(
                    'error in PsyPopulation: Chrom[i] is None.')
            self.Chroms[i][index] = pop.Chroms[i]
        if (self.ObjV is None) ^ (pop.ObjV is None):
            raise RuntimeError(
                'error in PsyPopulation: ObjV disagree. ')
        if (self.FitnV is None) ^ (pop.FitnV is None):
            raise RuntimeError(
                'error in PsyPopulation: FitnV disagree.')
        if (self.CV is None) ^ (pop.CV is None):
            raise RuntimeError(
                'error in PsyPopulation: CV disagree. ')
        if (self.Phen is None) ^ (pop.Phen is None):
            raise RuntimeError(
                'error in PsyPopulation: Phen disagree.')
        if self.ObjV is not None:
            self.ObjV[index] = pop.ObjV
        if self.FitnV is not None:
            self.FitnV[index] = pop.FitnV
        if self.CV is not None:
            self.CV[index] = pop.CV
        if self.Phen is not None:
            self.Phen[index] = pop.Phen
        self.sizes = self.Phen.shape[0]

    def __add__(self, pop):
        """
        Description: Merge populations of individuals
        Usage: Suppose pop1, pop2 are two populations, their number of individuals can be equal or unequal, then
             pop = pop1 + pop2 to merge the individuals of pop1 and pop2 populations.
        Note that.
            This function does not actively reset the fitness after performing a population merge.
            If you need to re-evaluate the fitness of all individuals due to population merging, you need to update the fitness of the population by handwriting the code.

        """

        NIND = self.sizes + pop.sizes
        NewChroms = self.Chroms
        for i in range(self.ChromNum):
            if self.Encodings[i] != pop.Encodings[i]:
                raise RuntimeError(
                    'error in PsyPopulation: Encoding disagree. ')
            if np.all(self.Fields[i] == pop.Fields[i]) == False:
                raise RuntimeError(
                    'error in PsyPopulation: Field disagree.')
            if self.Chroms[i] is None or pop.Chroms[i] is None:
                raise RuntimeError(
                    'error in PsyPopulation: Chrom is None.')
            NewChroms[i] = np.vstack([NewChroms[i], pop.Chroms[i]])
        if (self.ObjV is None) ^ (pop.ObjV is None):
            raise RuntimeError(
                'error in PsyPopulation: ObjV disagree.')
        if (self.CV is None) ^ (pop.CV is None):
            raise RuntimeError(
                'error in PsyPopulation: CV disagree. ')
        if (self.Phen is None) ^ (pop.Phen is None):
            raise RuntimeError(
                'error in PsyPopulation: Phen disagree.')
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
        Description: Calculate the population size
        Usage: Suppose pop is a population, then len(pop) gives the number of individuals in the population.
             In fact, the population size can also be obtained from pop.sizes.

        """

        return self.sizes

    def save(self):
        """
        Description: Saves the information about the population to a file.
        This function will save the information of the population in the "Result" folder, where.
        "Encodingsi.txt" holds the chromosome code of the population, i is 0,1,2,3... i is 0,1,2,3...
        "Fieldsi.csv" holds the decoding matrix of the population chromosomes, i is 0,1,2,3...; "Fieldsi.csv" holds the decoding matrix of the population chromosomes, i is 0,1,2,3... ;.
        "Chromsi.csv" holds the chromosome matrix of the population, i is 0,1,2,3... ;.
        "ObjV.csv" holds the matrix of the population's objective function.
        "FitnV.csv" holds the fitness column vectors of the population individuals.
        "CV.csv" holds the matrix of constraint violations of the population individuals.
        "Phen.csv" holds the population chromosome phenotype matrix.
        Note: this function does not check the legality of the population.

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


class Algorithm:
    """
Algorithm : class - Algorithm template top-level parent class

Description:
    The Algorithm Settings class is a class used to store information related to the setting of parameters for running the algorithm.

Attribute:
    name : str - the name of the algorithm (the name can be set freely).

    problem : class <Problem> - object of the problem class.

    MAXGEN : int - The maximum number of evolutionary generations.

    currentGen : int - The number of generations of the current evolution.

    MAXTIME : float - time limit (in seconds).

    timeSlot : float - Timestamp (in seconds).

    passTime : float - Used time (in seconds).

    MAXEVALS : int - The maximum number of evaluations.

    evalsNum : int - The current number of evaluations.

    MAXSIZE : int - The maximum number of optimal solutions.

    population : class <Population> - Population object.

    drawing : int - parameter for the drawing method.
                                 0 means no drawing.
                                 1 means draw the result, and
                                 2 means drawing the target space dynamics in real time, and
                                 3 means draw the decision space dynamics in real time.

Function:
    terminated() : calculate whether to terminate the evolution, the specific function needs to be implemented in the inherited class i.e. algorithm template.

    run() : execute function, need to be implemented in the inherited class, i.e. algorithm template.

    check() : Used to check if the data of ObjV and CV of the population object are wrong.

    call_aimFunc() : Used to call aimFunc() in the problem class to compute ObjV and CV (if there are constraints).

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
        Used to check the data of ObjV and CV of the population object for errors.

        """

        if np.any(np.isnan(pop.ObjV)):
            warnings.warn(
                "Warning: Some elements of ObjV are NAN, please check the calculation of ObjV.",
                RuntimeWarning)
        elif np.any(np.isinf(pop.ObjV)):
            warnings.warn(
                "Warning: Some elements of ObjV are Inf, please check the calculation of ObjV.",
                RuntimeWarning)
        if pop.CV is not None:
            if np.any(np.isnan(pop.CV)):
                warnings.warn(
                    "Warning: Some elements of CV are NAN, please check the calculation of CV.",
                    RuntimeWarning)
            elif np.any(np.isinf(pop.CV)):
                warnings.warn(
                    "Warning: Some elements of CV are Inf, please check the calculation of CV.",
                    RuntimeWarning)

    def call_aimFunc(self, pop):
        """
        Note on use:
        The target function called by this function is shaped like: aimFunc(pop), (implemented in the custom problem class).
        where pop is an object of the population class, representing a population.
        The Phen property of the pop object (i.e., the phenotype of the population chromosome) is equivalent to a matrix consisting of decision variables for all individuals of the population.
        The function computes the matrix consisting of the objective function values of all individuals of the population based on this Phen and assigns it to the ObjV attribute of the pop object.
        If there is a constraint, it is assigned to the CV attribute of the pop object after calculating the constraint violation matrix CV (see Geatpy data structure for details).
        The function does not return any return value, and the value of the objective function is stored in the ObjV property of the pop object.
                              The constraint violation matrix is stored in the CV attribute of the population object.
        For example: population is a population object, then call_aimFunc(population) to complete the calculation of the objective function value.
             After that, you can get the obtained objective function value by population.ObjV and the constraint violation degree matrix by population.CV.
        If the above specification is not met, please modify the algorithm template or customize a new one.

        """

        pop.Phen = pop.decoding()
        if self.problem is None:
            raise RuntimeError(
                'error: problem has not been initialized.')
        self.problem.aimFunc(pop)
        self.evalsNum = self.evalsNum + \
            pop.sizes if self.evalsNum is not None else pop.sizes
        if type(pop.ObjV) != np.ndarray or pop.ObjV.ndim != 2 or pop.ObjV.shape[0] != pop.sizes or pop.ObjV.shape[
                1] != self.problem.M:
            raise RuntimeError(
                'error: ObjV is illegal. ')
        if pop.CV is not None:
            if type(pop.CV) != np.ndarray or pop.CV.ndim != 2 or pop.CV.shape[0] != pop.sizes:
                raise RuntimeError(
                    'error: CV is illegal.')


class soea_SEGA_templet(ea.SoeaAlgorithm):
    """
soea_SEGA_templet : class - Strengthen Elitist GA templet (Enhanced Genetic Algorithm Template for Elite Retention)

Algorithm Description:
    This template implements the genetic algorithm for enhanced elite retention. The algorithm flow is as follows.
    1) Initialize a population of N individuals according to the coding rules.
    2) Stop if the stopping condition is satisfied, otherwise continue the execution.
    3) Statistical analysis is performed on the current population, such as recording its optimal individuals, average fitness, etc.
    4) Independently select N females from the current population.
    5) Independently perform crossover operations on these N females.
    6) Independently mutate these N crossed individuals.
    7) Merge the parent population and the crossover variant to obtain a population of size 2N.
    8) Select N individuals from the merged population according to the selection algorithm to obtain the new generation population.
    9) Return to step 2.
    It is advisable to set a large crossover and mutation probability for this algorithm, otherwise the new generation population generated will have more and more duplicate individuals.

"""

    def __init__(self, problem, population):
        ea.SoeaAlgorithm.__init__(self, problem, population)
        if population.ChromNum != 1:
            raise RuntimeError(
                'The incoming population object must be a single chromosome population type.')
        self.name = 'SEGA'
        self.selFunc = 'tour'
        self.MAXGEN = problem.MAXGEN
        if population.Encoding == 'P':
            self.recOper = ea.Xovpmx(XOVR=0.7)
            self.mutOper = ea.Mutinv(Pm=0.5)
        else:
            self.recOper = ea.Xovdp(XOVR=0.7)
            if population.Encoding == 'BG':

                self.mutOper = ea.Mutbin(Pm=None)
            elif population.Encoding == 'RI':
                self.mutOper = ea.Mutbga(
                    Pm=1 / self.problem.Dim, MutShrink=0.5, Gradient=20)
            else:
                raise RuntimeError(
                    ' The encoding method must be ''BG'', ''RI'' or ''P''...')

    def run(self, individual, prophetPop=None):
        # ==========================Initial Configuration===========================
        population = self.population
        NIND = population.sizes
        self.initialization()
        # ===========================Prepare============================

        population.warmup_Chrom(individual, NIND)
        self.call_aimFunc(population)

        if prophetPop is not None:
            population = (prophetPop + population)[:NIND]
        population.FitnV = ea.scaling(
            population.ObjV, population.CV, self.problem.maxormins)
        # ===========================Start to evolve============================
        while self.terminated(population) == False:

            offspring = population[ea.selecting(
                self.selFunc, population.FitnV, NIND)]

            offspring.Chrom = self.recOper.do(offspring.Chrom)

            offspring.Chrom = self.mutOper.do(
                offspring.Encoding, offspring.Chrom, offspring.Field)

            self.call_aimFunc(offspring)

            population = population + offspring

            population.FitnV = ea.scaling(
                population.ObjV, population.CV, self.problem.maxormins)

            population = population[ea.selecting(
                'dup', population.FitnV, NIND)]

        return self.finishing(population)


class MoeaAlgorithm(Algorithm):

    """
    Description:
        This is the parent class of the multi-objective evolutionary optimization algorithm template, from which all multi-objective optimization algorithm templates are inherited.
        In order to make the algorithm also good at solving constrained optimization problems, this algorithm template is slightly modified by adding a "forgetting strategy".
        When there are no feasible individuals in a generation, the evolutionary recorder ignores this generation and does not record the individuals in this generation, but does not affect the evolution.

    """

    def __init__(self, problem, population):
        Algorithm.__init__(self)
        self.problem = problem
        self.population = population
        self.drawing = 0
        self.ax = None
        self.forgetCount = None
        self.maxForgetCount = None
        self.pop_trace = None

    def initialization(self):
        """
        Description: This function is used to initialize the parameters of the algorithm template before its evolution.
        This function needs to be called at the beginning of the execution of the run() method of the algorithm template, while starting the timer, to ensure that all these parameters are initialized correctly.
        to ensure that all these parameters are initialized correctly.

        """
        self.ax = None
        self.passTime = 0
        self.forgetCount = 0
        self.maxForgetCount = 100000
        self.pop_trace = []
        self.currentGen = 0  # Set initial to generation 0
        self.evalsNum = 0
        self.timeSlot = time.time()

    def stat(self, population):
        feasible = np.where(np.all(population.CV <= 0, 1))[0] if population.CV is not None else np.array(
            range(population.sizes))
        if len(feasible) > 0:
            tempPop = population[feasible]
            self.pop_trace.append(tempPop)
            self.forgetCount = 0
            self.passTime += time.time() - self.timeSlot
            if self.drawing == 2:

                self.ax = ea.moeaplot(
                    tempPop.ObjV, 'objective values', False, self.ax, self.currentGen, gridFlag=True)
            elif self.drawing == 3:

                self.ax = ea.varplot(tempPop.Phen, 'decision variables', False, self.ax, self.currentGen,
                                     gridFlag=False)
            self.timeSlot = time.time()
        else:
            self.currentGen -= 1
            self.forgetCount += 1

    def terminated(self, population):
        """
        Description:
            This function is used to determine if evolution should be terminated, population is the incoming population.

        """

        self.check(population)
        self.stat(population)

        if self.currentGen + 1 >= self.MAXGEN or self.forgetCount >= self.maxForgetCount:
            return True
        else:
            self.currentGen += 1
            return False

    def finishing(self, population):
        """
        The function to call when the evolution is complete.

        """

        [levels, criLevel] = ea.ndsortDED(
            population.ObjV, None, 1, population.CV, self.problem.maxormins)
        NDSet = population[np.where(levels == 1)[0]]
        if NDSet.CV is not None:
            NDSet = NDSet[np.where(np.all(NDSet.CV <= 0, 1))[0]]
        self.passTime += time.time() - self.timeSlot

        # if self.drawing != 0:
        #     if NDSet.ObjV.shape[1] == 2 or NDSet.ObjV.shape[1] == 3:
        #         ea.moeaplot(NDSet.ObjV, 'Pareto Front', saveFlag=True, gridFlag=True)
        #     else:
        #         ea.moeaplot(NDSet.ObjV, 'Value Path', saveFlag=True, gridFlag=False)

        return NDSet


class SoeaAlgorithm(Algorithm):

    """
    Description:
        This is the parent class of the single-objective evolutionary optimization algorithm template, from which all single-objective optimization algorithm templates are inherited.
        In order to make the algorithm also good at solving constrained optimization problems, this algorithm template is slightly modified by adding a "forgetting strategy".
        When there are no feasible individuals in a generation, the evolutionary recorder ignores this generation and does not record the individuals in this generation, but does not affect the evolution.。
    """

    def __init__(self, problem, population):
        Algorithm.__init__(self)
        self.problem = problem
        self.population = population
        self.drawing = 0
        self.forgetCount = None
        self.maxForgetCount = 100000
        self.trappedCount = 0

        self.trappedValue = 0
        self.maxTrappedCount = 10000000
        self.preObjV = np.nan
        self.ax = None

    def initialization(self):
        """
        Description: This function is used to initialize some dynamic parameters of the algorithm template before its evolution.
        This function needs to be called at the beginning of the execution of the run() method of the algorithm template, while starting the timer, to ensure that all these parameters are initialized correctly.
        to ensure that all these parameters are initialized correctly.

        """

        self.ax = None
        self.passTime = 0
        self.forgetCount = 0
        self.preObjV = np.nan
        self.trappedCount = 0
        self.obj_trace = np.zeros((self.MAXGEN, 2)) * \
            np.nan  # Define an objective function value recorder with an initial value of nan
        # Define variable recorder to record decision variable values, initial value is nan
        ''' save space
        self.var_trace = np.zeros((self.MAXGEN, self.problem.Dim)) * np.nan
        '''
        self.currentGen = 0
        self.evalsNum = 0
        self.timeSlot = time.time()

    def stat(self, population):

        feasible = np.where(np.all(population.CV <= 0, 1))[0] if population.CV is not None else np.array(
            range(population.sizes))
        if len(feasible) > 0:
            tempPop = population[feasible]
            bestIdx = np.argmax(tempPop.FitnV)
            self.obj_trace[self.currentGen, 0] = np.sum(
                tempPop.ObjV) / tempPop.sizes
            self.obj_trace[self.currentGen,
                           1] = tempPop.ObjV[bestIdx]
            '''save space
            self.var_trace[self.currentGen,
                           :] = tempPop.Phen[bestIdx, :]   	
            '''
            self.forgetCount = 0
            if np.abs(self.preObjV - self.obj_trace[self.currentGen, 1]) < self.trappedValue:
                self.trappedCount += 1
            else:
                self.trappedCount = 0
            self.passTime += time.time() - self.timeSlot
            if self.drawing == 2:
                self.ax = ea.soeaplot(self.obj_trace[:, [1]], Label='Objective Value', saveFlag=False, ax=self.ax,
                                      gen=self.currentGen, gridFlag=False)
            elif self.drawing == 3:
                self.ax = ea.varplot(tempPop.Phen, Label='decision variables', saveFlag=False, ax=self.ax,
                                     gen=self.currentGen, gridFlag=False)
            self.timeSlot = time.time()
        else:
            self.currentGen -= 1
            self.forgetCount += 1

    def terminated(self, population):
        """
        Description:
            This function is used to determine if evolution should be terminated, population is the incoming population.

        """

        self.check(population)
        self.stat(population)

        if self.currentGen + 1 >= self.MAXGEN or self.forgetCount >= self.maxForgetCount or self.trappedCount >= self.maxTrappedCount:
            return True
        else:

            self.preObjV = self.obj_trace[self.currentGen, 1]
            self.currentGen += 1
            return False

    def finishing(self, population):
        """
        The function to call when the evolution is complete.

        """

        delIdx = np.where(np.isnan(self.obj_trace))[0]
        self.obj_trace = np.delete(self.obj_trace, delIdx, 0)
        self.var_trace = np.delete(self.var_trace, delIdx, 0)
        if self.obj_trace.shape[0] == 0:
            raise RuntimeError(
                'error: No feasible solution.')
        self.passTime += time.time() - self.timeSlot

        return [population, self.obj_trace, self.var_trace]

    def save_profit(self):

        # delIdx = np.where(np.isnan(self.obj_trace))[0]
        # self.obj_trace = np.delete(self.obj_trace, delIdx, 0)
        # self.var_trace = np.delete(self.var_trace, delIdx, 0)
        # if self.obj_trace.shape[0] == 0:
        #     raise RuntimeError('error: No feasible solution.')

        return self.obj_trace


class soea_psy_SEGA_templet(ea.SoeaAlgorithm):
    """
soea_psy_SEGA_templet : class - Polysomy Strengthen Elitist GA templet (Enhanced Elitist Retained Multichromosome Genetic Algorithm Template)

Template description:
    This template is a polysomal version of the built-in algorithm template soea_SEGA_templet.
    Therefore, the population objects inside are objects of the PsyPopulation class, which supports mixed coding of multicromosomal populations.

Algorithm description:
    This template implements the genetic algorithm with enhanced elite retention. The algorithm flow is as follows.
    1) Initialize a population of N individuals according to the encoding rules.
    2) Stop if the stopping condition is satisfied, otherwise continue the execution.
    3) Statistical analysis is performed on the current population, such as recording its optimal individuals, average fitness, etc.
    4) Independently select N females from the current population.
    5) Independently perform crossover operations on these N females.
    6) Independently mutate these N crossed individuals.
    7) Merge the parent population and the crossover variant to obtain a population of size 2N.
    8) Select N individuals from the merged population according to the selection algorithm to obtain the new generation population.
    9) Return to step 2.
    It is advisable to set a large crossover and mutation probability for this algorithm, otherwise the new generation population generated will have more and more duplicate individuals.

"""

    def __init__(self, problem, population, XOVR):
        ea.SoeaAlgorithm.__init__(self, problem, population)
        if population.ChromNum == 1:
            raise RuntimeError('The incoming population object must be a multichromosomal population type.')
        self.name = 'psy-SEGA'
        self.selFunc = 'tour'
        self.MAXGEN = problem.MAXGEN

        self.recOpers = []
        self.mutOpers = []
        for i in range(population.ChromNum):
            if population.Encodings[i] == 'P':
                recOper = ea.Xovpmx(XOVR=XOVR)
                mutOper = ea.Mutinv(Pm=0.5)
            else:
                recOper = ea.Xovdp(XOVR=XOVR)
                if population.Encodings[i] == 'BG':

                    mutOper = ea.Mutbin(Pm=None)
                elif population.Encodings[i] == 'RI':

                    mutOper = ea.Mutbga(
                        Pm=1 / self.problem.Dim, MutShrink=0.5, Gradient=20)
                else:
                    raise RuntimeError('The encoding method must be ''BG'', ''RI'' or ''P''.')
            self.recOpers.append(recOper)
            self.mutOpers.append(mutOper)

    def save_policy(self, population):
        bestIdx = np.argmax(population.ObjV)
        bestIdiv = population.Phen[bestIdx]
        raw_policy = bestIdiv

        ''' Standardized policy'''
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

        X_best_sofar_path = self.problem.X_best_sofar_path.replace(
            ".mat", "iter@%d.mat" % self.currentGen)
        scio.savemat(X_best_sofar_path,
                     {'policy': list(policy)},
                     do_compression=True)

    def run(self, individual, prophetPop=None):
        # ==========================Initial Configuration===========================
        population = self.population
        NIND = population.sizes
        self.initialization()
        # ===========================Prepare============================

        population.warmup_Chroms(individual, NIND)
        self.call_aimFunc(population)

        if prophetPop is not None:
            population = (prophetPop + population)[:NIND]

        population.FitnV = ea.scaling(
            population.ObjV, population.CV, self.problem.maxormins)
        # ===========================Start to evolve============================
        while self.terminated(population) == False:

            offspring = population[ea.selecting(
                self.selFunc, population.FitnV, NIND)]

            for i in range(population.ChromNum):
                offspring.Chroms[i] = self.recOpers[i].do(
                    offspring.Chroms[i])
                offspring.Chroms[i] = self.mutOpers[i].do(offspring.Encodings[i], offspring.Chroms[i],
                                                          offspring.Fields[i])
            self.call_aimFunc(offspring)
            population = population + offspring
            population.FitnV = ea.scaling(
                population.ObjV, population.CV, self.problem.maxormins)

            population = population[ea.selecting(
                'dup', population.FitnV, NIND)]

            if self.currentGen % self.problem.save_interval == 0:
                self.save_policy(population)

        return population

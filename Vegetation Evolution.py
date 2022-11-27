#coding:UTF-8
'''
Created by Jun YU (yujun@ie.niigata-u.ac.jp) on November 22, 2022
benchmark function: 28 functions of the CEC2017 test suite (https://www3.ntu.edu.sg/home/epnsugan/index_files/CEC2017/CEC2017.htm)
reference paper: Jun Yu, "Vegetation Evolution: An Optimization Algorithm Inspired by the Life Cycle of Plants," 
                          International Journal of Computational Intelligence and Applications, vol. 21, no.2, Article No. 2250010
'''

# import packages
import os
import numpy as np
import matplotlib.pyplot as plt
from cec17_functions import cec17_test_func


POPULATION_SIZE = 10                                                  # the number of individuals (POPULATION_SIZE > 4)
DIMENSION_NUM = 50                                                    # the number of variables
LOWER_BOUNDARY = -100                                                 # the maximum value of the variable range
UPPER_BOUNDARY = 100                                                  # the minimum value of the variable range
REPETITION_NUM = 30                                                   # the number of independent runs
MAX_FITNESS_EVALUATION_NUM = 25000                                    # the maximum number of fitness evaluations
GC = 6                                                                # the maximum growth cycle of an individual
GR = 1                                                                # the maximum growth radius of an individual
MS = 2                                                                # the moving scale
SEED_NUM = 6                                                          # the number of generated seeds by each individual

Population = np.zeros((POPULATION_SIZE, DIMENSION_NUM))               # the coordinates of the individual (candidate solutions)
Population_fitenss = np.zeros(POPULATION_SIZE)                        # the fitness value of all individuals
current_lifespan = 0                                                  # the current growth cycle of an individual
Current_fitness_evaluations = 0                                       # the current number of fitness evaluations
Fun_num = 1                                                           # the serial number of benchmark function


# evaluate the fitness of the incoming individual
def Fitness_Evaluation(individual):
    global Fun_num, Current_fitness_evaluations
    f = [0]
    cec17_test_func(individual, f, DIMENSION_NUM, 1, Fun_num)
    Current_fitness_evaluations = Current_fitness_evaluations + 1
    return f[0]


# if the individual is out of the search range, map back to the search space
def CheckIndi(Indi):
    range_width = UPPER_BOUNDARY - LOWER_BOUNDARY
    for i in range(DIMENSION_NUM):
        if  Indi[i] > UPPER_BOUNDARY:
            n = int((Indi[i]  - UPPER_BOUNDARY) / range_width)
            mirrorRange = ( Indi[i]  - UPPER_BOUNDARY) - (n * range_width)
            Indi[i]  = UPPER_BOUNDARY - mirrorRange
        elif  Indi[i]  < LOWER_BOUNDARY:
            n = int((LOWER_BOUNDARY -  Indi[i] ) / range_width)
            mirrorRange = (LOWER_BOUNDARY - Indi[i] ) - (n * range_width)
            Indi[i]  = LOWER_BOUNDARY + mirrorRange
        else:
            pass 


# initialize the population randomly
def Initialization():
    global Population, Population_fitenss, current_lifespan
    for i in range(POPULATION_SIZE):
        for j in range(DIMENSION_NUM):
            Population[i][j] = np.random.uniform(LOWER_BOUNDARY, UPPER_BOUNDARY)      # randomly generate individuals with equal probability
        # calculate the fitness of the i-th individual
        Population_fitenss[i] = Fitness_Evaluation(Population[i])
    current_lifespan = 1


def Growth():
    global Population, Population_fitenss
    offspring = np.zeros((POPULATION_SIZE, DIMENSION_NUM))
    offspring_fitness = np.zeros(POPULATION_SIZE)

    for i in range(POPULATION_SIZE):
        # each individual performs local local to generate a new offspring
        for j in range(DIMENSION_NUM):
            offspring[i][j] = Population[i][j] + GR * (np.random.random() *2.0 - 1.0)
        CheckIndi(offspring[i])
        offspring_fitness[i] = Fitness_Evaluation(offspring[i])
        # replace the parent individual if the generated offspring individual is better
        if offspring_fitness[i] < Population_fitenss[i]:
            Population_fitenss[i] = offspring_fitness[i]
            Population[i] = offspring[i].copy()


def Maturity():
    global Population, Population_fitenss
    seed_individual = np.zeros((POPULATION_SIZE*SEED_NUM, DIMENSION_NUM))
    seed_individual_fitness = np.zeros(POPULATION_SIZE*SEED_NUM)

    for i in range(POPULATION_SIZE):
        # each individual generates multiple seeds
        for j in range(SEED_NUM):
            index1 = index2 = 0
            while index1 == i:
                index1 = np.random.randint(0, POPULATION_SIZE)
            while index2 == i or index2 == index1:
                index2 = np.random.randint(0, POPULATION_SIZE)
            seed_individual[i*SEED_NUM + j] = Population[i] + MS * (np.random.random() *2.0 - 1.0) * (Population[index1] - Population[index2])
            CheckIndi(seed_individual[i*SEED_NUM + j])
            seed_individual_fitness[i*SEED_NUM + j] = Fitness_Evaluation(seed_individual[i*SEED_NUM + j])
    # Select the top PS individuals from the current population and seeds to enter the next generation
    temp_individual = np.vstack((Population,seed_individual))
    temp_individual_fitness = np.hstack((Population_fitenss,seed_individual_fitness))
    tmp = list(map(list, zip(range(len(temp_individual_fitness)), temp_individual_fitness)))
    small = sorted(tmp, key = lambda x:x[1],reverse = False)
    for i in range(POPULATION_SIZE):
        key, _ = small[i]
        Population_fitenss[i] = temp_individual_fitness[key]
        Population[i] = temp_individual[key].copy()


# the implementation process of differential evolution
def VegetationEvolution():
    global current_lifespan, GC
    if current_lifespan < GC:
        Growth()
        current_lifespan += 1
    elif current_lifespan == GC:
        Maturity()
        current_lifespan = 0
    else:
        print("Error: Maximum generation period exceeded.")


def RunVEGE():
    global Current_fitness_evaluations, Fun_num, Population_fitenss
    All_Trial_Best = []                             # record the fitness of best individual in each generation for all independent runs
    All_Trial_Best_Evaluation_Num = []              
    Current_generation = 0
    for i in range(REPETITION_NUM):                 # run the algorithm independently multiple times
        Best_list = []                              # record the fitness of best individual in each generation in each independent run
        Best_Evaluation_Num_list = []                              # record the fitness of best individual in each generation in each independent run
        Current_fitness_evaluations = 0             # the current number of fitness evaluations is reset to 0
        Current_generation = 1
        np.random.seed(2022 + 88*i)                 # fix the seed of random number
        Initialization()                            # randomly initialize the population
        print("Trial {} of F{}: The best fitness of 1-th generation is: {}".format(i+1, Fun_num, min(Population_fitenss)))
        Best_list.append(min(Population_fitenss))
        Best_Evaluation_Num_list.append(Current_fitness_evaluations)
        while Current_fitness_evaluations < MAX_FITNESS_EVALUATION_NUM:
            VegetationEvolution()
            Current_generation = Current_generation + 1
            print("Trial {} of F{}: The best fitness of {}-th generation is: {}".format(i+1, Fun_num, Current_generation, min(Population_fitenss)))
            Best_list.append(min(Population_fitenss))
            Best_Evaluation_Num_list.append(Current_fitness_evaluations)
        All_Trial_Best.append(Best_list)
        All_Trial_Best_Evaluation_Num.append(Best_Evaluation_Num_list)

    # plot the average convergence curve of multiple trial runs
    # ave_fitness = np.average(All_Trial_Best, axis=0)
    # ave_fitness_num = np.average(All_Trial_Best_Evaluation_Num, axis=0)
    # plt.figure()
    # myfig = plt.gcf()
    # plt.plot(ave_fitness_num, ave_fitness, label='VEGE')
    # plt.xlabel("# of fitness evaluations")
    # plt.ylabel("best fitness")
    # plt.legend()
    # myfig.savefig('./VEGE_images/F{}_{}D.png'.format(Fun_num, DIMENSION_NUM))
    # plt.close()
    # # write the fitness of best individual found in multiple trials to a file
    # # (each row represents an independent run, and each column from left to right represents the best individual in each generation with the convergence of generations)
    np.savetxt('./VEGE_Data/F{}_{}D.csv'.format(Fun_num, DIMENSION_NUM), All_Trial_Best, delimiter=",")

def main():
    global Fun_num

    if os.path.exists( './VEGE_images' ) == False: # Automatically create a folder when the folder does not exist
        os.makedirs( './VEGE_images' )
    if os.path.exists( './VEGE_Data' ) == False:   # Automatically create a folder when the folder does not exist
        os.makedirs( './VEGE_Data' )
        
    # run the 29 functions of the cec2017 test set in turn
    for i in range(3, 31):
        Fun_num = i
        RunVEGE()


if __name__=="__main__":
    main()
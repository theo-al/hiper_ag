import numpy as np
from   numpy.typing import NDArray

import math
from typing import Callable, Optional

from geneticalgorithm2 import geneticalgorithm2 as ga
from geneticalgorithm2 import Callbacks, Population_initializer

import matplotlib.pyplot as plt


## limites do espaço de busca para as variáveis
num_vars = 2
var_bound = np.array([[-5.12, 5.12]]*num_vars)

## parâmetros do AG
num_generations      = 20
population_size      = 100
mutation_probability = .001

elite_ratio = .1 # percentual de individuos preservados na próxima geracao
                 # com zero o algoritmo genético implementa um algoritmo genético 
                 # padrão em vez do GA elitista;
            
crossover_probability = .6
parents_portion = .1 # zero significa que toda a população é 
                     # preenchida com as soluções recém-geradas
                     # não aceita valores que resultem em menos de
                     # 2 indivíduos na próxima geração
                    
crossover_type = 'one_point'
selection_type = 'roulette'
mutation_type  = 'uniform_by_center' # acho que é pra deixar fixo

num_experiments = 5 # número de rodadas sucessivas
                   
algo_params = {
    'max_num_iteration': num_generations,
    'population_size':   population_size,
    'elit_ratio':        elite_ratio,

    'mutation_probability': mutation_probability,
    'mutation_type':        mutation_type,

    'crossover_probability': crossover_probability,
    'parents_portion':       parents_portion,
    'crossover_type':        crossover_type,

    'selection_type': selection_type,

    'max_iteration_without_improv': None,
}

  
## Função do problema
def Rastrigin(X: NDArray):
    OF = 0
    for i in range(len(X)):
        OF += (X[i]**2)-10*math.cos(2*math.pi*X[i])+10
    return OF 

# def Rastringin(X: NDArray):
#     return sum(map(lambda x: (x**2) - 10*math.cos(2*math.pi*x) + 10, X))  

## simulações
def make_model(algo_params: dict,
               var_bound: Optional[NDArray]=None,
               var_type:  Optional[str | list[str]]=None,
               var_num:   Optional[int]=None):

    if   var_type == 'bool': var_bound = None
    elif var_bound is not None:
        var_type = var_type if var_type is not None else \
                   ['real' if np.issubdtype(bound.dtype, np.floating) else \
                    'int'  if np.issubdtype(bound.dtype, np.integer)  else 'bool'
                           for bound in var_bound]
    else:
        raise ValueError("You need to specify both variable boundaries and variable types")

    if var_num is None:
        if var_bound is not None: var_num = len(var_bound)
        else: 
            raise ValueError("You need to specify both variable boundaries and number")

    model = ga(variable_boundaries=var_bound,
               variable_type=var_type,
               dimension=var_num,
               algorithm_parameters=algo_params)

    return model


def simulate(func: Callable[[NDArray], float], num_experiments: int, model: ga, *,
             seed: int=42, plot: bool=False,
             save_path="graficos/", save_prefix=""):
    simulations = []
    for simu in range(num_experiments):
        print('-------------------------------------------------------------------')
        print(f'Experimento número {simu}:')

        solution = model.run(function=func,
                             no_plot=True,
                             start_generation={'variables': None, 'scores': None},
                             studEA=True,
                             revolution_part=0,
                             remove_duplicates_generation_step=2,
                             population_initializer=Population_initializer(select_best_of=1,
                                                                           local_optimization_step='never',
                                                                           local_optimizer=None),
                             callbacks=[Callbacks.SavePopulation('callback_pop_example',
                                                                  save_gen_step=1,
                                                                  file_prefix='constraints'),
                                        Callbacks.PlotOptimizationProcess('callback_plot_example',
                                                                           save_gen_step=300,
                                                                           show=False,
                                                                           main_color='red',
                                                                           file_prefix='plot')],                            
                             middle_callbacks=[],

                             apply_function_to_parents=False, 
                             mutation_indexes=None,
                             revolution_after_stagnation_step=None,
                             stop_when_reached=None,
                             time_limit_secs=None, 
                             save_last_generation_as=None,
                             seed=seed) 

        convergence = model.report

        if plot:
            model.plot_results(title=f"Busca do ótimo para função {func.__name__}, experimento {simu}",
                               save_as=f"{save_path}{save_prefix}experimento{simu}convergencia.png", main_color='green')
            model.plot_generation_scores(title=f"Avaliações da última geração (nº {len(convergence)}) do experimento {simu}", 
                                         save_as=f"{save_path}{save_prefix}experimento{simu}solucao.png") #!

        print(f"Melhores indivíduos por geração: {convergence}")
        print()
        
        simulations.append(convergence)

        seed = hash(seed + 3)

    simulation_averages = []
    for i in range(num_generations):
        generation_sim_sum = 0
        for j in range(num_experiments):
            generation_sim_sum += simulations[j][i]
        simulation_averages.append(generation_sim_sum/num_experiments)

    return simulation_averages

model = make_model(algo_params, var_bound)
avgs  = simulate(Rastrigin, num_experiments, model)

print('------------------------------------------------------------------------')    
print('Valores médios dos melhores por Geração:')
print(avgs)

fig1, ax1 = plt.subplots()
ax1.set_title('Média dos Melhores por Geração')
ax1.boxplot(avgs)
plt.show()

plt.plot(avgs, label='Média dos Melhores por Geração')
plt.legend(loc='upper right')
plt.show()

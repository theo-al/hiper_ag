import numpy as np
from   numpy.typing import NDArray

import math

from geneticalgorithm2 import geneticalgorithm2 as ga
from geneticalgorithm2 import Callbacks
from geneticalgorithm2 import Population_initializer

import matplotlib.pyplot as plt


## Função do problema
def f(X: NDArray):
    OF = 0
    for i in range(len(X)):
        OF += (X[i]**2)-10*math.cos(2*math.pi*X[i])+10
    return OF 

# def _f(x: float): return (x**2)-10*math.cos(2*math.pi*x)+10
# def  f(X: NDArray):
#     return sum(map(_f, X))


## limites do espaço de busca para as variáveis
var_bound = np.array([[-5.12, 5.12]]*2)

## parâmetros do AG
num_generations      = 20  # número de gerações
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

model = ga(variable_boundaries=var_bound,
           variable_type='real',
           dimension=len(var_bound),
           algorithm_parameters=algo_params)
    
## simulações
simulations = []
for simu in range(num_experiments):
    print('-------------------------------------------------------------------')
    print(f'Experimento número {simu}:')

    model.run(function=f,
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
              init_creator=None,
              init_oppositors=None,
              duplicates_oppositor=None,
              revolution_oppositor=None,
              revolution_after_stagnation_step=None,
              stop_when_reached=None,
              time_limit_secs=None, 
              save_last_generation_as=None,
              seed=None)
 
    # title = f"Busca do ótimo para {type(f).__name__}"
    # model.plot_results(title=title, save_as=f"{title}.png", main_color='green')
    model.plot_generation_scores()
	
    convergence = model.report
    print(f"Melhores indivíduos por geração: {convergence}")
    print()
    
    simulations.append(convergence)

simulation_averages = [] # average?
for i in range(num_generations):
    generation_sim_sum = 0
    for j in range(num_experiments):
        generation_sim_sum += simulations[j][i]
    simulation_averages.append(generation_sim_sum/num_experiments)


print('------------------------------------------------------------------------')    
print('Valores médios dos melhores por Geração:')
print(simulation_averages)
print()

fig1, ax1 = plt.subplots()
ax1.set_title('Média dos Melhores por Geração')
ax1.boxplot(simulation_averages)
plt.show()

plt.plot(simulation_averages,
         label='Média dos Melhores por Geração')
plt.legend(loc='upper right')
plt.show()

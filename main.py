import numpy as np
from   numpy.typing import NDArray

import matplotlib.pyplot as plt

from hyper_eval import get_simulation_info
from typing     import NamedTuple, Iterable, Optional


## constantes
PLOT_DIR = 'graficos'
 

## função do problema
def Rastrigin(X: NDArray):
    return np.sum(np.square(X) - 10*np.cos(2*np.pi*X) + 10)


## limites do espaço de busca para as variáveis
num_vars = 2
var_bounds = np.array([[-5.12, 5.12]]*num_vars)


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
                   
default_params = {
    'elit_ratio': elite_ratio,

    'mutation_probability': mutation_probability,
    'mutation_type':        mutation_type,

    'crossover_probability': crossover_probability,
    'parents_portion':       parents_portion,
    'crossover_type':        crossover_type,

    'selection_type': selection_type,

    'max_num_iteration': num_generations,
    'population_size':   population_size,

    'max_iteration_without_improv': None,
}


## parâmetros da avaliação
num_experiments = 10

field_info = NamedTuple("field_info", [
    ("universe",        Iterable[float | int | str]),
    ("num_experiments", Optional[int]),
    ("format",          str),
])

hyperfields: dict[str, field_info] = {
    'elit_ratio':            field_info(np.linspace(.00, .50, num=4), None, '.2f'),
    'mutation_probability':  field_info(np.linspace(.01, .70, num=5), None, '.2f'),
    'crossover_probability': field_info(np.linspace(.01, .90, num=5), None, '.2f'),
    'parents_portion':       field_info(np.linspace(.01, .90, num=5), None, '.2f'),

    'mutation_type': field_info((
        'uniform_by_x',
        'uniform_by_center',
        'gauss_by_center',
        'gauss_by_x',
        #'uniform_discrete',
    ), None, ''),
    'crossover_type': field_info((
        'uniform',
        'one_point',
        'two_point',
        'segment',
        'shuffle',
    ), None, ''),
    'selection_type': field_info((
        'roulette',
        'ranking',
        'linear_ranking',
        'tournament',
        'sigma_scaling',
        'stochastic',
        'fully_random',
    ), None, ''),

    'max_iteration_without_improv': field_info((None, 1, 3, 7, 15,), None, ''),

    'max_num_iteration': field_info(np.arange(5, 105, 100//6), None, ''),
    'population_size':   field_info(np.arange(5, 155, 150//6), None, ''),
}


## roda a simulação e salva o resultado das parametrizações em arquivos na pasta {PLOT_DIR}
for param, (universe, num_exps, fmt) in hyperfields.items():
    num_exps = num_experiments if num_exps is None else num_exps

    plt.title(param, loc='center')
    params = default_params.copy()
    for sample in universe:
        print('--------------------------------------------------------')    
        print(f'parâmetro atual: {param}; amostra atual: {sample:{fmt}}')

        params[param] = sample
        try: avgs, curr_sz = get_simulation_info(params, var_bounds,
                                                 num_exps, func=Rastrigin)
        except (AssertionError, ZeroDivisionError) as e:
            print(f"{param}={sample}: {e}")
            continue

        print(f'Valores médios dos melhores por Geração:')
        print(avgs)
        
        # plota resultado do experimento
        plt.plot(avgs, label=f'={sample:{fmt}}')
        plt.legend(loc='upper right')

        # plota última geração
        last_color = plt.gca().lines[-1].get_color()
        plt.plot([curr_sz-1], [avgs[-1]],
                 marker='o', color=last_color)

    plt.savefig(f"./{PLOT_DIR}/{param}.png")
    plt.cla()

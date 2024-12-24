import numpy as np
from   numpy.typing import NDArray

import matplotlib.pyplot as plt

from hyper_eval import get_simulation_info
from typing     import NamedTuple, Iterable, Optional

from sys import argv

## constantes e tipos
PLOT_DIR = 'graficos'
 
class field_info(NamedTuple):
    universe: Iterable[float | int | str]
    num_exps: Optional[int] = None
    fmt:      str           = ''


## função do problema
def Rastrigin(X: NDArray):
    return np.sum(np.square(X) - 10*np.cos(2*np.pi*X) + 10)


## limites do espaço de busca para as variáveis
num_vars = 2
var_bounds = np.array([[-5.12, 5.12]]*num_vars)


## parâmetros "padrão" do AG
num_generations      = 20
population_size      = 100
mutation_probability = .001

elite_ratio = .1 # percentual de indivíduos preservados na próxima geração
                 # com zero o algoritmo genético implementa um algoritmo genético 
                 # padrão em vez do AG elitista;
            
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


## novos parâmetros
default_params.update({
    'crossover_probability': .46,
    'mutation_probability':  .34,

    'elit_ratio':      .09,
    'parents_portion': .23,

    'crossover_type': 'segment',
    'selection_type': 'tournament',
    'mutation_type':  'gauss_by_x',

    'max_num_iteration': 100,
    'population_size':   160,
    'max_iteration_without_improv': 25, #!
})


## parâmetros da avaliação
num_experiments = 10

hyperfields: dict[str, field_info] = {
    'elit_ratio':            field_info(np.linspace(.06, .14, num=7),  fmt='.2f'),
    'parents_portion':       field_info(np.linspace(.20, .50, num=8),  fmt='.2f'),
    'mutation_probability':  field_info(np.linspace(.25, .40, num=8),  fmt='.2f'),
    'crossover_probability': field_info(np.linspace(.24, .90, num=10), fmt='.2f'),

    'mutation_type': field_info((
        'uniform_by_x',
        'uniform_by_center',
        'gauss_by_center',
        'gauss_by_x',
        #'uniform_discrete',
    )),
    'selection_type': field_info((
        'roulette',
        'ranking',
        'linear_ranking',
        'tournament',
        'sigma_scaling',
        'stochastic',
        'fully_random',
    )),
    'crossover_type': field_info((
        'uniform',
        'one_point',
        'two_point',
        'segment',
        'shuffle',
    ), num_exps=num_experiments*2),

    'population_size':   field_info((20, 50, 80, 100, 130, 160)),
    'max_num_iteration': field_info((10, 50, 100)),

    'max_iteration_without_improv': field_info((None, 1, 3, 10, 20, 30),
                                               num_exps=num_experiments*3),
}

## roda a simulação e salva o resultado das parametrizações em arquivos na pasta {PLOT_DIR}
override = None if len(argv) <= 1 else argv[1]

for param, (universe, num_exps, fmt) in hyperfields.items():
    num_exps = num_experiments if num_exps is None else num_exps

    if override:
        if   param == override: num_exps *= 2
        elif param != "max_iteration_without_improvement" and \
             param != "max_num_iteration": continue

    # coloca o título e labels nos eixos do gráfico
    plt.title(param, loc='center')
    plt.ylabel("avaliação"), plt.xlabel("nº gerações")

    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(5)) # marca cada 5 gerações
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1)) # marca cada geração

    # passa por todas as possibilidades
    params = default_params.copy()
    for sample in universe:
        params[param] = sample

        # computa se puder
        try:
            print('--------------------------------------------------------')    
            print(f'parâmetro atual: {param}; amostra atual: {sample:{fmt}}')
            avgs, curr_sz = get_simulation_info(params, var_bounds,
                                                num_exps, func=Rastrigin)
            print(f'valores médios dos melhores por geração:')
            print(curr_sz) #!
            print(avgs[:curr_sz])

        except (AssertionError, ZeroDivisionError, ValueError) as e:
            print(f"erro: {param}={sample}: {str(e).strip()}")
            continue

        # plota resultado do experimento
        plt.plot(avgs, label=f'com {sample:{fmt}}')
        plt.legend(loc='upper right')

        # pega valores pros próximos plots
        last_color = plt.gca().lines[-1].get_color()
        last_x, last_y = curr_sz-1, avgs[-1]

        # plota valores da última geração
        plt.plot([last_x], [last_y], marker='o', color=last_color)
        plt.annotate(text=f"{last_y:.4f}", xy=(last_x, last_y),
                     xytext=(-10,15), textcoords='offset points', 
                     fontsize=11, color=last_color)

    # salva o plot
    plt.savefig(f"./{PLOT_DIR}/{param}.png")
    plt.cla()

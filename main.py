import numpy as np
from   numpy.typing import NDArray

import matplotlib.pyplot as plt

from hyper_eval import make_model, get_simulation_averages

 
## Função do problema
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
                   
hyperparams = {
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

## roda a simulação e mostra o resultado da parametrização
num_experiments = 10

model = make_model(hyperparams, var_bounds)
avgs  = get_simulation_averages(Rastrigin, num_experiments, model)#, silence=True)

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

import numpy as np
from   numpy.typing import NDArray

from typing import Callable, Optional

from geneticalgorithm2 import geneticalgorithm2 as ga
from geneticalgorithm2 import Callbacks, Population_initializer

from contextlib import redirect_stdout
 

## alias
isdtype = np.issubdtype

## funções para facilitar as simulações
def make_model(params: dict,
               var_bounds: Optional[NDArray]=None,
               var_types:  Optional[str | list[str]]=None,
               num_vars:   Optional[int]=None):

    if   var_types == 'bool': var_bounds = None
    elif var_bounds is not None:
        var_types = var_types if var_types is not None else \
                    ['real'   if isdtype(bound.dtype, np.floating) else \
                     'int'    if isdtype(bound.dtype, np.integer)  else 'bool'
                              for bound in var_bounds]
    else:
        raise ValueError("'var_type' and 'var_bound' may only be None if there if their values can be inferred from the rest")

    if num_vars is None:
        if var_bounds is not None: num_vars = len(var_bounds)
        else: 
            raise ValueError("'var_num' may only be None if 'var_bound' isn't")

    model = ga(variable_boundaries=var_bounds,
               variable_type=var_types,
               dimension=num_vars,
               algorithm_parameters=params)

    return model


def get_simulation_averages(func: Callable[[NDArray], float], num_experiments: int, model: ga, *,
                            seed: int=42, plot: bool=False, save_path="graficos/", save_prefix=""):
    num_generations = model.param.max_num_iteration

    simulations = np.zeros((num_experiments, num_generations))
    for i in range(num_experiments):
        print('-------------------------------------------------------------------')
        print(f'Experimento número {i}:')

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

        convergence = np.array(model.report)

        if plot:
            model.plot_results(title=f"Busca do ótimo para função {func.__name__}, experimento {i}",
                               save_as=f"{save_path}{save_prefix}experimento{i}convergencia.png", main_color='green')
            model.plot_generation_scores(title=f"Avaliações da última geração (nº {len(convergence)}) do experimento {i}", 
                                         save_as=f"{save_path}{save_prefix}experimento{i}solucao.png") #!

        print(f"Melhores indivíduos por geração: \n{convergence}")
        print()
        
        simulations[i] = convergence

        seed = hash(seed + 3)

    simulation_averages = np.zeros(num_generations)
    for i, gen in enumerate(simulations.transpose()):
        simulation_averages[i] = np.sum(gen)/num_experiments

    return simulation_averages
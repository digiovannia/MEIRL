import MEIRL

params_dict = {'sigsq_list': [[0.01]*4, [0.1]*4, [1]*4, [5]*4],
               'ETA_COEF': [0.01, 0.05, 0.5],
               'N': [20, 50, 100],
               'INTERCEPT_ETA': [-1]}
    
def experiment(id_num, params_dict, verbose):
    '''
    Computes results for the hyperparameter combination corresponding to id_num
    '''
    hyparams = {'D': 16,
            'MOVE_NOISE': 0.05,
            'INTERCEPT_ETA': 0,
            'WEIGHT': 2,
            'COEF': 0.1,
            'ETA_COEF': 0.01, # = omega in the report
            'GAM': 0.9,
            'M': 20,
            'N': 100,
            'J': 20,
            'T': 50,
            'Ti': 20,
            'B': 50,
            'INTERCEPT_REW': -1,
            'learn_rate': 0.5,
            'cr_reps': 10,
            'reps': 5,
            'sigsq_list': [0.1, 0.1, 0.1, 0.1]}

    seed = 20*((id_num - 1) // 8 + 1)
    if (id_num - 1) % 8 < 4:
        td = 'myo'
    else:
        td = 'boltz'
    pkey = id_num % 4
    parameter = sorted(params_dict.keys())[pkey]
    value_list = params_dict[parameter]
    MEIRL.results_var_hyper(id_num, parameter, value_list, seed, td,
      hyparams=hyparams, verbose=verbose)

'''
To create results files, I ran the equivalent of this script with 'ALL_80'
for the first input. Although each algorithm computes a solution in a few
seconds, evaluating all algorithms on all the hyperparameter combinations of
interest takes several hours and is best executed in parallel across multiple
machines/instances. The limiting step for these evaluations is the forward RL
step, that is, computing the optimal policy for each estimate of theta.
'''

if __name__ == '__main__':
    input_str = '''
    NOTE: The runtime of executing all 80 experiments is prohibitive; to fully
    replicate these results, it is recommended to run several instances of this
    script in parallel, with the id_nums split up accordingly.

    If you would like to run all 80 experiments, enter ALL_80 to confirm. Otherwise,
    you will be prompted for the range of id_nums you want the script to run on
    (inclusive):
    '''

    warn = input(input_str)
    if warn == 'ALL_80':
        start = 1
        end = 80
    else:
        start = int(input('Start id_num: '))
        end = int(input('End id_num: '))

    verbose = bool(input("Verbose output? ('Yes' maps to True, else False): "))

    for id_num in range(start, end + 1):
        experiment(id_num, params_dict, verbose)
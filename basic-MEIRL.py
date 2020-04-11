#%%
import aux_funcs
import timeit

'''
A simple DxD gridworld to test out multiple-experts IRL.
Agent can move in cardinal directions; moves in intended direction
with prob 1-MOVE_NOISE, else uniform direction. If moves in illegal
direction, stays in place.

Actions: 0 = up, 1 = right, 2 = down, 3 = left
'''





#%%
    
'''
To create results files, I did the equivalent of the following loop. Although
each algorithm computes a solution in a few seconds, evaluating all algorithms
on all the hyperparameter combinations of interest takes several hours and is
best executed in parallel across multiple machines/instances. The limiting
step for these evaluations is the forward RL step, that is, computing the
optimal policy for each estimate of theta.

params_dict = {'sigsq_list': [[0.01]*4, [0.1]*4, [1]*4, [5]*4],
               'ETA_COEF': [0.01, 0.05, 0.5],
               'N': [20, 50, 100],
               'INTERCEPT_ETA': [-1]}
id_num = 1
for i in range(10):
    seed = 20*(i+1)
    for parameter, value_list in params_dict.items():
        for td in ['myo', 'boltz']:
            results_var_hyper(id_num, parameter, value_list, seed, td,
              verbose=True)
            id_num += 1
'''

df = aux_funcs.summary()
aux_funcs.generate_figures(df)


# %%

setup = '''
import aux_funcs
import numpy as np

np.random.seed(20)

hyparams = aux_funcs.HYPARAMS

global D, MOVE_NOISE, INTERCEPT_ETA, WEIGHT, COEF
global ETA_COEF, GAM, M, N, J, T, Ti, B, INTERCEPT_REW, TP
(D, MOVE_NOISE, INTERCEPT_ETA, WEIGHT, COEF, ETA_COEF, GAM, M, N, J,
  T, Ti, B, INTERCEPT_REW, learn_rate, cr_reps, reps,
  sigsq_list) = (hyparams['D'], hyparams['MOVE_NOISE'],
  hyparams['INTERCEPT_ETA'], hyparams['WEIGHT'], hyparams['COEF'],
  hyparams['ETA_COEF'], hyparams['GAM'], hyparams['M'],
  hyparams['N'], hyparams['J'], hyparams['T'], hyparams['Ti'],
  hyparams['B'], hyparams['INTERCEPT_REW'], hyparams['learn_rate'],
  hyparams['cr_reps'], hyparams['reps'], hyparams['sigsq_list'])

pdict = {}

d = D // 2 + 1
state_space = np.array([(i,j) for i in range(D) for j in range(D)])
action_space = list(range(4))

global TP
TP = aux_funcs.transition(state_space, action_space, D, MOVE_NOISE)

alpha1 = np.array([WEIGHT, 0, 0, 0, 1])
alpha2 = np.array([0, 0, WEIGHT, 0, 1])
alpha3 = np.array([0, 0, 0, WEIGHT, 1]) 
alpha4 = np.array([0, WEIGHT, 0, 0, 1])

p = alpha1.shape[0]
m = 4

pdict['ex_alphas'] = np.stack([alpha1, alpha2, alpha3, alpha4]) 
pdict['ex_sigsqs'] = np.array(sigsq_list)

init_Q = np.random.rand(D, D, 4)

pdict['centers_x'] = np.random.choice(D, D//2)
pdict['centers_y'] = np.random.choice(D, D//2)

pdict['theta_true'] = 3*np.random.rand(D//2 + 1) - 2 
rewards = aux_funcs.lin_rew_func(pdict['theta_true'], state_space,
  pdict['centers_x'], pdict['centers_y'])

opt_policy, Q = aux_funcs.value_iter(state_space, action_space, rewards, TP,
  GAM, 1e-5)

pdict['phi'] = np.random.rand(m,2)
pdict['alpha'] = np.random.normal(size=(m,p), scale=0.05)
pdict['sigsq'] = np.random.rand(m)
pdict['beta'] = np.random.rand(m)
pdict['theta'] = np.random.normal(size=d)

traj_data = aux_funcs.make_data(pdict['ex_alphas'], pdict['ex_sigsqs'], rewards,
  N, Ti, state_space, action_space, TP, m)
boltz_data = aux_funcs.make_data(pdict['ex_alphas'], pdict['ex_sigsqs'], rewards,
  N, Ti, state_space, action_space, TP, m, Q)
rand_data = aux_funcs.random_data(pdict['ex_alphas'], pdict['ex_sigsqs'], rewards,
  N, Ti, state_space, action_space, TP, m)

dataset = traj_data
'''

code = '''
aux_funcs.MEIRL_det(pdict['theta'], pdict['alpha'], pdict['sigsq'], pdict['phi'],
  pdict['beta'], dataset, TP, state_space, action_space, B, m, M, Ti, N,
  learn_rate, reps, pdict['centers_x'], pdict['centers_y'])
'''

timeit.timeit(setup = setup, 
                    stmt = code, 
                    number = 10)

# %%

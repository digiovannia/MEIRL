#%%
import MEIRL
import timeit

'''
Generates .txt file of runtime of MEIRL_EM, MEIRL_det, and MEIRL_unif on seed
20 (since each algorithm executes a fixed number of iterations, the choice of
seed is arbitrary), over 100 repeats.
'''

setup = '''
import MEIRL
import numpy as np

np.random.seed(20)

hyparams = MEIRL.HYPARAMS

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
TP = MEIRL.transition(state_space, action_space, D, MOVE_NOISE)

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
rewards = MEIRL.lin_rew_func(pdict['theta_true'], state_space,
  pdict['centers_x'], pdict['centers_y'], COEF, INTERCEPT_REW, D)

opt_policy, Q = MEIRL.value_iter(state_space, action_space, rewards, TP,
  GAM, 1e-5, D)

pdict['phi'] = np.random.rand(m,2)
pdict['alpha'] = np.random.normal(size=(m,p), scale=0.05)
pdict['sigsq'] = np.random.rand(m)
pdict['beta'] = np.random.rand(m)
pdict['theta'] = np.random.normal(size=d)

traj_data = MEIRL.make_data(pdict['ex_alphas'], pdict['ex_sigsqs'], rewards,
  N, Ti, state_space, action_space, TP, m, MOVE_NOISE, ETA_COEF, D, INTERCEPT_ETA)
boltz_data = MEIRL.make_data(pdict['ex_alphas'], pdict['ex_sigsqs'], rewards,
  N, Ti, state_space, action_space, TP, m, MOVE_NOISE, ETA_COEF, D, INTERCEPT_ETA, Q)
rand_data = MEIRL.random_data(pdict['ex_alphas'], pdict['ex_sigsqs'], rewards,
  N, Ti, state_space, action_space, TP, m, MOVE_NOISE, D)

dataset = traj_data
'''

code_EM = '''
MEIRL.MEIRL_EM(pdict['theta'], pdict['alpha'], pdict['sigsq'], pdict['phi'],
  pdict['beta'], dataset, TP, state_space, action_space, B, m, M, Ti, N,
  learn_rate, reps, pdict['centers_x'], pdict['centers_y'],
  COEF, ETA_COEF, INTERCEPT_REW, INTERCEPT_ETA, D)
'''

code_det = '''
MEIRL.MEIRL_det(pdict['theta'], pdict['alpha'], pdict['sigsq'], pdict['phi'],
  pdict['beta'], dataset, TP, state_space, action_space, B, m, M, Ti, N,
  learn_rate, reps, pdict['centers_x'], pdict['centers_y'],
  COEF, ETA_COEF, INTERCEPT_REW, INTERCEPT_ETA, D)
'''

code_unif = '''
MEIRL.MEIRL_unif(pdict['theta'], pdict['alpha'], pdict['sigsq'], pdict['phi'],
  pdict['beta'], dataset, TP, state_space, action_space, B, m, M, Ti, N,
  learn_rate, reps, pdict['centers_x'], pdict['centers_y'],
  COEF, ETA_COEF, INTERCEPT_REW, INTERCEPT_ETA, D)
'''

EM_time = timeit.timeit(setup = setup, stmt = code_EM, number = 100)

det_time = timeit.timeit(setup = setup, stmt = code_det, number = 100)

unif_time = timeit.timeit(setup = setup, stmt = code_unif, number = 100)

f = open('runtimes.txt', 'w')
f.write('MEIRL-EM avg time over 100 runs = ' + str(EM_time / 100) + '\n')
f.write('MEIRL-Det avg time over 100 runs = ' + str(det_time / 100) + '\n')
f.write('MEIRL-Unif avg time over 100 runs = ' + str(unif_time / 100) + '\n')
f.close()
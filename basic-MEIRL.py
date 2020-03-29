import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os

'''
A simple DxD gridworld to test out multiple-experts IRL.
Agent can move in cardinal directions; moves in intended direction
with prob 1-MOVE_NOISE, else uniform direction. If moves in illegal
direction, stays in place.

Actions: 0 = up, 1 = right, 2 = down, 3 = left
'''

############################### Helper functions #############################

def log_mean_exp(tensor):
    '''
    Avoids overflow in computation of log of a mean of exponentials
    '''
    K = np.max(tensor, axis=2)
    expo = np.exp(tensor - K[:,:,None,:])
    return np.log(np.mean(expo,axis=2)) + K


def softmax(v, beta, axis=False):
    '''
    May delete if clause? no need for axis...
    '''
    if axis:
        x = beta[:,:,None]*v
        w = np.exp(x - np.max(x, axis=axis)[:,:,None])
        z = np.sum(w, axis=axis)
        return np.divide(w, z[:,:,None])
    else:
        w = np.exp(beta*v)
        z = np.sum(w)
        return w / z
    
########################### Grid world functions #############################

def manh_dist(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def act_to_coord(a):
    d = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0,-1)}
    return d[a]


def state_index(tup):
    return D*tup[0] + tup[1]


def multi_state_index(states):
    return D*states[:,0]+states[:,1]


def transition(state_space, action_space):
    '''
    Creates transition dynamics tensor based on step rules of the grid world.
    '''
    # 0 index = start state
    # 1 = action
    # 2 = next state
    S = len(state_space)
    A = len(action_space)
    TP = np.zeros((S, A, S))
    for s in range(S):
        x1 = state_space[s][0]
        y1 = state_space[s][1]
        for a in range(A):
            action = act_to_coord(a)
            for x2 in range(D):
                for y2 in range(D):
                    if ((x2 == x1 + action[0]) and
                      (y2 == y1 + action[1])):
                        TP[s,a,state_index((x2,y2))] += 1 - MOVE_NOISE
                    if manh_dist((x2, y2), state_space[s]) == 1:
                        TP[s,a,state_index((x2,y2))] += MOVE_NOISE/4
                    if ((x2, y2) == state_space[s]).all():
                        count = (x2 == 0 or x2 == D-1) + (y2 == 0 or y2 == D-1)
                        TP[s,a,state_index((x2,y2))] += count*MOVE_NOISE/4
                        if ((x1 + action[0] < 0) or
                          (x1 + action[0] >= D) or
                          (y1 + action[1] < 0) or
                          (y1 + action[1] >= D)):
                            TP[s,a,state_index((x2,y2))] += 1 - MOVE_NOISE
    return TP

################# Functions for generating trajectories ######################

def grid_step(s, a):
    '''
    Given current state s (in tuple form) and action a, returns resulting state.
    '''
    flip = np.random.rand()
    if flip < MOVE_NOISE:
        a = np.random.choice([0,1,2,3])
    new_state = s + act_to_coord(a)
    return np.minimum(np.maximum(new_state, 0), D-1)


def episode(s, T, policy, rewards, action_space, a=-1):
    '''
    Given any generic state, time horizon, policy, and reward structure indexed
    by the state, generates an episode starting from that state.
    '''
    states = [s]
    if a < 0:
        a = np.random.choice(action_space, p=policy[tuple(s)])
    actions = [a]
    reward_list = [0]
    for _ in range(T-1):
        s = grid_step(s,a)
        states.append(s)
        reward_list.append(rewards[tuple(s)])
        a = np.random.choice(action_space, p=policy[tuple(s)])
        actions.append(a)
    reward_list.append(rewards[tuple(s)])
    return np.array(states), actions, reward_list


def value_iter(state_space, action_space, rewards, TP, gam, tol):
    '''
    Action-value iteration for Q* and optimal policy (using known TPs)
    '''
    Q = np.random.rand(D**2, 4)
    delta = np.inf
    expect_rewards = TP.dot(np.ravel(rewards)) # S x A
    while delta > tol:
        delta = 0
        for s in range(len(state_space)):
            a = np.random.choice(action_space)
            #st = tuple(state_space[s])
            qval = Q[s,a]
            Q[s,a] = expect_rewards[s,a] + gam*(TP.dot(Q.max(axis=1)))[s,a]
            delta = np.max([delta, abs(qval - Q[s,a])])
    policy = np.zeros((D, D, 4))
    for i in range(D):
        for j in range(D):
            policy[i,j,np.argmax(Q[(D*i+j)])] = 1
    return policy, Q.reshape(D, D, 4)


def eta(st):
    '''
    Features for beta function
    '''
    return np.array([np.exp(-ETA_COEF*((st[0]-1)**2+(st[1]-1)**2)),
      np.exp(-ETA_COEF*((st[0]-(D-2))**2+(st[1]-(D-2))**2)),
      np.exp(-ETA_COEF*((st[0]-1)**2+(st[1]-(D-2))**2)),
      np.exp(-ETA_COEF*((st[0]-(D-2))**2+(st[1]-1)**2)),
      INTERCEPT_ETA])


def myo_policy(beta, s, TP, rewards):
    '''
    Policy for myopic expert (action probability proportional to next-step
    expected reward)
    '''
    expect_rew = TP[state_index(s)].dot(np.ravel(rewards))
    return softmax(expect_rew, beta)


def synthetic_traj(rewards, alpha, sigsq, i, Ti, state_space, action_space,
                   TP, Q=False):
    '''
    Generates trajectory based on myopic policy by default, or by Q-value-based
    policy if a Q array is supplied.
    '''
    s = state_space[np.random.choice(len(state_space))]
    states = [s]
    beta = np.dot(eta(s), alpha[i]) + np.random.normal(scale=np.sqrt(sigsq[i]))
    if type(Q) == np.ndarray:
        a = np.random.choice(action_space, p = softmax(Q[s[0],s[1]], beta))
    else:
        a = np.random.choice(action_space, p = myo_policy(beta, s, TP, rewards))
    actions = [a]
    for _ in range(Ti-1):
        s = grid_step(s,a)
        states.append(s)
        beta = np.dot(eta(s), alpha[i]) + np.random.normal(scale=np.sqrt(sigsq[i]))
        if type(Q) == np.ndarray:
            a = np.random.choice(action_space, p = softmax(Q[s[0],s[1]], beta))
        else:
            a = np.random.choice(action_space, p = myo_policy(beta, s, TP,
              rewards))
        actions.append(a)
    return list(multi_state_index(np.array(states))), actions


def make_data(alpha, sigsq, rewards, N, Ti, state_space, action_space,
              TP, m, Q=False):
    '''
    Makes a list of N trajectories based on the given parameters and the
    myopic policy by default, or the Q-value-based policy if Q is provided.
    '''
    trajectories = [[synthetic_traj(rewards, alpha, sigsq, i, Ti, state_space,
      action_space, TP, Q) for i in range(m)] for _ in range(N)]
    return trajectories 


def random_traj(rewards, alpha, sigsq, i, Ti, state_space, action_space,
                   TP, Q=False):
    '''
    '''
    s = state_space[np.random.choice(len(state_space))]
    states = [s]
    a = np.random.choice(action_space)
    actions = [a]
    for _ in range(Ti-1):
        s = grid_step(s,a)
        states.append(s)
        a = np.random.choice(action_space)
        actions.append(a)
    return list(multi_state_index(np.array(states))), actions


def random_data(alpha, sigsq, rewards, N, Ti, state_space, action_space,
              TP, m, Q=False):
    '''
    '''
    trajectories = [
        [random_traj(rewards, alpha, sigsq, i, Ti, state_space, action_space,
                       TP, Q) for i in range(m)]
        for _ in range(N)
    ]
    return trajectories 

####################### Functions for visualizing ############################

def visualize_policy(rewards, policy):
    '''
    Heatmap representing the input policy in the state space.
    '''
    pol = np.argmax(policy, axis=2)
    sns.heatmap(rewards)
    for x in range(D):
        for y in range(D):
            dx = (pol[x,y] % 2)*0.2*(2 - pol[x,y])
            dy = ((pol[x,y]-1) % 2)*0.2*(pol[x,y] - 1)
            plt.arrow(y+0.5,x+0.5,dx,dy,head_width=0.1,
                      color="black",alpha=0.9)


def mu_all(alpha):
    '''
    Visualizing mean of beta function for all states
    '''
    muvals = np.zeros((D,D))
    for i in range(D):
        for j in range(D):
            muvals[i,j] = np.dot(eta((i,j)), alpha)
    return muvals


def beta_func(alpha, sigsq):
    '''
    Computes a sample of betas for whole state space.
    '''
    noise = np.random.normal(loc=0,scale=np.sqrt(sigsq),
      size=(D,D))
    muvals = mu_all(alpha)
    return muvals + noise


def expect_reward_all(rewards, TP):
    '''
    Next-step reward equivalent of table of Q values
    '''
    grid = np.mgrid[0:4,0:(D**2)]
    return TP[grid[1],grid[0]].dot(np.ravel(rewards)).reshape(4,D,D)


def compare_myo_opt(rewards, TP, Q, save=False, hyp=False):
    '''
    Shows plot of best action with respect to next-step reward vs Q* 
    '''
    er = expect_reward_all(rewards, TP)
    sns.heatmap(np.argmax(er, axis=0))
    if hyp:
        res_str = 'hyp_results/'
    else:
        res_str = 'results/'
    if save:
        plt.savefig(res_str + save + '/' + 'best_myo.png')
    plt.show()
    sns.heatmap(np.argmax(Q, axis=2))
    if save:
        plt.savefig(res_str + save + '/' + 'best_Q.png')
    plt.show()
    
    
def see_trajectory(reward_map, state_seq, state_space):
    '''
    Input state seq is in index form; can transform to tuples
    '''
    state_seq = [state_space[s] for s in state_seq]
    for s in state_seq:
        sns.heatmap(reward_map)
        plt.annotate('*', (s[1]+0.2,s[0]+0.7), color='b', size=24)
        plt.show()

################################ Model functions #############################

def psi_all_states(state_space, centers_x, centers_y):
    '''
    Basis functions for the entire state space
    '''
    dist_x = state_space[:,0][:,None] - centers_x
    dist_y = state_space[:,1][:,None] - centers_y
    signs = np.array([1,-1] * (len(centers_x) // 2))
    bases = np.exp(-COEF*(dist_x**2 + dist_y**2))*signs[None,:] - 1/2
    inter = INTERCEPT_REW*np.ones(len(state_space))[:,None]
    return np.concatenate((bases, inter), axis=1)


def lin_rew_func(theta, state_space, centers_x, centers_y):
    return np.reshape(psi_all_states(state_space, centers_x,
      centers_y).dot(theta), (D, D))
    

def arr_expect_reward(rewards, data, TP, state_space):
    '''
    Expected next-step reward for an array of states and actions
    '''
    return TP[data[:,0], data[:,1]].dot(np.ravel(rewards))


def grad_lin_rew(data, state_space, centers_x, centers_y):
    '''
    Gradient of reward function for array of states and actions
    '''
    probs = TP[data[:,0], data[:,1]]
    return np.swapaxes(probs.dot(psi_all_states(state_space, centers_x,
      centers_y)), 1, 2)


def RE_all(reward_est, data, TP, state_space, m, centers_x, centers_y):
    '''
    Expected reward and beta function bases (eta) for an array of states and
    actions.
    '''
    # converting int representation of states to coordinates
    data_x = data[:,0,:] // D
    data_y = data[:,0,:] % D
    arr = np.array([np.exp(-ETA_COEF*((data_x-1)**2+(data_y-1)**2)),
      np.exp(-ETA_COEF*((data_x-(D-2))**2+(data_y-(D-2))**2)),
      np.exp(-ETA_COEF*((data_x-1)**2+(data_y-(D-2))**2)),
      np.exp(-ETA_COEF*((data_x-(D-2))**2+(data_y-1)**2)),
      INTERCEPT_ETA*np.ones(data[:,0,:].shape)])
    R_all = arr_expect_reward(reward_est, data, TP, state_space)
    E_all = np.swapaxes(arr, 0, 1)
    return R_all, E_all


def imp_samp_data(data, impa, j, m, Ti):
    '''
    Combines states from true data with actions from importance-sampling
    distributions into data array for log Z computation
    '''
    actions = impa[j]*np.ones((m,Ti))
    return np.swapaxes(np.stack((data[:,0,:], actions)), 0, 1)


def traj_TP(data, TP, Ti, m):
    '''
    Computes TPs for (s1, a1) to s2, ..., (st-1, at-1) to st in trajectories
    '''
    s2_thru_sTi = TP[data[:,0,:(Ti-1)],data[:,1,:(Ti-1)]]
    return s2_thru_sTi[np.arange(m)[:,None], np.arange(Ti-1), data[:,0,1:]]


def GD(phi, theta, alpha, sigsq, g_phi, g_theta, g_alpha, g_sigsq, learn_rate):
    '''
    Gradient descent with projection onto positive or non-negative orthant for
    the appropriate parameters
    '''
    phi = phi + learn_rate*g_phi
    phi[:,1] = np.maximum(phi[:,1], 0.01)
    theta = theta + learn_rate*g_theta
    alpha = np.maximum(alpha + learn_rate*g_alpha, 0)
    sigsq = np.maximum(sigsq + learn_rate*g_sigsq, 0.01)
    return phi, theta, alpha, sigsq


def y_t_nest(phi, phi_m, theta, theta_m, alpha, alpha_m, sigsq, sigsq_m, t):
    '''
    Making intermediate iterates for Nesterov acceleration
    '''
    const = (t-1)/(t+2)
    return (phi - const*(phi - phi_m),
            theta - const*(theta - theta_m),
            alpha - const*(alpha - alpha_m),
            sigsq - const*(sigsq - sigsq_m))

############################# Full AEVB models ###############################

def grad_terms(normals, phi, alpha, sigsq, theta, data, R_all, E_all, Ti, m):
    '''
    Computes quantities used in multiple computations of gradients.
    '''
    denom = sigsq + phi[:,1]
    sc_normals = (denom**(1/2))[:,None,None]*normals
    aE = np.einsum('ij,ijk->ik', alpha, E_all)
    mn = sigsq[:,None]*R_all + phi[:,0][:,None]*np.ones((m,Ti))
    meanvec = sc_normals + (aE + mn)[:,None,:]
    gvec = sc_normals + mn[:,None,:]
    gnorm = np.einsum('ijk,ijk->ij', gvec, gvec)
    return meanvec, denom, gvec, gnorm


def elbo(state_space, Ti, sigsq, gnorm, data, TP, m, normals, R_all,
             logZvec, meanvec, denom):
    '''
    Evidence lower bound
    '''
    lrho = -np.log(len(state_space))
    p1 = lrho - Ti/2*np.log(2*np.pi*sigsq)[:,None] - 1/(2*sigsq)[:,None]*gnorm
    logT = np.sum(np.log(traj_TP(data, TP, Ti, m)), axis=1)
    p2 = np.einsum('ijk,ik->ij', meanvec, R_all) - logZvec + logT[:,None]
    lp = p1 + p2    
    epsnorm = np.einsum('ijk,ijk->ij', normals, normals)
    lq = -Ti/2*np.log(2*np.pi*denom)[:,None] - epsnorm/2
    return (lp - lq).mean(axis=1).sum()


'''
In the gradient functions that follow, mean over an axis is for the Monte
Carlo estimate of the expectation with respect to multivariate standard normal.
'''

def phi_grad_ae(phi, m, Ti, normals, denom, sigsq, gZ_phi):
    x1 = (phi[:,0][:,None]*np.ones((m,Ti)))[:,None,:]
    x2 = (denom**(1/2))[:,None,None]*normals
    x = x1 + x2
    y1 = 1/sigsq[:,None]*x.sum(axis=2)
    y2 = np.einsum('ijk,ijk->ij', normals, x)/((2*sigsq*denom**(1/2))[:,None])
    result = -gZ_phi - np.stack((y1, y2 - Ti/(2*denom)[:,None]))
    return np.swapaxes(np.mean(result, axis=2), 0, 1)


def alpha_grad_ae(gZ_alpha, E_all, R_all):
    result = -gZ_alpha + np.einsum('ijk,ik->ij', E_all, R_all)[:,None,:]
    return np.mean(result, axis=1)


def sigsq_grad_ae(gZ_sigsq, normals, Ti, sigsq, gnorm, denom, R_all, gvec):
    q_grad = -Ti/(2*denom)
    x = -Ti/(2*sigsq) + np.einsum('ij,ij->i', R_all, R_all)
    y = np.einsum('ijk,ik->ij', normals, R_all)/(2*denom**(1/2))[:,None]
    z1 = R_all[:,None,:] + normals/(2*denom**(1/2))[:,None,None]
    z = 1/(sigsq[:,None])*np.einsum('ijk,ijk->ij', z1, gvec)
    w = 1/(2*sigsq**2)[:,None]*gnorm
    result = -gZ_sigsq + x[:,None] + y - z + w - q_grad[:,None]
    return np.mean(result, axis=1)


def theta_grad_ae(gZ_theta, data, state_space, R_all, E_all, sigsq, alpha,
               centers_x, centers_y):
    gradR = grad_lin_rew(data, state_space, centers_x, centers_y)
    X = sigsq[:,None]*R_all + np.einsum('ij,ijk->ik', alpha, E_all)
    result = -gZ_theta + np.einsum('ijk,ik->ij', gradR, X)[:,None,:]
    return np.sum(np.mean(result, axis=1), axis=0)


def logZ(sigsq, normals, meanvec, denom, impa, reward_est, data, M, TP, R_all,
         E_all, action_space, centers_x, centers_y, state_space, m, Ti):
    '''
    Estimates of normalization constant and its gradient via importance
    sampling
    '''
    # Expected reward and feature expectations for imp-sampling data
    R_Z = np.swapaxes(np.array([arr_expect_reward(reward_est,
      imp_samp_data(data, impa, j, m, Ti).astype(int), TP,
      state_space) for j in range(M)]), 0, 1)
    lst = []
    for j in range(M):
        newdata = imp_samp_data(data, impa, j, m, Ti).astype(int)
        feat_expect = grad_lin_rew(newdata, state_space, centers_x, centers_y)
        lst.append(feat_expect)
    gradR_Z = np.swapaxes(np.array(lst), 0, 1)

    # logZ
    bterm = np.einsum('ijk,ilk->ijlk', meanvec, R_Z)
    lvec = np.log(len(action_space)) + log_mean_exp(bterm)
    logZvec = lvec.sum(axis=2)

    # gradients of logZ
    gradR = grad_lin_rew(data, state_space, centers_x, centers_y)
    num1 = sigsq[:,None,None,None]*np.einsum('ijk,ilk->ijlk', R_Z, gradR)
    num2 = np.einsum('ijk,ilmk->ijlmk', meanvec, gradR_Z)
    expo = np.exp(bterm - np.max(bterm, axis=2)[:,:,None,:])
    num = expo[:,:,:,None,:]*(num1[:,None,:,:,:]+num2)
    den = expo.sum(axis=2)
    gZ_theta = (num.sum(axis=2)/den[:,:,None,:]).sum(axis=3)

    num_a = expo[:,:,:,None,:]*np.einsum('ijk,ilk->ijlk', R_Z,
      E_all)[:,None,:,:,:]
    gZ_alpha = (num_a.sum(axis=2)/den[:,:,None,:]).sum(axis=3)

    normterm = normals/((2*denom**2)[:,None,None])
    num_s = expo*np.einsum('ijk,ilk->iljk', R_Z, R_all[:,None,:] + normterm)
    gZ_sigsq = (num_s.sum(axis=2)/den).sum(axis=2)

    num_p1 = expo*R_Z[:,None,:,:]
    num_p2 = expo*np.einsum('ijk,ilk->iljk', R_Z, normterm)
    gZ_phi = np.array([(num_p1.sum(axis=2)/den).sum(axis=2),
      (num_p2.sum(axis=2)/den).sum(axis=2)])
    return logZvec, gZ_theta, gZ_alpha, gZ_sigsq, gZ_phi


def AR_AEVB(theta, alpha, sigsq, phi, beta, traj_data, TP, state_space,
         action_space, B, m, M, Ti, N, learn_rate, reps, centers_x, centers_y,
         plot=False):
    '''
    Autoencoder algorithm with adaptive restart and Nesterov acceleration
    '''
    impa = list(np.random.choice(action_space, M)) # uniformly sampled actions
    elbos = []
    phi_m = np.zeros_like(phi)
    theta_m = np.zeros_like(theta)
    alpha_m = np.zeros_like(alpha)
    sigsq_m = np.zeros_like(sigsq)
    y_phi = phi.copy()
    y_theta = theta.copy()
    y_alpha = alpha.copy()
    y_sigsq = sigsq.copy()
    # saving the best-performing parameters
    best = -np.inf
    best_phi = phi_m.copy()
    best_theta = theta_m.copy()
    best_alpha = alpha_m.copy()
    best_sigsq = sigsq_m.copy()
    tm = 1 # Nesterov acceleration hyperparameter
    last_lpd = -np.inf # saving last objective value for adaptive restart
    normals = np.random.multivariate_normal(np.zeros(Ti), np.eye(Ti), (m, B))
    for _ in range(reps):
        permut = list(np.random.permutation(range(N))) # shuffling data
        for n in permut:
            t = 1/2*(1 + np.sqrt(1 + 4*tm**2))
            
            data = np.array(traj_data[n])
            reward_est = lin_rew_func(y_theta, state_space, centers_x,
              centers_y)
            R_all, E_all = RE_all(reward_est, data, TP, state_space, m,
              centers_x, centers_y)
            meanvec, denom, gvec, gnorm = grad_terms(normals,
              y_phi, y_alpha, y_sigsq, y_theta, data, R_all, E_all, Ti, m)
            logZvec, gZ_theta, gZ_alpha, gZ_sigsq, gZ_phi = logZ(y_sigsq,
              normals, meanvec, denom, impa, reward_est, data, M, TP, R_all,
              E_all, action_space, centers_x, centers_y, state_space, m, Ti)
          
            logprobdiff = elbo(state_space, Ti, y_sigsq, gnorm, data, TP, m,
              normals, R_all, logZvec, meanvec, denom)
            elbos.append(logprobdiff)
            
            if logprobdiff > best:
                best = logprobdiff
                best_phi = y_phi.copy()
                best_theta = y_theta.copy()
                best_alpha = y_alpha.copy()
                best_sigsq = y_sigsq.copy()
              
            g_phi = phi_grad_ae(y_phi, m, Ti, normals, denom, y_sigsq, gZ_phi)
            g_theta = theta_grad_ae(gZ_theta, data, state_space, R_all, E_all,
              y_sigsq, y_alpha, centers_x, centers_y)
            g_alpha = alpha_grad_ae(gZ_alpha, E_all, R_all)
            g_sigsq = sigsq_grad_ae(gZ_sigsq, normals, Ti, y_sigsq, gnorm,
              denom, R_all, gvec)
            
            # gradient clipping
            g_phi = g_phi / np.linalg.norm(g_phi)
            g_theta = g_theta / np.linalg.norm(g_theta)
            g_alpha = g_alpha / np.linalg.norm(g_alpha, 'f')
            g_sigsq = g_sigsq / np.linalg.norm(g_sigsq)
          
            phi_m, theta_m, alpha_m, sigsq_m = phi, theta, alpha, sigsq
            phi, theta, alpha, sigsq = GD(y_phi, y_theta, y_alpha, y_sigsq,
              g_phi, g_theta, g_alpha, g_sigsq, learn_rate)
            
            mult = (tm - 1)/t
            y_phi = phi + mult*(phi - phi_m)
            y_phi[:,1] = np.maximum(y_phi[:,1], 0.01)
            y_theta = theta + mult*(theta - theta_m)
            y_alpha = np.maximum(alpha + mult*(alpha - alpha_m), 0)
            y_sigsq = np.maximum(sigsq + mult*(sigsq - sigsq_m), 0.01)
            
            learn_rate *= 0.99
            tm = t
            
            if logprobdiff < last_lpd:
                y_phi = phi.copy()
                y_theta = theta.copy()
                y_alpha = alpha.copy()
                y_sigsq = sigsq.copy()
                tm = 1
                
            last_lpd = logprobdiff
    if plot:
        plt.plot(elbos)
    return best_theta, best_phi, best_alpha, best_sigsq



############################ Uniform beta model ##############################

def beta_grad_unif(gZ_beta, R_all):
    return -gZ_beta + R_all.sum(axis=1)


def theta_grad_unif(data, beta, state_space, gZ_theta, centers_x, centers_y):
    gradR = grad_lin_rew(data, state_space, centers_x, centers_y)
    return -gZ_theta + (beta[:,None,None]*gradR).sum(axis=2).sum(axis=0)


def logZ_unif(beta, impa, reward_est, data, M, TP, state_space, action_space,
              m, Ti, centers_x, centers_y):
    R_Z = np.swapaxes(np.array([arr_expect_reward(reward_est,
      imp_samp_data(data, impa, j, m, Ti).astype(int),
      TP, state_space) for j in range(M)]), 0, 1)
    lst = []
    for j in range(M):
        newdata = imp_samp_data(data, impa, j, m, Ti).astype(int)
        feat_expect = grad_lin_rew(newdata, state_space, centers_x, centers_y)
        lst.append(feat_expect)
    gradR_Z = np.swapaxes(np.array(lst), 0, 1)
    
    expo = np.exp(beta[:,None,None]*R_Z)
    lvec = np.log(len(action_space)*np.mean(expo,axis=1))
    logZvec = lvec.sum(axis=1)

    num_t = expo[:,:,None,:]*beta[:,None,None,None]*gradR_Z
    num_b = expo*R_Z
    numsum_t = num_t.sum(axis=1)
    numsum_b = num_b.sum(axis=1)
    den = expo.sum(axis=1)
    gZ_theta = ((numsum_t/den[:,None,:]).sum(axis=2)).sum(axis=0)
    gZ_beta = (numsum_b/den).sum(axis=1)
    return logZvec, gZ_theta, gZ_beta


def GD_unif(theta, beta, g_theta, g_beta, learn_rate):
    theta = theta + learn_rate*g_theta
    beta = beta + learn_rate*g_beta
    return theta, beta


def MEIRL_unif(theta, alpha, sigsq, phi, beta, traj_data, TP, state_space,
         action_space, B, m, M, Ti, N, learn_rate, reps, centers_x, centers_y,
         plot=False):
    impa = list(np.random.choice(action_space, M))
    lik = []
    theta_m = np.zeros_like(theta)
    beta_m = np.zeros_like(beta)
    y_theta = theta.copy()
    y_beta = beta.copy()
    best = -np.inf
    best_theta = theta_m.copy()
    best_beta = beta_m.copy()
    tm = 1
    last_lik = -np.inf
    for _ in range(reps):
        permut = list(np.random.permutation(range(N)))
        for n in permut:
            t = 1/2*(1 + np.sqrt(1 + 4*tm**2))
            
            data = np.array(traj_data[n])
            reward_est = lin_rew_func(y_theta, state_space, centers_x,
              centers_y)
            R_all = RE_all(reward_est, data, TP, state_space, m,
              centers_x, centers_y)[0]     
            logZvec, gZ_theta, gZ_beta = logZ_unif(y_beta, impa, reward_est,
              data, M, TP, state_space, action_space, m, Ti, centers_x,
              centers_y)
          
            loglikelihood = loglik(state_space, Ti, y_beta, data, TP, m, R_all,
              logZvec, unif=True).sum()
            lik.append(loglikelihood)
            
            if loglikelihood > best:
                best = loglikelihood
                best_theta = y_theta.copy()
                best_beta = y_beta.copy()
              
            g_theta = theta_grad_unif(data, y_beta, state_space, gZ_theta,
              centers_x, centers_y)
            g_beta = beta_grad_unif(gZ_beta, R_all)
            
            g_theta = g_theta / np.linalg.norm(g_theta)
            g_beta = g_beta / np.linalg.norm(g_beta)
          
            theta_m, beta_m, = theta, beta
            theta = y_theta + learn_rate*g_theta
            beta = y_beta + learn_rate*g_beta
            
            mult = (tm - 1)/t
            y_theta = theta + mult*(theta - theta_m)
            y_beta = np.maximum(beta + mult*(beta - beta_m), 0)
            
            learn_rate *= 0.99
            tm = t
            
            if loglikelihood < last_lik:
                y_theta = theta.copy()
                y_beta = beta.copy()
                tm = 1
                
            last_lik = loglikelihood
    if plot:
        plt.plot(lik)
    return best_theta, best_beta


################### Deterministic state-varying beta model ###################

def alpha_grad_det(gZ_alpha, R_all, E_all):
    return -gZ_alpha + np.einsum('ijk,ik->ij', E_all, R_all)


def theta_grad_det(data, beta, state_space, gZ_theta, centers_x, centers_y):
    gradR = grad_lin_rew(data, state_space, centers_x, centers_y)
    return -gZ_theta + np.einsum('ij,ikj->k', beta, gradR)


def logZ_det(beta, impa, reward_est, data, M, TP, R_all, E_all, state_space,
             action_space, m, Ti, centers_x, centers_y):
    R_Z = np.swapaxes(np.array([arr_expect_reward(reward_est,
                      imp_samp_data(data, impa, j, m, Ti).astype(int),
                      TP, state_space) for j in range(M)]), 0, 1)
    lst = []
    for j in range(M):
        newdata = imp_samp_data(data, impa, j, m, Ti).astype(int)
        feat_expect = grad_lin_rew(newdata, state_space, centers_x, centers_y)
        lst.append(feat_expect)
    gradR_Z = np.swapaxes(np.array(lst), 0, 1)
    
    expo = np.exp(beta[:,None,:]*R_Z)
    lvec = np.log(len(action_space)*np.mean(expo,axis=1))
    logZvec = lvec.sum(axis=1)

    num_t = expo[:,:,None,:]*beta[:,None,None,:]*gradR_Z
    num_a = np.einsum('ijk,ilk->ilk', expo*R_Z, E_all)
    numsum_t = num_t.sum(axis=1)
    den = expo.sum(axis=1)
    gZ_theta = ((numsum_t/den[:,None,:]).sum(axis=2)).sum(axis=0)
    gZ_alpha = (num_a/den[:,None,:]).sum(axis=2)
    return logZvec, gZ_theta, gZ_alpha


def loglik(state_space, Ti, beta, data, TP, m, R_all, logZvec, unif=False):
    logT = np.log(1/len(state_space)) + np.sum(np.log(traj_TP(data, TP, Ti,
      m)), axis=1)
    if unif:
        return -logZvec + logT + beta*R_all.sum(axis=1)
    else:
        return -logZvec + logT + np.einsum('ij,ij->i', beta, R_all)


def MEIRL_det_pos(theta, alpha, sigsq, phi, beta, traj_data, TP, state_space,
         action_space, B, m, M, Ti, N, learn_rate, reps, centers_x, centers_y,
         plot=False):
    impa = list(np.random.choice(action_space, M))
    lik = []
    theta_m = np.zeros_like(theta)
    alpha_m = np.zeros_like(alpha)
    y_theta = theta.copy()
    y_alpha = alpha.copy()
    best = -np.inf
    best_theta = theta_m.copy()
    best_alpha = alpha_m.copy()
    tm = 1
    last_lik = -np.inf
    for _ in range(reps):
        permut = list(np.random.permutation(range(N)))
        for n in permut:
            t = 1/2*(1 + np.sqrt(1 + 4*tm**2))
            
            data = np.array(traj_data[n])
            reward_est = lin_rew_func(y_theta, state_space, centers_x,
              centers_y)
            R_all, E_all = RE_all(reward_est, data, TP, state_space, m,
              centers_x, centers_y)
            beta = np.einsum('ij,ijk->ik', y_alpha, E_all)
            
            logZvec, gZ_theta, gZ_alpha = logZ_det(beta,
              impa, reward_est, data, M, TP, R_all, E_all, state_space,
              action_space, m, Ti, centers_x, centers_y)
          
            loglikelihood = loglik(state_space, Ti, beta, data, TP, m, R_all,
              logZvec).sum()
            # appears not to be maximized at true theta/alpha, why?
            lik.append(loglikelihood)
            
            if loglikelihood > best:
                best = loglikelihood
                best_theta = y_theta.copy()
                best_alpha = y_alpha.copy()
              
            g_theta = theta_grad_det(data, beta, state_space, gZ_theta,
              centers_x, centers_y)
            g_alpha = alpha_grad_det(gZ_alpha, R_all, E_all)
            
            g_theta = g_theta / np.linalg.norm(g_theta)
            g_alpha = g_alpha / np.linalg.norm(g_alpha, 'f')
          
            theta_m, alpha_m, = theta, alpha
            theta = y_theta + learn_rate*g_theta
            alpha = y_alpha + learn_rate*g_alpha
            
            mult = (tm - 1)/t
            y_theta = theta + mult*(theta - theta_m)
            y_alpha = np.maximum(alpha + mult*(alpha - alpha_m), 0)
            
            learn_rate *= 0.99
            tm = t
            
            if loglikelihood < last_lik:
                y_theta = theta.copy()
                y_alpha = alpha.copy()
                tm = 1
                
            last_lik = loglikelihood
    if plot:
        plt.plot(lik)
    return best_theta, best_alpha


def random_algo(theta, alpha, sigsq, phi, beta, traj_data, TP, state_space,
         action_space, B, m, M, Ti, N, learn_rate, reps, centers_x, centers_y,
         plot=False):
    return (np.random.normal(size=theta.shape), 0)


'''
Setting ETA_COEF to 5 (so that experts behave basically randomly everywhere)
it's evident that the unif model isn't magic. Its robustness in some challenging
settings still suggests to me that it's using some helpful information -
could the feature expectations really help that much? The other algos have
the feature expect too but they don't show this robustness.
 -- Plus, feature expects *depend on the states visited in the trajectories*. If
 the trajectories are noisy, so will the feature expectations be.
'''

####################### Functions for evaluation/testing ######################

def cumulative_reward(s_list, cr_reps, policy, T, state_space, action_space,
                      rewards):
    reward_list = []
    for i in range(cr_reps):
        ret = episode(s_list[i], T, policy, rewards, action_space)[2]
        reward_list.extend(ret)
    return reward_list


def evaluate_general(theta, alpha, sigsq, phi, beta, traj_data, TP, state_space,
                     action_space, B, m, M, Ti, N, learn_rate, reps, policy, T,
                     rewards, init_Q, J, centers_x, centers_y,
                     cr_reps, algo_a, algo_b, random=False, save=False):
    start = datetime.datetime.now()
    s_list = [state_space[np.random.choice(len(state_space))] for _ in range(cr_reps)]
    true_rew = cumulative_reward(s_list, cr_reps, policy, T, state_space,
      action_space, rewards)
    plt.plot(np.cumsum(true_rew), color='b') 
    true_total = np.sum(true_rew)
    totals = [[],[]]
    cols = ['r', 'g']
    for j in range(J):
        ta = algo_a(theta, alpha, sigsq, phi, beta, traj_data, TP,
          state_space, action_space, B, m, M, Ti, N, learn_rate, reps, centers_x,
          centers_y)[0]
        if random:
            tb = np.random.normal(size=theta.shape)
        else:
            tb = algo_b(theta, alpha, sigsq, phi, beta,
              traj_data, TP, state_space, action_space, B, m, M, Ti, N, learn_rate,
              reps, centers_x, centers_y)[0]
        theta_stars = [ta, tb]
        for i in range(2):
            reward_est = lin_rew_func(theta_stars[i], state_space, centers_x,
              centers_y)
            est_policy = value_iter(state_space, action_space, reward_est, TP,
              GAM, 1e-5)[0]
            est_rew = cumulative_reward(s_list, cr_reps, est_policy, T,
              state_space, action_space, rewards)
            plt.plot(np.cumsum(est_rew), color=cols[i])
            totals[i].append(np.sum(est_rew))
        sec = (datetime.datetime.now() - start).total_seconds()
        print(str(round((j+1)/J*100, 3)) + '% done: ' + str(round(sec / 60, 3)))
    if save:
        plt.savefig(save[0] + '___' + save[1] + '.png')
        plt.show()
    return true_total, totals[0], totals[1], np.std(totals[0]), np.std(totals[1])


def evaluate_all(theta, alpha, sigsq, phi, beta, traj_data, TP, state_space,
                     action_space, B, m, M, Ti, N, learn_rate, reps, policy, T,
                     rewards, init_Q, J, centers_x, centers_y,
                     cr_reps, save=False):
    start = datetime.datetime.now()
    s_list = [state_space[np.random.choice(len(state_space))] for _ in range(cr_reps)]
    true_rew = cumulative_reward(s_list, cr_reps, policy, T, state_space,
      action_space, rewards)
    plt.plot(np.cumsum(true_rew), color='b') 
    true_total = np.sum(true_rew)
    totals = [[],[], [], []]
    cols = ['r', 'g', 'k', 'm']
    for j in range(J):
        ta = MEIRL_det_pos(theta, alpha, sigsq, phi, beta, traj_data, TP,
          state_space, action_space, B, m, M, Ti, N, learn_rate, reps,
          centers_x, centers_y)[0]
        tb = MEIRL_unif(theta, alpha, sigsq, phi, beta, traj_data, TP,
          state_space, action_space, B, m, M, Ti, N, learn_rate, reps,
          centers_x, centers_y)[0]
        tc = AR_AEVB(theta, alpha, sigsq, phi, beta, traj_data, TP,
          state_space, action_space, B, m, M, Ti, N, learn_rate, reps,
          centers_x, centers_y)[0]
        td = random_algo(theta, alpha, sigsq, phi, beta, traj_data, TP,
          state_space, action_space, B, m, M, Ti, N, learn_rate, reps,
          centers_x, centers_y)[0]
        theta_stars = [ta, tb, tc, td]
        for i in range(4):
            reward_est = lin_rew_func(theta_stars[i], state_space, centers_x,
              centers_y)
            est_policy = value_iter(state_space, action_space, reward_est, TP,
              GAM, 1e-5)[0]
            est_rew = cumulative_reward(s_list, cr_reps, est_policy, T,
              state_space, action_space, rewards)
            plt.plot(np.cumsum(est_rew), color=cols[i])
            totals[i].append(np.sum(est_rew))
        sec = (datetime.datetime.now() - start).total_seconds()
        print(str(round((j+1)/J*100, 3)) + '% done: ' + str(round(sec / 60, 3)))
    if save:
        plt.savefig(save[0] + '___' + save[1] + '.png')
        plt.show()
    return (true_total, totals[0], totals[1], totals[2], totals[3],
      np.std(totals[0]), np.std(totals[1]), np.std(totals[2]),
      np.std(totals[3]))


seeds_1 = [20,40,60,80,100]
seeds_2 = [120,140,160,180,200]

HYPARAMS = {'D': 16,
            'MOVE_NOISE': 0.05,
            'INTERCEPT_ETA': 0,
            'WEIGHT': 2,
            'RESET': 20,
            'COEF': 0.1,
            'ETA_COEF': 0.01,
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

def save_results(id_num, algo_a=AR_AEVB, algo_b=MEIRL_unif, random=False,
                 test_data='myo', seeds=seeds_1, hyparams=HYPARAMS):
    '''
    Compares two algorithms with each other (and with ground truth reward)
    with varying seed values, including varying reward functions
    '''
    
    # Global params
    global D, MOVE_NOISE, INTERCEPT_ETA, WEIGHT, RESCALE, RESET, COEF
    global ETA_COEF, GAM, M, N, J, T, Ti, B, INTERCEPT_REW, TP
    (D, MOVE_NOISE, INTERCEPT_ETA, WEIGHT, RESET, COEF, ETA_COEF, GAM, M, N, J,
      T, Ti, B, INTERCEPT_REW, learn_rate, cr_reps, reps,
      sigsq_list) = (hyparams['D'], hyparams['MOVE_NOISE'],
      hyparams['INTERCEPT_ETA'], hyparams['WEIGHT'], hyparams['RESET'],
      hyparams['COEF'], hyparams['ETA_COEF'], hyparams['GAM'], hyparams['M'],
      hyparams['N'], hyparams['J'], hyparams['T'], hyparams['Ti'],
      hyparams['B'], hyparams['INTERCEPT_REW'], hyparams['learn_rate'],
      hyparams['cr_reps'], hyparams['reps'], hyparams['sigsq_list'])
    d = D // 2 + 1
    state_space = np.array([(i,j) for i in range(D) for j in range(D)])
    action_space = list(range(4))
    TP = transition(state_space, action_space)
    
    alpha1 = np.array([WEIGHT, 0, 0, 0, 1])
    alpha2 = np.array([0, 0, WEIGHT, 0, 1])
    alpha3 = np.array([0, 0, 0, WEIGHT, 1]) 
    alpha4 = np.array([0, WEIGHT, 0, 0, 1])
    
    p = alpha1.shape[0]
    m = 4
    
    ex_alphas = np.stack([alpha1, alpha2, alpha3, alpha4])
    ex_sigsqs = np.array(sigsq_list)
    
    init_Q = np.random.rand(D,D,4)
    
    for seed in seeds:
        filename = '_'.join(str(datetime.datetime.now()).split())
        fname = str(id_num) + '$' + filename.replace(':', '--')
        os.mkdir('results/' + fname)
        
        SEED_NUM = seed#100#50#80#70#60
        np.random.seed(SEED_NUM) #50) #40) #30) #20) #10)
        
        centers_x = np.random.choice(D, D//2)
        centers_y = np.random.choice(D, D//2)
        
        theta_true = 3*np.random.rand(D // 2 + 1) - 2 #np.random.normal(size = D // 2 + 1, scale=3)
        rewards = lin_rew_func(theta_true, state_space, centers_x, centers_y)
        sns.heatmap(rewards)
        plt.savefig('results/' + fname + '/' + 'true_reward.png')
        plt.show()
        
        opt_policy, Q = value_iter(state_space, action_space, rewards, TP, GAM, 1e-5)
        
        compare_myo_opt(rewards, TP, Q, save=fname)
        
        phi = np.random.rand(m,2)
        alpha = np.random.normal(size=(m,p), scale=0.05)
        #sigsq = 1e-16 + np.zeros(m)
        sigsq = np.random.rand(m)
        beta = np.random.rand(m)
        theta = np.random.normal(size=d) #np.zeros_like(theta_true)
        
        traj_data = make_data(ex_alphas, ex_sigsqs, rewards, N, Ti, state_space, action_space,
                             TP, m)
        boltz_data = make_data(ex_alphas, ex_sigsqs, rewards, N, Ti, state_space, action_space,
                             TP, m, Q)
        dumb_data = random_data(ex_alphas, ex_sigsqs, rewards, N, Ti, state_space, action_space,
                             TP, m)
        
        alg_a_str = str(algo_a).split()[1]
        if random:
            alg_b_str = 'random'
        else:
            alg_b_str = str(algo_b).split()[1]
        
        if test_data == 'myo':
            true_tot, a_tot, b_tot, a_sd, b_sd = evaluate_general(theta, alpha, sigsq, phi, beta, 
                                                         traj_data,
              TP, state_space,
              action_space, B, m, M, Ti, N, learn_rate, reps, opt_policy, T,
              rewards, init_Q, J, centers_x, centers_y,
              cr_reps, algo_a, algo_b, random=random,
              save=['results/' + fname + '/' + fname,
                    alg_a_str + '__' + alg_b_str])
        else:
            true_tot, a_tot, b_tot, a_sd, b_sd = evaluate_general(theta, alpha, sigsq, phi, beta,
                                                          boltz_data,
              TP, state_space,
              action_space, B, m, M, Ti, N, learn_rate, reps, opt_policy, T,
              rewards, init_Q, J, centers_x, centers_y,
              cr_reps, algo_a, algo_b, random=random,
              save=['results/' + fname + '/' + fname,
                    alg_a_str + '__' + alg_b_str])
            
        f = open('results/' + fname + '/' + fname + '.txt', 'w')
        f.write('D = ' + str(D) + '\n')
        f.write('MOVE_NOISE = ' + str(MOVE_NOISE) + '\n')
        f.write('INTERCEPT_ETA = ' + str(INTERCEPT_ETA) + '\n')
        f.write('INTERCEPT_REW = ' + str(INTERCEPT_REW) + '\n')
        f.write('WEIGHT = ' + str(WEIGHT) + '\n')
        f.write('RESET = ' + str(RESET) + '\n')
        f.write('COEF = ' + str(COEF) + '\n')
        f.write('GAM = ' + str(GAM) + '\n')
        f.write('ETA_COEF = ' + str(ETA_COEF) + '\n')
        f.write('M = ' + str(M) + '\n')
        f.write('N = ' + str(N) + '\n')
        f.write('J = ' + str(J) + '\n')
        f.write('T = ' + str(T) + '\n')
        f.write('Ti = ' + str(Ti) + '\n')
        f.write('B = ' + str(B) + '\n')
        f.write('learn_rate = ' + str(learn_rate) + '\n')
        f.write('cr_reps = ' + str(cr_reps) + '\n')
        f.write('reps = ' + str(reps) + '\n')
        f.write('centers_x = ' + str(centers_x) + '\n')
        f.write('centers_y = ' + str(centers_y) + '\n')
        f.write('SEED_NUM = ' + str(SEED_NUM) + '\n')
        f.write('theta_true = ' + str(theta_true) + '\n')
        f.write('ex_alphas = ' + str(ex_alphas) + '\n')
        f.write('ex_sigsqs = ' + str(ex_sigsqs) + '\n')
        f.write('alpha = ' + str(alpha) + '\n')
        f.write('sigsq = ' + str(sigsq) + '\n')
        f.write('theta = ' + str(theta) + '\n')
        f.write('beta = ' + str(beta) + '\n')
        f.write('phi = ' + str(phi) + '\n')
        f.write('test_data = ' + str(test_data) + '\n')
            
        f.write('true_tot = ' + str(true_tot) + '\n')
        f.write('mean algo_a_tot = ' + str(np.mean(a_tot)) + '\n')
        f.write('sd algo_a_tot = ' + str(a_sd) + '\n')
        f.write('mean algo_b_tot = ' + str(np.mean(b_tot)) + '\n')
        f.write('sd algo_b_tot = ' + str(b_sd) + '\n')
        f.close()
        

def results_var_hyper(id_num, param, par_vals, seed, test_data='myo',
                      hyparams=HYPARAMS):
    SEED_NUM = seed
    for par in par_vals:
        hyparams[param] = par
        
        np.random.seed(SEED_NUM)
        global D, MOVE_NOISE, INTERCEPT_ETA, WEIGHT, RESCALE, RESET, COEF
        global ETA_COEF, GAM, M, N, J, T, Ti, B, INTERCEPT_REW, TP
        (D, MOVE_NOISE, INTERCEPT_ETA, WEIGHT, RESET, COEF, ETA_COEF, GAM, M, N, J,
          T, Ti, B, INTERCEPT_REW, learn_rate, cr_reps, reps,
          sigsq_list) = (hyparams['D'], hyparams['MOVE_NOISE'],
          hyparams['INTERCEPT_ETA'], hyparams['WEIGHT'], hyparams['RESET'],
          hyparams['COEF'], hyparams['ETA_COEF'], hyparams['GAM'], hyparams['M'],
          hyparams['N'], hyparams['J'], hyparams['T'], hyparams['Ti'],
          hyparams['B'], hyparams['INTERCEPT_REW'], hyparams['learn_rate'],
          hyparams['cr_reps'], hyparams['reps'], hyparams['sigsq_list'])
        d = D // 2 + 1
        state_space = np.array([(i,j) for i in range(D) for j in range(D)])
        action_space = list(range(4))
        TP = transition(state_space, action_space)
        
        alpha1 = np.array([WEIGHT, 0, 0, 0, 1])
        alpha2 = np.array([0, 0, WEIGHT, 0, 1])
        alpha3 = np.array([0, 0, 0, WEIGHT, 1]) 
        alpha4 = np.array([0, WEIGHT, 0, 0, 1])
        
        p = alpha1.shape[0]
        m = 4
        
        ex_alphas = np.stack([alpha1, alpha2, alpha3, alpha4])
        ex_sigsqs = np.array(sigsq_list)
        
        init_Q = np.random.rand(D,D,4)
    
        filename = '_'.join(str(datetime.datetime.now()).split())
        fname = str(id_num) + '$' + filename.replace(':', '--')
        os.mkdir('hyp_results/' + fname)
        
        centers_x = np.random.choice(D, D//2)
        centers_y = np.random.choice(D, D//2)
        
        theta_true = 3*np.random.rand(D // 2 + 1) - 2 #np.random.normal(size = D // 2 + 1, scale=3)
        rewards = lin_rew_func(theta_true, state_space, centers_x, centers_y)
        sns.heatmap(rewards)
        plt.savefig('hyp_results/' + fname + '/' + 'true_reward.png')
        plt.show()
        
        opt_policy, Q = value_iter(state_space, action_space, rewards, TP,
          GAM, 1e-5)
        
        compare_myo_opt(rewards, TP, Q, save=fname, hyp=True)
        
        phi = np.random.rand(m,2)
        alpha = np.random.normal(size=(m,p), scale=0.05)
        #sigsq = 1e-16 + np.zeros(m)
        sigsq = np.random.rand(m)
        beta = np.random.rand(m)
        theta = np.random.normal(size=d) #np.zeros_like(theta_true)
        
        traj_data = make_data(ex_alphas, ex_sigsqs, rewards, N, Ti, state_space, action_space,
                             TP, m)
        boltz_data = make_data(ex_alphas, ex_sigsqs, rewards, N, Ti, state_space, action_space,
                             TP, m, Q)
        dumb_data = random_data(ex_alphas, ex_sigsqs, rewards, N, Ti, state_space, action_space,
                             TP, m)
        
        
        if test_data == 'myo':
            (true_tot, a_tot, b_tot, c_tot, d_tot, a_sd, b_sd, c_sd,
             d_sd) = evaluate_all(theta, alpha, sigsq, phi, beta, traj_data,
              TP, state_space, action_space, B, m, M, Ti, N, learn_rate, reps,
              opt_policy, T, rewards, init_Q, J, centers_x, centers_y,
              cr_reps, save=['hyp_results/' + fname + '/' + fname, param])
        else:
            (true_tot, a_tot, b_tot, c_tot, d_tot, a_sd, b_sd, c_sd,
             d_sd) = evaluate_all(theta, alpha, sigsq, phi, beta, boltz_data,
              TP, state_space, action_space, B, m, M, Ti, N, learn_rate, reps,
              opt_policy, T, rewards, init_Q, J, centers_x, centers_y,
              cr_reps, save=['hyp_results/' + fname + '/' + fname, param])
            
        f = open('hyp_results/' + fname + '/' + fname + '.txt', 'w')
        f.write('D = ' + str(D) + '\n')
        f.write('MOVE_NOISE = ' + str(MOVE_NOISE) + '\n')
        f.write('INTERCEPT_ETA = ' + str(INTERCEPT_ETA) + '\n')
        f.write('INTERCEPT_REW = ' + str(INTERCEPT_REW) + '\n')
        f.write('WEIGHT = ' + str(WEIGHT) + '\n')
        f.write('RESET = ' + str(RESET) + '\n')
        f.write('COEF = ' + str(COEF) + '\n')
        f.write('GAM = ' + str(GAM) + '\n')
        f.write('ETA_COEF = ' + str(ETA_COEF) + '\n')
        f.write('M = ' + str(M) + '\n')
        f.write('N = ' + str(N) + '\n')
        f.write('J = ' + str(J) + '\n')
        f.write('T = ' + str(T) + '\n')
        f.write('Ti = ' + str(Ti) + '\n')
        f.write('B = ' + str(B) + '\n')
        f.write('learn_rate = ' + str(learn_rate) + '\n')
        f.write('cr_reps = ' + str(cr_reps) + '\n')
        f.write('reps = ' + str(reps) + '\n')
        f.write('centers_x = ' + str(centers_x) + '\n')
        f.write('centers_y = ' + str(centers_y) + '\n')
        f.write('SEED_NUM = ' + str(SEED_NUM) + '\n')
        f.write('theta_true = ' + str(theta_true) + '\n')
        f.write('ex_alphas = ' + str(ex_alphas) + '\n')
        f.write('ex_sigsqs = ' + str(ex_sigsqs) + '\n')
        f.write('alpha = ' + str(alpha) + '\n')
        f.write('sigsq = ' + str(sigsq) + '\n')
        f.write('theta = ' + str(theta) + '\n')
        f.write('beta = ' + str(beta) + '\n')
        f.write('phi = ' + str(phi) + '\n')
        f.write('test_data = ' + str(test_data) + '\n')
            
        f.write('true_tot = ' + str(true_tot) + '\n')
        f.write('mean MEIRL_det_tot = ' + str(np.mean(a_tot)) + '\n')
        f.write('sd MEIRL_det_tot = ' + str(a_sd) + '\n')
        f.write('mean MEIRL_unif_tot = ' + str(np.mean(b_tot)) + '\n')
        f.write('sd MEIRL_unif_tot = ' + str(b_sd) + '\n')
        f.write('mean AR_AEVB_tot = ' + str(np.mean(c_tot)) + '\n')
        f.write('sd AR_AEVB_tot = ' + str(c_sd) + '\n')
        f.write('mean random_tot = ' + str(np.mean(d_tot)) + '\n')
        f.write('sd random_tot = ' + str(d_sd) + '\n')
        f.close()
    
#%%

'''
How is the unif model so robust???

Performance of AR and unif seem totally unaffected by sigsq


(Bearing in mind that J=10 for these, so not too rigorous)

ETA_COEF = 0.01, J=10:

    Results from Seed = [20,40,60,80,100]; ETA_COEF = 0.01; N=100; not sparse reward:
        * both det and unif do very well on all but 100 - quite unclear what makes
        this one so much harder
        * Random does not do well, suggesting it's not just that non-sparse reward
        MDPs are easy - the uniform model is using *some* sort of critical info.
        Probably the feature expectations are helping, but shouldn't that still
        be contaminated by incorrect beta, and R_Z?
        
    Next looking at performance on exact same seeds, but boltz_data:
        * For seeds 20, 40, 80, the algos based on myopic models still successfully
        match the performance of optimal even when data come from Q-based model!
         - on seed 60, they do worse than optimal, but still much better than
         random; unif does better than det here
         - again they sorta struggle with 100, although unif does decently well
         (500 vs opt 600)
         
ETA_COEF = 0.05, J=20 (less coverage of expertise):
    * on seed 20, both fall p short of optimal (1600), but still much better than random
    and here det noticeably outperforms unif (~1080 to 790)
    * seed 40, 60, 100 has opposite pattern, both quite suboptimal but unif does slightly better

ETA_COEF = 0.01, J=20, sigsq=0.1, AR vs unif:
    * seed 20: AR does well on some trials but sucks on others, seems to find
    anti-optimizing solutions. But at least now it isn't *always* bad.
    * seed 40: again better than avg but AR still p bad relative to unif
    * seed 60: same story
    
ETA_COEF = 0.01, J=20, sigsq=0.4, reps=5, AR vs unif:
    * seed 20: AR does well on some trials but sucks on others, seems to find
    anti-optimizing solutions. But at least now it isn't *always* bad.
    * seed 40: again better than avg but AR still p bad relative to unif
    * seed 60: same story
'''

#regular
theta_star, phi_star, alpha_star, sigsq_star = ann_AEVB(theta, alpha, sigsq, phi, beta, traj_data, TP, state_space,
         action_space, B, m, M, Ti, N, learn_rate, reps, centers_x, centers_y)
theta_star_2, phi_star_2, alpha_star_2, sigsq_star_2 = AEVB(theta, alpha, sigsq, phi, beta, traj_data, TP, state_space,
         action_space, B, m, M, Ti, N, learn_rate, reps, centers_x, centers_y)
theta_star_p, alpha_star_p = MEIRL_det_pos(theta, alpha, sigsq, phi, beta, traj_data, TP, state_space,
         action_space, B, m, M, Ti, N, learn_rate, reps, centers_x, centers_y)
theta_star_AR, phi_star_AR, alpha_star_AR, sigsq_star_AR = AR_AEVB(theta, alpha,
  sigsq, phi, beta, traj_data, TP, state_space, action_space, B, m, M, Ti, N,
  learn_rate, reps, centers_x, centers_y)
theta_star_u, beta_star_u = MEIRL_unif(theta, alpha, sigsq, phi, beta, traj_data, TP, state_space,
         action_space, B, m, M, Ti, N, learn_rate, reps, centers_x, centers_y)
'''
Testing AR. Seems to consistently find a certain solution, but that solution is
very wrong...
* Trying different samples of trajectories, this doesn't some to be a problem
of sensitivity to the sample. It still consistently settles on this wrong
answer.
* Also not because of negative alphas - I used the projection and it still doesn't
work!
* Evidently sensitive to INITIAL THETA. varying initial other params didn't change
much, but varying theta does a lot


sigsq of 0.5 --> unif does terrible on non-sparse, hallucinates high reward
in a corner consistently...
'''
# these all seem to do terribly on seed 10, except when sigsq is 0.01 rather than 2 -
# then MEIRL_det_pos works quite well 

#boltz - FIX THIS
phi_star_b, theta_star_b, alpha_star_b, sigsq_star_b = ann_AEVB(theta, alpha, sigsq, phi, beta, boltz_data, TP, state_space,
         action_space, B, m, M, Ti, N, learn_rate, reps, centers_x, centers_y)
phi_star_2, theta_star_2, alpha_star_2, sigsq_star_2 = AEVB(theta, alpha, sigsq, phi, beta, boltz_data, TP, state_space,
         action_space, B, m, M, Ti, N, learn_rate, reps, centers_x, centers_y)
theta_star_p, alpha_star_p = MEIRL_det_pos(theta, alpha, sigsq, phi, beta, boltz_data, TP, state_space,
         action_space, B, m, M, Ti, N, learn_rate, reps, centers_x, centers_y)
phi_star_AR, theta_star_AR, alpha_star_AR, sigsq_star_AR = AR_AEVB(theta, alpha, sigsq, phi, beta, boltz_data, TP, state_space,
         action_space, B, m, M, Ti, N, learn_rate, reps, centers_x, centers_y)
theta_star_u, beta_star_u = MEIRL_unif(theta, alpha, sigsq, phi, beta, boltz_data, TP, state_space,
         action_space, B, m, M, Ti, N, learn_rate, reps, centers_x, centers_y)

# see_trajectory(rewards, np.array(traj_data[0])[0,0])

dumb_data = make_data(alpha_star_2, sigsq_star_2, lin_rew_func(theta_star_2,
                            state_space, centers_x, centers_y), N, Ti, state_space, action_space,
                            TP, m)

#sns.heatmap(lin_rew_func(theta_true, state_space, centers_x, centers_y))

theta_s, beta_s = MEIRL_unif(theta, beta, traj_data, TP, state_space,
         action_space, B, m, M, Ti, N, learn_rate, 1, GD_unif, centers_x, centers_y)

'''
^ results from this, ****SEED 10****:
true_tot
Out[87]: -287.57249017468143

np.mean(det_tot_p)
Out[88]: -404.53112168237095

np.mean(unif_tot_p)
Out[89]: -484.9908606854497

for comparison, random results in -1000

SEED 20: det and unif both match optimal p consistently, this MDP seems easy

SEED 30: again, det and unif matching optimal; random theta def doesn't, so
 not trivial
'''


# evidently sensitive to initialization, but maybe there's something principled
# about initializing at theta = 0? Implies prior of ignorance about reward


# promising results when using N = 50 and reps = 5, but might
# not replicate...
## ^ Yeah, when tried to replicate, the AEVB occasionally gave lackluster results
## But! Still on average much better than uniform model:
## mean(AEVB_tot) = 1231.5
## mean(unif_tot) = 621.5
# Another rep: no better than random...

# when less sparse reward, does *worse* than uniform model...
# Problem seems to boil down to degeneracy - many thetas have close-to-optimal ELBO
# (about -290 for true params vs -330 for theta_star that gives wrong answer)
# yet assign drastically different reward profiles, e.g. bad state becomes good
# and good becomes bad
# **** lemme see if increasing sample size changes this
# **** yeah it widens gap a bit, although not much; now its -273 for true, -320 for a
# **** very wrong answer

# works p well even under misspecification for D = 8
# doesn't really work on D = 16 grid

'''
To do:
    * misspec of reward function?
    * misspec of beta mean?
    * more/longer trajectories? Could shorten the training traj to speed training
     but have longer test trajectories to see if long-term reward is improved
    * try different mu functions for the locally optimal experts; maybe
     need some threshold of optimality for each expert to get good results
     -- Looks like when ETA_COEF is set to 0.1 (i.e. very little area covered
     by optimal experts), the results (on seed 30) are consistently a wide
     negative blob at the top! Similar results for AEVB. Det does not do well
     here.
     -- The uniform model has no such regularity. Maybe that's the key; the
     well-specified algorithms are perhaps less robust because they consistently
     reach a wrong answer (when the noise-to-signal ratio is just too high).
     But nonetheless the uniform model seems to reach something consistently
     better than random, indeed much better
     -- maybe learn_rate...
    * Try restricting beta to be positive - in principle, quite hard to
      distinguish a state with high positive reward being successfully pursued
      by most experts from a state with high neg reward being anti-optimized
         --- done
    * Vary:
        - sample size -- see how many samples necessary to get good performance
            -- tentatively looks like N=100 each is sufficient for det to work,
            even N=20! N=20 also sufficient for unif
        - sigma -- maybe true model will outperform uniform when high variance?
    * Maybe test sample complexity necessary to get some epsilon-close results,
     plot against (1) size of grid world, (2) amt of noise, (3) coef for mu?
     * May need to force reward function to be such that doing well in the MDP
     is hard, i.e. not sufficient to just get one or two "nodes" of the reward
     space correct
     * Compare performance for different split of steps in Ti vs N (length vs
     number of trajectories)
     * Check action-value distribution vs next-step reward distribution, check
     that the reason robust to Boltz isn't just that those distributions are
     v similar
         -- sort of done but need to organize this
     * compare diff sigsqs for different experts
     * test on MCMC...
     * vary seed used for everything *after* definition of the MDP
     * increase number of features, for same size of MDP
     * count of states where myo best and opt best differ
     * look at learning process of meirl-unif on seed 160, boltz data, in detail, how does
     it work so well?
     
QUALITATIVE NOTES:
    * Good performance is basically elusive in large D MDPs when beta is allowed
    to be negative.
    * The varying-beta model appears to do better than uniform when rewards are
    sparse *and* states-of-high-expertise by the demonstrators are also sparse,
    *and* the coverage of these states-of-high-expertise is wide (e.g. all 4
    corners).
     - In sparse-reward MDP, results seem to be hit-or-miss, as you'd expect;
     either the algo places highest reward on the right spot and pursues it above
     all others - thus doing well - or it doesn't, and does quite poorly.
     - Increasing coverage of the state space by the experts' high betas seems
     to improve performance of both det and unif (they get to be higher fraction
     of optimal in cumulative return). But increasing this coverage helps the
     unif model a lot more than det, apparently.
    * All the models seem to do better than random even when there are larger
    chunks of the state space on which the demonstrators act randomly.
    * Slightly slower to train on higher Ti:N ratio, holding Ti*N constant -
    will see if performance changes, though
    
Future directions:
    * sliding scale of a parameter for how myopic the experts are - expected
    k-step rewards...
    * robustness - avoiding cases where the algo hallucinates high pos reward
    in a high NEGATIVE state(s) and anti-optimizes
    * test on simulations from experts whose beta functions mimick different
    cognitive biases
'''


######
'''
DEFAULTS FOR PARAMS:
    D=16 #8 #6x
    MOVE_NOISE = 0.05
    INTERCEPT_ETA = 0
    WEIGHT = 2
    RESCALE = 1   <---- getting rid of this, just alternating signs
    RESET = 20
    COEF = 0.1
    ETA_COEF = 0.01
    GAM = 0.9
    M = 20 # number of actions used for importance sampling
    N = 100 #20 #100 #2000 # number of trajectories per expert
    J = 20 #10 # should be 30....
    T = 50
    Ti = 20 # length of trajectory
    B = 50 #100 # number of betas/normals sampled for expectation
    Q_ITERS = 30000 #50000
    learn_rate = 0.5 #0.0001
    cr_reps = 10
    reps = 5
    sigsqs = 0.1 for all experts

'''
######


'''
Results I've recorded:
    
seeds 20,40,60,80,100
19) ETA_COEF = 5, all suck
20) 
21) like #20 but BOLTZ:
 * seed 100 has many states where best myopic action is different from
 best long-term action, yet both algos work very well on the Boltzmann data!
22)               
23) like #20 but Ti = 50; interestingly the results barely change for seed 20, seems
                  insensitive to trajectory length at least above 20 - good
                  news for sample complexity
 - or maybe not, seed 40, 60 det takes a big hit; on 80 both do worse.
 - helps on seed 100 tho
24) now Ti = 21, trying to see if just extremely sensitive to change in seed order
 - not *too* drastic a change from #20
 - pretty strong change (~400 difference) from #20 on seed 60
 - I guess not surprising since this shifts back the random seed determining
   start states -- could try to make this consistent...?
                
   
seeds 120,140,160,180,200
25) like #20
26) like #23
 - both algos *beat the expert* in seed 200!
                  

Added gradient clipping to all algos for >= 23!                  
                  
                  
### 100s are results using thetas drawn from uniform, apparently no longer
### have issue where unif does better than random when trained on random data
### * after that point,     RESCALE = np.array([1,-1] * ((d-1) // 2))
### * Note that for these trials, the unif model uses different Nesterov than
### others

100) meirl_unif vs random  ;  INTERCEPT_REW = -1, sigsqs = 1.5; seeds_1
101) meirl_det_pos vs meirl_unif  ;  INTERCEPT_REW = -1, sigsqs = 1.5; seeds_1
102) meirl_det_pos vs meirl_unif  ;  INTERCEPT_REW = -1, sigsqs = 1.5; seeds_2
103) meirl_det_pos vs meirl_unif  ;  INTERCEPT_REW = -1, sigsqs = 1.5; seeds_1; boltz
104) meirl_det_pos vs meirl_unif  ;  INTERCEPT_REW = -1, sigsqs = 1.5; seeds_2; boltz
105) AR_AEVB vs meirl_unif  ;  INTERCEPT_REW = -1, sigsqs = 1.5; seeds_1




                             # AFTER MODIFYING UNIF TO USE FISTA #
200) meirl_unif vs random  ;  INTERCEPT_REW = -1, sigsqs = 1.5; seeds_1 
201) AR_AEVB vs ann_AEVB  ;  INTERCEPT_REW = -1, sigsqs = 0.1; seeds_1 
                  
    
**********(NOTE: MAY NEED TO REDO EVERYTHING UNDER #100)***************
    
    * det_pos vs random; sigsq = 1.5
    * det_pos vs unif; sigsq = 1.5; ETA_COEF = 5
     - mostly sucks, but this isn't surprising bc noise drowns it out; this is
     good, means unif isn't cheating
    * det_pos vs AR_AEVB; sigsq = 0.8 - both suck but AEVB sometimes does well
    * det_pos vs AR_AEVB; sigsq = 0.1 - det consistently sucks here
    * det_pos vs AR_AEVB; sigsq = 0.01 - still det sucks, weird, it used to work
    * det_pos vs AR_AEVB; sigsq = 0.01, learn_rate = 0.0001 - barely changes from
    init_theta so both are consistently bad
    
    * det_pos vs AR_AEVB; sigsq = 0.01; learn_rate = 0.1; init sigsq = 0;
    N = 500 - now AEVB seems to do much better on all seeds; but det
    does a bit worse;
    * AR_AEVB vs random; sigsq = 0.01; learn_rate = 0.1; init sigsq = 1e-16; N = 2000:
        extra samples don't help, in fact does worse than with N = 500
    * unif vs random; sigsq = 2; init sigsq = 1e-16; ETA_COEF = 5 -- there should
    be basically no signal for the algorithm to learn from here...
    * det_pos vs unif; sigsq = 2; ETA_COEF = 5; sigsq init 1e-16 -- much better
    results than the second bullet; I guess initialization responsible, but this
    is still such a drastic change for one difference (which pushes back every
    other random draw)
      - how on earth is this going so well, when ETA_COEF is so high (and hence
      signal should be drowned out)?
    * det_pos vs unif; sigsq = 1.5; ETA_COEF = 0.01 - so now there's at least
    some signal, but large sigsq still may be an issue
      - Interestingly, they both do quite well, though det occasionally anti-optimizes.
      It seems the algorithms are able to recover the signal amid strong noise in
      sampling of beta, but not so much a complete lack of expertise
    * same as above but now on BOLTZ data:
      - det does worse but still better than random, unif does *better*...




Did a sanity check with junk data (0s for all states and actions) - unif sucks
with this input, so it's not cheating evidently (*)

Maybe also check on junk data less trivial than above - demonstrator equally
likely to take each action in each state


Trying on random data when D = 8 makes unif indistinguishable (in total reward)
from random algo. So I guess this problem scales with D...

Switched to uniform distribution to generate theta and now it's finally not
working on random data
'''


'''
[need to test all 10 seeds; vary sigsq, ETA_COEF, N?]

RESULTS FROM results_var_hyper:
1) [seed 20] sigsq varying from 0.01, 0.1, 1, 5 --- good candidate for boltz comparison
2) [seed 60] sigsq varying from 0.01, 0.1, 1, 5
3) [seed 100] sigsq varying from 0.01, 0.1, 1, 5 --- good candidate for boltz comparison
4) [seed 140] sigsq varying from 0.01, 0.1, 1, 5
5) [seed 180] sigsq varying from 0.01, 0.1, 1, 5
6) [seed 20, boltz] sigsq varying from 0.01, 0.1, 1, 5
7) [seed 60, boltz] sigsq varying from 0.01, 0.1, 1, 5
8)
9)
10)
11) [seed 20] ETA_COEF varying from 0.01, 0.05, 0.5
12) [seed 60] ETA_COEF varying from 0.01, 0.05, 0.5
13) [seed 100] ETA_COEF varying from 0.01, 0.05, 0.5
14) [seed 140] ETA_COEF varying from 0.01, 0.05, 0.5
15) [seed 180] ETA_COEF varying from 0.01, 0.05, 0.5
16) 
17)
18)
19)
20)
21) [seed 40] sigsq varying from 0.01, 0.1, 1, 5 -- boltz comparison?
22) [seed 80] sigsq varying from 0.01, 0.1, 1, 5
23) [seed 120] sigsq varying from 0.01, 0.1, 1, 5
24) [seed 160] sigsq varying from 0.01, 0.1, 1, 5
25) [seed 200] sigsq varying from 0.01, 0.1, 1, 5
26) [seed 40, boltz] sigsq varying from 0.01, 0.1, 1, 5
27) [seed 80, boltz] sigsq varying from 0.01, 0.1, 1, 5
28) [seed 120, boltz] sigsq varying from 0.01, 0.1, 1, 5
29) [seed 160, boltz] sigsq varying from 0.01, 0.1, 1, 5
30) [seed 200, boltz] sigsq varying from 0.01, 0.1, 1, 5
31) [seed 40] ETA_COEF varying from 0.01, 0.05, 0.5
32) [seed 80] ETA_COEF varying from 0.01, 0.05, 0.5
33) [seed 120] ETA_COEF varying from 0.01, 0.05, 0.5
34) [seed 160] ETA_COEF varying from 0.01, 0.05, 0.5
35) [seed 200] ETA_COEF varying from 0.01, 0.05, 0.5
36)
                 
                 
The difference between 24 and 29 is INCREDIBLE - does much better when data
are boltz despite not modeling the demonstrators as such                 
                 
                 
                 [seed 20] N varying from 20, 50, 100

                 
                 
                 
ETA_COEF varying from 0.01, 0.05, 0.5
'''

'''
Include results from random data and from very high ETA_COEF as evidence that
the algos aren't cheating
-- ETA_COEF=0.5 results confirm this
'''


'''
MOVE_NOISE = 0.05
    INTERCEPT_ETA = 0
    WEIGHT = 2
    RESET = 20
    COEF = 0.1
    ETA_COEF = 0.01 #0.05 #0.1 #1
    GAM = 0.9
    M = 20 # number of actions used for importance sampling
    N = 100#20 #100 #2000 # number of trajectories per expert
    J = 20#10 # should be 30....
    T = 50
    Ti = 20 # length of trajectory
    B = 50#100 # number of betas/normals sampled for expectation
    INTERCEPT_REW = -1
    learn_rate = 0.5 #0.0001
    cr_reps = 10
    reps = 5
'''


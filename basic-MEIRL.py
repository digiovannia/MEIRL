import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

'''
A simple DxD gridworld to test out multiple-experts IRL.
Agent can move in cardinal directions; moves in intended direction
with prob 1-MOVE_NOISE, else uniform direction. If moves in illegal
direction, stays in place.

Actions: 0 = up, 1 = right, 2 = down, 3 = left
'''

### Helper functions

def manh_dist(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def act_to_coord(a):
    d = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0,-1)}
    return d[a]

def state_index(tup):
    return D*tup[0] + tup[1]

def multi_state_index(states):
    return D*states[:,0]+states[:,1]

def stoch_policy(det_policy, action_space):
    '''
    Turns an array of actions corresponding to a
    deterministic policy into a "stochastic" policy
    (1 on the action chosen, 0 else)
    '''
    x = np.repeat(range(D), D)
    y = np.tile(range(D), D)
    z = np.ravel(det_policy)
    out = np.zeros((D,D,len(action_space)))
    out[x,y,z] = 1
    return out

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

def grid_step(s, a):
    '''
    Given current state s (in tuple form) and action a, returns resulting state.
    '''
    flip = np.random.rand()
    if flip < MOVE_NOISE:
        a = np.random.choice([0,1,2,3])
    new_state = s + act_to_coord(a)
    return np.minimum(np.maximum(new_state, 0), D-1)

def episode(s,T,policy,rewards,a=-1):
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

def eps_greedy(Q, eps, action_space):
    best = np.argmax(Q, axis=2)
    po = stoch_policy(best, action_space)
    po += eps/(len(action_space)-1)*(1 - po) - eps*po
    return po

def Qlearn(rate, gam, eps, K, T, state_space, action_space,
           rewards, policy, Q):
    '''
    Q-learning. Returns the corresponding optimal policy and Q*.
    '''
    for _ in range(K):
        st = state_space[np.random.choice(len(state_space))]
        for _ in range(T):
            policy = eps_greedy(Q, eps, action_space)
            at = np.random.choice(action_space, p=policy[tuple(st)])    
            sp = grid_step(st,at)
            rt = rewards[tuple(sp)]
            qnext = np.max(Q[sp[0],sp[1]])
            qnow = Q[st[0],st[1],at]
            Q[st[0],st[1],at] += rate*(rt + gam*qnext - qnow)
            st = sp
    policy *= 0
    for i in range(D):
        for j in range(D):
            policy[i,j,np.argmax(Q[i,j])] = 1
    return policy, Q

def value_iter(state_space, action_space, rewards, TP, gam, tol):
    '''
    Action-value iteration for use when transition dynamics are known.
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
    policy = np.zeros((16, 16, 4))
    for i in range(D):
        for j in range(D):
            policy[i,j,np.argmax(Q[(16*i+j)])] = 1
    return policy, Q.reshape(D, D, 4)

def eta(st):
    '''
    Features for beta function
    '''
    return np.array([RESCALE*np.exp(-ETA_COEF*((st[0]-1)**2+(st[1]-1)**2)),
      RESCALE*np.exp(-ETA_COEF*((st[0]-(D-2))**2+(st[1]-(D-2))**2)),
      RESCALE*np.exp(-ETA_COEF*((st[0]-1)**2+(st[1]-(D-2))**2)),
      RESCALE*np.exp(-ETA_COEF*((st[0]-(D-2))**2+(st[1]-1)**2)),
      INTERCEPT_ETA])

def mu(s, alpha):
    '''
    Mean of distribution for beta
    '''
    return np.dot(eta(s), alpha)

def mu_all(alpha):
    muvals = np.zeros((D,D))
    for i in range(D):
        for j in range(D):
            muvals[i,j] = mu((i,j), alpha)
    return muvals

def softmax(v, beta, axis=False):
    if axis:
        x = beta[:,:,None]*v
        w = np.exp(x - np.max(x, axis=axis)[:,:,None])
        z = np.sum(w, axis=axis)
        return np.divide(w, z[:,:,None])
    else:
        w = np.exp(beta*v)
        z = np.sum(w)
        return w / z

def beta_func(alpha, sigsq):
    '''
    Computes a sample of betas for whole state space.
    '''
    noise = np.random.normal(loc=0,scale=np.sqrt(sigsq),
      size=(D,D))
    muvals = mu_all(alpha)
    return muvals + noise

def arr_radial(s, c, coef):
    '''
    Radial basis functions for the reward function, applied to an array of
    states. "c" is a fixed center point.
    '''
    return RESCALE*np.exp(-coef*((s[:,0]-c[0])**2+(s[:,1]-c[1])**2))

def psi_all_states(state_space, centers_x, centers_y):
    '''
    Basis functions for the entire state space
    '''
    lst = list([arr_radial(state_space, (centers_x[i],centers_y[i]),
      COEF) for i in range(len(centers_x))])
    lst.append(INTERCEPT_REW*np.ones(len(state_space)))
    return np.array(lst)

def lin_rew_func(theta, state_space, centers_x, centers_y):
    return np.reshape(theta.dot(psi_all_states(state_space, centers_x,
      centers_y)), (D, D))
    
def arr_expect_reward(rewards, data, TP, state_space):
    '''
    Expected next-step reward for an array of states and actions
    '''
    return TP[data[:,0], data[:,1]].dot(np.ravel(rewards))

def grad_lin_rew(data, state_space, centers_x, centers_y):
    '''
    Gradient of reward function for array of states and actions
    '''
    probs = TP[data[:,0], data[:,1]] # m x Ti x D**2
    return np.swapaxes(probs.dot(psi_all_states(state_space, centers_x,
      centers_y).transpose()), 1, 2)

def init_state_sample(state_space):
    '''
    Returns initial state index; uniform for simplicity
    '''
    return state_space[np.random.choice(len(state_space))]

def myo_policy(beta, s, TP, rewards):
    '''
    Policy for myopic expert (action probability proportional to next-step
    expected reward)
    '''
    expect_rew = TP[state_index(s)].dot(np.ravel(rewards))
    return softmax(expect_rew, beta)

def synthetic_traj(rewards, alpha, sigsq, i, Ti, state_space, action_space,
                       init_state_sample, TP, Q=False):
    '''
    Generates trajectory based on myopic policy by default, or by Q-value-based
    policy if a Q array is supplied.
    '''
    s = init_state_sample(state_space)
    states = [s]
    beta = mu(s, alpha[i]) + np.random.normal(scale=np.sqrt(sigsq[i]))
    if type(Q) == np.ndarray:
        a = np.random.choice(action_space, p = softmax(Q[s[0],s[1]], beta))
    else:
        a = np.random.choice(action_space, p = myo_policy(beta, s, TP, rewards))
    actions = [a]
    for _ in range(Ti-1):
        s = grid_step(s,a)
        states.append(s)
        beta = mu(s, alpha[i]) + np.random.normal(scale=np.sqrt(sigsq[i]))
        if type(Q) == np.ndarray:
            a = np.random.choice(action_space, p = softmax(Q[s[0],s[1]], beta))
        else:
            a = np.random.choice(action_space, p = myo_policy(beta, s, TP,
              rewards))
        actions.append(a)
    return list(multi_state_index(np.array(states))), actions

def see_trajectory(reward_map, state_seq):
    '''
    Input state seq is in index form; can transform to tuples
    '''
    state_seq = [state_space[s] for s in state_seq]
    for s in state_seq:
        sns.heatmap(reward_map)
        plt.annotate('*', (s[1]+0.2,s[0]+0.7), color='b', size=24)
        plt.show()

def make_data(alpha, sigsqs, rewards, N, Ti, state_space, action_space,
              init_state_sample, TP, m, Q=False):
    '''
    Makes a list of N trajectories based on the given parameters and the
    myopic policy by default, or the Q-value-based policy if Q is provided.
    '''
    trajectories = [
        [synthetic_traj(rewards, alpha, sigsq, i, Ti, state_space, action_space,
                       init_state_sample, TP, Q) for i in range(m)]
        for _ in range(N)
    ]
    return trajectories 

def RE_all(theta, data, TP, state_space, m, centers_x, centers_y):
    '''
    Expected reward and beta function bases (eta) for an array of states and
    actions.
    '''
    reward_est = lin_rew_func(theta, state_space, centers_x, centers_y)
    # converting int representation of states to coordinates
    data_x = data[:,0,:] // D
    data_y = data[:,0,:] % D
    arr = np.array([RESCALE*np.exp(-ETA_COEF*((data_x-1)**2+(data_y-1)**2)),
      RESCALE*np.exp(-ETA_COEF*((data_x-(D-2))**2+(data_y-(D-2))**2)),
      RESCALE*np.exp(-ETA_COEF*((data_x-1)**2+(data_y-(D-2))**2)),
      RESCALE*np.exp(-ETA_COEF*((data_x-(D-2))**2+(data_y-1)**2)),
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

def grad_terms(normals, phi, alpha, sigsq, theta, data, R_all, E_all, Ti, m):
    '''
    Computes quantities used in multiple computations of gradients.
    '''
    denom = sigsq + phi[:,1]
    sc_normals = (denom**(1/2))[:,None,None]*normals
    aE = np.einsum('ij,ijk->ik', alpha, E_all) #faster than tensordot
    mn = sigsq[:,None]*R_all + phi[:,0][:,None]*np.ones((m,Ti))
    meanvec = sc_normals + (aE + mn)[:,None,:]
    gvec = sc_normals + mn[:,None,:]
    gnorm = np.einsum('ijk,ijk->ij', gvec, gvec) #faster than tensordot, but still p slow
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

def phi_grad(phi, m, Ti, normals, denom, sigsq, glogZ_phi):
    x = (phi[:,0][:,None]*np.ones((m,Ti)))[:,None,:] + (denom**(1/2))[:,None,None]*normals
    y1 = 1/sigsq[:,None]*x.sum(axis=2)
    y2 = np.einsum('ijk,ijk->ij', normals, x)/((2*sigsq*denom**(1/2))[:,None]) - Ti/(2*denom)[:,None]
    result = -glogZ_phi - np.stack((y1, y2))
    return np.swapaxes(np.mean(result, axis=2), 0, 1)

def alpha_grad(glogZ_alpha, E_all, R_all):
    result = -glogZ_alpha + np.einsum('ijk,ik->ij', E_all, R_all)[:,None,:]
    return np.mean(result, axis=1)

def sigsq_grad(glogZ_sigsq, normals, Ti, sigsq, gnorm, denom, R_all,
                  gvec):
    q_grad = -Ti/(2*denom)
    x = -Ti/(2*sigsq) + np.einsum('ij,ij->i', R_all, R_all)
    y = np.einsum('ijk,ik->ij', normals, R_all)/(2*denom**(1/2))[:,None]
    z1 = R_all[:,None,:] + normals/(2*denom**(1/2))[:,None,None]
    z = 1/(sigsq[:,None])*np.einsum('ijk,ijk->ij', z1, gvec)
    w = 1/(2*sigsq**2)[:,None]*gnorm
    result = -glogZ_sigsq + x[:,None] + y - z + w - q_grad[:,None]
    return np.mean(result, axis=1)

def log_mean_exp(tensor):
    K = np.max(tensor, axis=2)
    expo = np.exp(tensor - K[:,:,None,:])
    return np.log(np.mean(expo,axis=2)) + K

def logZ(normals, meanvec, denom, impa, theta, data, M, TP,
            R_all, E_all, action_space, centers_x, centers_y):
    reward_est = lin_rew_func(theta, state_space, centers_x, centers_y)
    R_Z = np.swapaxes(np.array([arr_expect_reward(reward_est,
                      imp_samp_data(data, impa, j, m, Ti).astype(int),
                      TP, state_space) for j in range(M)]), 0, 1)
    lst = []
    for j in range(M):
        newdata = imp_samp_data(data, impa, j, m, Ti).astype(int)
        feat_expect = grad_lin_rew(newdata, state_space, centers_x, centers_y)
        lst.append(feat_expect)
    gradR_Z = np.swapaxes(np.array(lst), 0, 1)

    volA = len(action_space)
    bterm = np.einsum('ijk,ilk->ijlk', meanvec, R_Z)
    expo = np.exp(bterm)
    lvec = np.log(volA) + log_mean_exp(bterm)
    logZvec = lvec.sum(axis=2)

    gradR = grad_lin_rew(data, state_space, centers_x, centers_y)
    num1 = sigsq[:,None,None,None]*np.einsum('ijk,ilk->ijlk', R_Z, gradR)
    num2 = np.einsum('ijk,ilmk->ijlmk', meanvec, gradR_Z)
    expo = np.exp(bterm - np.max(bterm, axis=2)[:,:,None,:])
    num = expo[:,:,:,None,:]*(num1[:,None,:,:,:]+num2)
    numsum = num.sum(axis=2)
    den = expo.sum(axis=2)
    glogZ_theta = (numsum/den[:,:,None,:]).sum(axis=3)


    num_a = expo[:,:,:,None,:]*np.einsum('ijk,ilk->ijlk', R_Z, E_all)[:,None,:,:,:]
    numsum_a = num_a.sum(axis=2)
    glogZ_alpha = (numsum_a/den[:,:,None,:]).sum(axis=3)

    num_s = expo*np.einsum('ijk,ilk->iljk', R_Z, R_all[:,None,:] + normals/((2*denom**2)[:,None,None]))
    numsum_s = num_s.sum(axis=2)
    glogZ_sigsq = (numsum_s/den).sum(axis=2)

    num_p1 = expo*R_Z[:,None,:,:]
    num_p2 = expo*np.einsum('ijk,ilk->iljk', R_Z, normals/((2*denom**2)[:,None,None]))
    numsum_p1 = num_p1.sum(axis=2)
    numsum_p2 = num_p2.sum(axis=2)
    glogZ_phi = np.array([(numsum_p1/den).sum(axis=2), (numsum_p2/den).sum(axis=2)])
    return logZvec, glogZ_theta, glogZ_alpha, glogZ_sigsq, glogZ_phi

def theta_grad(glogZ_theta, data, state_space, R_all, E_all, sigsq, alpha,
                  centers_x, centers_y):
    '''
    Output m x d

    WORKS!!!
    '''
    gradR = grad_lin_rew(data, state_space, centers_x, centers_y)
    X = sigsq[:,None]*R_all + np.einsum('ij,ijk->ik', alpha, E_all)
    result = -glogZ_theta + np.einsum('ijk,ik->ij', gradR, X)[:,None,:]
    return np.sum(np.mean(result, axis=1), axis=0)

def cumulative_reward(s_list, cr_reps, policy, T, state_space, rewards):
    reward_list = []
    for i in range(cr_reps):
        ret = episode(s_list[i],T,policy,rewards)[2] #true reward
        reward_list.extend(ret)
    return reward_list

def evaluate_general(theta, alpha, sigsq, phi, beta, traj_data, TP, state_space,
                     action_space, B, m, M, Ti, learn_rate, reps, policy, T,
                     rewards, init_policy, init_Q, J, centers_x, centers_y,
                     cr_reps, algo_a, algo_b, random=False):
    start = datetime.datetime.now()
    s_list = [state_space[np.random.choice(len(state_space))] for _ in range(cr_reps)]
    true_rew = cumulative_reward(s_list, cr_reps, policy, T, state_space, rewards)
    plt.plot(np.cumsum(true_rew), color='b') 
    true_total = np.sum(true_rew)
    totals = [[],[]]
    cols = ['r', 'g']
    for j in range(J):
        ta = algo_a(theta, alpha, sigsq, phi, beta, traj_data, TP,
          state_space, action_space, B, m, M, Ti, learn_rate, reps, centers_x,
          centers_y, plot=False)[0]
        if random:
            tb = np.random.normal(size=d)
        else:
            tb = algo_b(theta, alpha, sigsq, phi, beta,
              traj_data, TP, state_space, action_space, B, m, M, Ti, learn_rate,
              reps, centers_x, centers_y, plot=False)[0]
        theta_stars = [ta, tb]
        for i in range(2):
            reward_est = lin_rew_func(theta_stars[i], state_space, centers_x,
              centers_y)
            est_policy = value_iter(state_space, action_space, reward_est, TP,
              GAM, 1e-5)[0]
            #Qlearn(0.5, 0.8, 0.1, Q_ITERS, 20, state_space,
            #action_space, reward_est, init_policy, init_Q)[0]
            est_rew = cumulative_reward(s_list, cr_reps, est_policy, T,
              state_space, rewards)
            plt.plot(np.cumsum(est_rew), color=cols[i])
            totals[i].append(np.sum(est_rew))
        sec = (datetime.datetime.now() - start).total_seconds()
        print(str(round((j+1)/J*100, 3)) + '% done: ' + str(round(sec / 60, 3)))
    return true_total, totals[0], totals[1]

def logZ_unif(beta, impa, theta, data, M, TP, action_space, centers_x, centers_y):
    '''
    Importance sampling approximation of logZ
    and grad logZ
    '''
    reward_est = lin_rew_func(theta, state_space, centers_x, centers_y)

    R_Z = np.swapaxes(np.array([arr_expect_reward(reward_est,
                      imp_samp_data(data, impa, j, m, Ti).astype(int),
                      TP, state_space) for j in range(M)]), 0, 1)
    lst = []
    for j in range(M):
        newdata = imp_samp_data(data, impa, j, m, Ti).astype(int)
        feat_expect = grad_lin_rew(newdata, state_space, centers_x, centers_y)
        #probs = TP[newdata[:,0], newdata[:,1]] 
        lst.append(feat_expect)
    gradR_Z = np.swapaxes(np.array(lst), 0, 1)
    
    expo = np.exp(beta[:,None,None]*R_Z)
    volA = len(action_space) # m x N x Ti
    lvec = np.log(volA*np.mean(expo,axis=1)) # for all times
    logZvec = lvec.sum(axis=1)

    num_t = expo[:,:,None,:]*beta[:,None,None,None]*gradR_Z
    num_b = expo*R_Z
    numsum_t = num_t.sum(axis=1)
    numsum_b = num_b.sum(axis=1)
    den = expo.sum(axis=1)
    glogZ_theta = ((numsum_t/den[:,None,:]).sum(axis=2)).sum(axis=0)
    glogZ_beta = (numsum_b/den).sum(axis=1)
    # This appears to approximate the true logZ for the
    # grid world with 4 actions very well!
    return logZvec, glogZ_theta, glogZ_beta # m x N; Not averaged over beta!

def beta_grad_unif(glogZ_beta, R_all):
    return -glogZ_beta + R_all.sum(axis=1)

def theta_grad_unif(data, beta, state_space, glogZ_theta, centers_x, centers_y):
    '''
    Output m x d
    '''
    gradR = grad_lin_rew(data, state_space, centers_x, centers_y) # m x d x Ti 
    return -glogZ_theta + (beta[:,None,None]*gradR).sum(axis=2).sum(axis=0) # each term is quite large

def logZ_det(beta, impa, theta, data, M, TP, R_all, E_all, action_space,
             centers_x, centers_y):
    '''
    Importance sampling approximation of logZ
    and grad logZ
    '''
    reward_est = lin_rew_func(theta, state_space, centers_x, centers_y)

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
    volA = len(action_space) # m x N x Ti
    lvec = np.log(volA*np.mean(expo,axis=1)) # for all times
    logZvec = lvec.sum(axis=1)

    num_t = expo[:,:,None,:]*beta[:,None,None,:]*gradR_Z
    num_a = np.einsum('ijk,ilk->ilk', expo*R_Z, E_all) #expo*R_Z
    numsum_t = num_t.sum(axis=1)
    den = expo.sum(axis=1)
    glogZ_theta = ((numsum_t/den[:,None,:]).sum(axis=2)).sum(axis=0)
    glogZ_alpha = (num_a/den[:,None,:]).sum(axis=2)
    # This appears to approximate the true logZ for the
    # grid world with 4 actions very well!
    return logZvec, glogZ_theta, glogZ_alpha

def alpha_grad_det(glogZ_alpha, R_all, E_all):
    return -glogZ_alpha + np.einsum('ijk,ik->ij', E_all, R_all)

def theta_grad_det(data, beta, state_space, glogZ_theta, centers_x, centers_y):
    gradR = grad_lin_rew(data, state_space, centers_x, centers_y) # m x d x Ti 
    return -glogZ_theta + np.einsum('ij,ikj->k', beta, gradR)

def GD_unif(theta, beta, g_theta, g_beta, learn_rate):
    theta = theta + learn_rate*g_theta
    beta = beta + learn_rate*g_beta
    return theta, beta

def y_t_GD_unif(theta, theta_m, beta, beta_m, t):
    return phi, beta

def y_t_nest_unif(theta, theta_m, beta, beta_m, t):
    const = (t-1)/(t+2)
    return (theta - const*(theta - theta_m),
            beta - const*(beta - beta_m))

def loglik(state_space, Ti, beta, data, TP, m, R_all, logZvec):
    logT = np.log(1/len(state_space)) + np.sum(np.log(traj_TP(data, TP, Ti, m)), axis=1)
    return -logZvec + logT + np.einsum('ij,ij->i', beta, R_all)

def AEVB(theta, alpha, sigsq, phi, beta, traj_data, TP, state_space,
         action_space, B, m, M, Ti, learn_rate, reps, centers_x, centers_y,
         plot=True):
    '''
    y_t is the function used to define modified iterate for Nesterov, if
    applicable
    
    update is e.g. GD or Adam
    '''
    impa = list(np.random.choice(action_space, M))
    N = len(traj_data)
    elbos = []
    phi_m = np.zeros_like(phi)
    theta_m = np.zeros_like(theta)
    alpha_m = np.zeros_like(alpha)
    sigsq_m = np.zeros_like(sigsq)
    t = 1
    # while error > eps:
    for _ in range(reps):
        permut = list(np.random.permutation(range(N)))
        for n in permut:
            y_phi, y_theta, y_alpha, y_sigsq = y_t_nest(phi, phi_m, theta, theta_m,
              alpha, alpha_m, sigsq, sigsq_m, t)
            data = np.array(traj_data[n]) # m x 2 x Ti
            R_all, E_all = RE_all(y_theta, data, TP, state_space, m, centers_x, centers_y)
            normals = np.random.multivariate_normal(np.zeros(Ti), np.eye(Ti), (m,B))
            meanvec, denom, gvec, gnorm = grad_terms(normals,
              y_phi, y_alpha, y_sigsq, y_theta, data, R_all, E_all, Ti, m)
            logZvec, glogZ_theta, glogZ_alpha, glogZ_sigsq, glogZ_phi = logZ(normals,
              meanvec, denom, impa, y_theta, data, M, TP, R_all, E_all, action_space, centers_x, centers_y)
          
            logprobdiff = elbo(state_space, Ti, y_sigsq, gnorm, data, TP, m, normals, R_all,
              logZvec, meanvec, denom)
            elbos.append(logprobdiff)
            #print(lp - lq)
              
            g_phi = phi_grad(y_phi, m, Ti, normals, denom, y_sigsq, glogZ_phi)
            g_theta = theta_grad(glogZ_theta, data, state_space, R_all, E_all,
              y_sigsq, y_alpha, centers_x, centers_y)
            g_alpha = alpha_grad(glogZ_alpha, E_all, R_all)
            g_sigsq = sigsq_grad(glogZ_sigsq, normals, Ti, y_sigsq, gnorm, denom,
              R_all, gvec)
          
            phi_m, theta_m, alpha_m, sigsq_m = phi, theta, alpha, sigsq
            phi, theta, alpha, sigsq = GD(y_phi, y_theta, y_alpha, y_sigsq, g_phi,
              g_theta, g_alpha, g_sigsq, learn_rate)
            
            learn_rate *= 0.99
            t += 1
    if plot:
        plt.plot(elbos)
    return theta, phi, alpha, sigsq

def AR_AEVB(theta, alpha, sigsq, phi, beta, traj_data, TP, state_space,
         action_space, B, m, M, Ti, learn_rate, reps, centers_x, centers_y,
         plot=True):
    '''
    y_t is the function used to define modified iterate for Nesterov, if
    applicable
    
    update is e.g. GD or Adam
    '''
    impa = list(np.random.choice(action_space, M))
    N = len(traj_data)
    elbos = []
    phi_m = np.zeros_like(phi)
    theta_m = np.zeros_like(theta)
    alpha_m = np.zeros_like(alpha)
    sigsq_m = np.zeros_like(sigsq)
    y_phi = phi.copy()
    y_theta = theta.copy()
    y_alpha = alpha.copy()
    y_sigsq = sigsq.copy()
    best = -np.inf
    best_phi = phi_m.copy()
    best_theta = theta_m.copy()
    best_alpha = alpha_m.copy()
    best_sigsq = sigsq_m.copy()
    tm = 1
    last_lpd = -np.inf
    # maybe don't need to resample normals every time? testing this out
    normals = np.random.multivariate_normal(np.zeros(Ti), np.eye(Ti), (m,B))
    # while error > eps:
    for _ in range(reps):
        permut = list(np.random.permutation(range(N)))
        for n in permut:#permut[20:30]:
            t = 1/2*(1 + np.sqrt(1 + 4*tm**2))
            
            data = np.array(traj_data[n]) # m x 2 x Ti
            R_all, E_all = RE_all(y_theta, data, TP, state_space, m, centers_x, centers_y)
            meanvec, denom, gvec, gnorm = grad_terms(normals,
              y_phi, y_alpha, y_sigsq, y_theta, data, R_all, E_all, Ti, m)
            logZvec, glogZ_theta, glogZ_alpha, glogZ_sigsq, glogZ_phi = logZ(normals,
              meanvec, denom, impa, y_theta, data, M, TP, R_all, E_all, action_space, centers_x, centers_y)
          
            logprobdiff = elbo(state_space, Ti, y_sigsq, gnorm, data, TP, m, normals, R_all,
              logZvec, meanvec, denom)
            elbos.append(logprobdiff)
            #print(lp - lq)
              
            g_phi = phi_grad(y_phi, m, Ti, normals, denom, y_sigsq, glogZ_phi)
            g_theta = theta_grad(glogZ_theta, data, state_space, R_all, E_all,
              y_sigsq, y_alpha, centers_x, centers_y)
            g_alpha = alpha_grad(glogZ_alpha, E_all, R_all)
            g_sigsq = sigsq_grad(glogZ_sigsq, normals, Ti, y_sigsq, gnorm, denom,
              R_all, gvec)
            
            '''
            testing out gradient clipping
            '''
            g_phi = g_phi / np.linalg.norm(g_phi)
            g_theta = g_theta / np.linalg.norm(g_theta)
            g_alpha = g_alpha / np.linalg.norm(g_alpha, 'f')
            g_sigsq = g_sigsq / np.linalg.norm(g_sigsq)
          
            phi_m, theta_m, alpha_m, sigsq_m = phi, theta, alpha, sigsq
            phi, theta, alpha, sigsq = GD(y_phi, y_theta, y_alpha, y_sigsq, g_phi,
              g_theta, g_alpha, g_sigsq, learn_rate)
            
            '''
            gradients seem to blow up... estimates swing wildly about.
            but if learn_rate is lowered, doesn't change much at all.
            
            Extremely unstable
            
            Actually stabilizes when clip gradient by norm
            '''
            
            mult = (tm - 1)/t
            y_phi = phi + mult*(phi - phi_m)
            y_theta = theta + mult*(theta - theta_m)
            #y_alpha = alpha + mult*(alpha - alpha_m)
            y_alpha = np.maximum(alpha + mult*(alpha - alpha_m), 0)
            y_sigsq = sigsq + mult*(sigsq - sigsq_m)
            
            learn_rate *= 0.99
            tm = t
            
            if logprobdiff > best:
                best = logprobdiff
                best_phi = y_phi.copy()
                best_theta = y_theta.copy()
                best_alpha = y_alpha.copy()
                best_sigsq = y_sigsq.copy()
            elif logprobdiff < last_lpd:
                y_phi = phi.copy()
                y_theta = theta.copy()
                y_alpha = alpha.copy()
                y_sigsq = sigsq.copy()
                tm = 1
                
            last_lpd = logprobdiff
            #sns.heatmap(lin_rew_func(theta, state_space, centers_x, centers_y))
            #plt.show()
    if plot:
        plt.plot(elbos)
    return best_theta, best_phi, best_alpha, best_sigsq

def ann_AEVB(theta, alpha, sigsq, phi, beta, traj_data, TP, state_space,
         action_space, B, m, M, Ti, learn_rate, reps, centers_x, centers_y,
         plot=True):
    '''
    y_t is the function used to define modified iterate for Nesterov, if
    applicable
    
    update is e.g. GD or Adam
    '''
    impa = list(np.random.choice(action_space, M))
    N = len(traj_data)
    elbos = []
    phi_m = np.zeros_like(phi)
    theta_m = np.zeros_like(theta)
    alpha_m = np.zeros_like(alpha)
    sigsq_m = np.zeros_like(sigsq)
    best = -np.inf
    best_phi = phi_m.copy()
    best_theta = theta_m.copy()
    best_alpha = alpha_m.copy()
    best_sigsq = sigsq_m.copy()
    time_since_best = 0
    t = 1
    start_lr = learn_rate
    # while error > eps:
    for _ in range(reps):
        permut = list(np.random.permutation(range(N)))
        for n in permut:
            time_since_best += 1
            y_phi, y_theta, y_alpha, y_sigsq = y_t_nest(phi, phi_m, theta,
              theta_m, alpha, alpha_m, sigsq, sigsq_m, t)
            data = np.array(traj_data[n]) # m x 2 x Ti
            R_all, E_all = RE_all(y_theta, data, TP, state_space, m, centers_x, centers_y)
            normals = np.random.multivariate_normal(np.zeros(Ti), np.eye(Ti), (m,B))
            meanvec, denom, gvec, gnorm = grad_terms(normals,
              y_phi, y_alpha, y_sigsq, y_theta, data, R_all, E_all, Ti, m)
            (logZvec, glogZ_theta, glogZ_alpha, glogZ_sigsq,
              glogZ_phi) = logZ(normals, meanvec, denom, impa, y_theta,
              data, M, TP, R_all, E_all, action_space, centers_x, centers_y)
          
            logprobdiff = elbo(state_space, Ti, y_sigsq, gnorm, data, TP, m,
              normals, R_all, logZvec, meanvec, denom)
            elbos.append(logprobdiff)
            #print(lp - lq)
              
            g_phi = phi_grad(y_phi, m, Ti, normals, denom, y_sigsq,
              glogZ_phi)
            g_theta = theta_grad(glogZ_theta, data, state_space,
              R_all, E_all, y_sigsq, y_alpha, centers_x, centers_y)
            g_alpha = alpha_grad(glogZ_alpha, E_all, R_all)
            g_sigsq = sigsq_grad(glogZ_sigsq, normals, Ti, y_sigsq,
              gnorm, denom, R_all, gvec)
          
            phi_m, theta_m, alpha_m, sigsq_m = phi, theta, alpha, sigsq
            phi, theta, alpha, sigsq = GD(y_phi, y_theta, y_alpha,
              y_sigsq, g_phi, g_theta, g_alpha, g_sigsq, learn_rate)

            if logprobdiff > best:
                best = logprobdiff
                best_phi = phi.copy()
                best_theta = theta.copy()
                best_alpha = alpha.copy()
                best_sigsq = sigsq.copy()
                time_since_best = 0
            
            learn_rate *= 0.99
            t += 1
            if time_since_best > RESET:
                phi_m = np.zeros_like(phi)
                theta_m = np.zeros_like(theta)
                alpha_m = np.zeros_like(alpha)
                sigsq_m = np.zeros_like(sigsq)
                learn_rate = start_lr
                phi += np.random.normal(scale=2*np.max(np.abs(phi)),
                  size=phi.shape)
                phi[:,1] = np.maximum(phi[:,1], 0.01)
                theta += np.random.normal(scale=2*np.max(np.abs(theta)),
                  size=theta.shape)
                alpha += np.random.normal(scale=2*np.max(np.abs(alpha)),
                  size=alpha.shape)
                sigsq += np.random.normal(scale=2*np.max(np.abs(sigsq)),
                  size=sigsq.shape)
                sigsq = np.maximum(sigsq, 0.01)
                time_since_best = 0
                #print('RESET')
    if plot:
        plt.plot(elbos)
    return best_theta, best_phi, best_alpha, best_sigsq

def MEIRL_det_pos(theta, alpha, sigsq, phi, beta, traj_data, TP, state_space,
         action_space, B, m, M, Ti, learn_rate, reps, centers_x, centers_y,
         plot=True):
    '''
    y_t is the function used to define modified iterate for Nesterov, if
    applicable
    
    update is e.g. GD or Adam
    '''
    impa = list(np.random.choice(action_space, M))
    N = len(traj_data)
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
            
            data = np.array(traj_data[n]) # m x 2 x Ti
            R_all, E_all = RE_all(y_theta, data, TP, state_space, m, centers_x, centers_y)
            beta = np.einsum('ij,ijk->ik', y_alpha, E_all)
            
            logZvec, glogZ_theta, glogZ_alpha = logZ_det(beta,
              impa, y_theta, data, M, TP, R_all, E_all, action_space, centers_x, centers_y)
          
            loglikelihood = loglik(state_space, Ti, beta, data, TP, m, R_all, logZvec).sum()
            # appears not to be maximized at true theta/alpha, why?
            lik.append(loglikelihood)
              
            g_theta = theta_grad_det(data, beta, state_space, glogZ_theta, centers_x, centers_y)
            g_alpha = alpha_grad_det(glogZ_alpha, R_all, E_all)
          
            theta_m, alpha_m, = theta, alpha
            theta = y_theta + learn_rate*g_theta
            alpha = y_alpha + learn_rate*g_alpha
            
            mult = (tm - 1)/t
            y_theta = theta + mult*(theta - theta_m)
            y_alpha = np.maximum(alpha + mult*(alpha - alpha_m), 0)
            
            learn_rate *= 0.99
            tm = t
            
            if loglikelihood > best:
                best = loglikelihood
                best_theta = y_theta.copy()
                best_alpha = y_alpha.copy()
            elif loglikelihood < last_lik:
                y_theta = theta.copy()
                y_alpha = alpha.copy()
                tm = 1
                
            last_lik = loglikelihood
    if plot:
        plt.plot(lik)
    return best_theta, best_alpha

def MEIRL_unif(theta, alpha, sigsq, phi, beta, traj_data, TP, state_space,
         action_space, B, m, M, Ti, learn_rate, reps, centers_x, centers_y,
         plot=True):
    '''
    y_t is the function used to define modified iterate for Nesterov, if
    applicable
    
    update is e.g. GD or Adam
    '''
    impa = list(np.random.choice(action_space, M))
    N = len(traj_data)
    theta_m = np.zeros_like(theta)
    beta_m = np.zeros_like(beta)
    t = 1
    # while error > eps:
    for _ in range(reps):
        permut = list(np.random.permutation(range(N)))
        for n in permut:
            y_theta, y_beta = y_t_nest_unif(theta, theta_m,
              beta, beta_m, t)
            data = np.array(traj_data[n]) # m x 2 x Ti
            R_all = RE_all(y_theta, data, TP, state_space, m, centers_x, centers_y)[0]
            logZvec, glogZ_theta, glogZ_beta = logZ_unif(y_beta, impa,
              y_theta, data, M, TP, action_space, centers_x, centers_y)
              
            g_theta = theta_grad_unif(data, y_beta, state_space, glogZ_theta, centers_x, centers_y)
            g_beta = beta_grad_unif(glogZ_beta, R_all)
          
            theta_m, beta_m = theta, beta
            theta, beta = GD_unif(y_theta, y_beta, g_theta, g_beta, learn_rate)
            beta = np.maximum(0, beta)
            
            learn_rate *= 0.99
            t += 1
    return theta, beta

def save_results():
    filename = '_'.join(str(datetime.datetime.now()).split())
    fname = filename.replace(':', '--')
    f = open(fname + '.txt', 'w')
    f.write('D = ' + str(D) + '\n')
    f.write('MOVE_NOISE = ' + str(MOVE_NOISE) + '\n')
    f.write('INTERCEPT_ETA = ' + str(INTERCEPT_ETA) + '\n')
    f.write('INTERCEPT_REW = ' + str(INTERCEPT_REW) + '\n')
    f.write('WEIGHT = ' + str(WEIGHT) + '\n')
    f.write('RESCALE = ' + str(RESCALE) + '\n')
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
    f.write('Q_ITERS = ' + str(Q_ITERS) + '\n')
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
    

    


# Initializations
#np.random.seed(1)
for seed in [20,40,60,80,100]:
    SEED_NUM = seed#100#50#80#70#60
    np.random.seed(SEED_NUM) #50) #40) #30) #20) #10)
    
    # Global params
    D=16 #8 #6x
    MOVE_NOISE = 0.05
    '''
    INTERCEPT_ETA = 3 # best for D=16
    WEIGHT = 0.2
    '''
    INTERCEPT_ETA = 0
    WEIGHT = 2
    RESCALE = 1
    RESET = 20
    COEF = 0.1
    ETA_COEF = 0.01 #0.01 #0.05 #0.1 #1
    GAM = 0.9
    '''
    Changing the above to see if making the experts suboptimal in a larger space
    makes the difference between det and unif more stark
    '''
    M = 20 # number of actions used for importance sampling
    N = 100#20 #100 #2000 # number of trajectories per expert
    J = 20#10 # should be 30....
    T = 50
    Ti = 20 # length of trajectory
    B = 50#100 # number of betas/normals sampled for expectation
    Q_ITERS = 30000#50000
    learn_rate = 0.0001
    cr_reps = 10
    reps = 1
    
    '''
    # Used in second wave of experiments, when D = 8
    centers_x = [0, D-2, 3, D-1]
    centers_y = [2, D-1, D-2, 0]
    '''
    centers_x = np.random.choice(D, D//2)
    centers_y = np.random.choice(D, D//2)
    d = D // 2 + 1
    
    
    # Making gridworld
    state_space = np.array([(i,j) for i in range(D) for j in range(D)])
    action_space = list(range(4))
    TP = transition(state_space, action_space)
    theta_true = np.random.normal(size = D // 2 + 1, scale=3)
    '''
    Trying out a more sparse-reward MDP (N = 20):
        Results (SEED 50):
            ETA_COEF 0.01: seems det and unif do roughly as well; only half as much reward
        as optimal though
            ETA_COEF 0.05: det does well about 2/3 of the time! unif much less so,
            only hits 1 out of 10. So evidently the setting where det is competitive
            is when reward is sparse, and expertise is also sparse
            * Next trying this same ETA_COEF, but with experts covering every 
            corner --> both do better, unif gets much better
            * Next trying same ETA_COEF, covering every corner, but with N=200
            --> both do much better, det gets pretty close to opt, but unif is close
            too
            * next keeping N=200, but going back to experts not covering every; weirdly,
            now unif does *better* than det, and unif only ~300 while 1000 true_tot
            * at N=1000, back to det doing slightly better
            
    Switched to SEED 60, N = 20, Ti = 20, ETA_COEF 0.05, experts cover each corner
    * result: both det and unif suck on this! the ONLY difference between case where
    both did very well is that the high-reward cluster is now in the bottom-left
    corner rather than top-left, weird.....
     - this replicates, consistently bad....
     
     
    Now trying SEED 70, same params, again reward hub is near bottom-left
    * same problem
    * when reward max is scaled up to 3.5, unif does much better:
        - 1400 for true, 600 for det, 970 for unif
    
    SEED 80 does a little better, but still not nearly as well as Seed 50; here
    the reward hub is bottom center-left
    
    seed 100 same problem
    '''
    #theta_true = np.zeros(d)
    #theta_true[0] += 5*np.random.rand() #3.5
    INTERCEPT_REW = 0
    rewards = lin_rew_func(theta_true, state_space, centers_x, centers_y)
    sns.heatmap(rewards)
    plt.show()
    # Misspecified reward bases?
    
    # Alpha vectors for the centers of the grid world
    # where each expert is closest to optimal.
    
    # # (1,1)
    # # (4,1)
    
    
    
    alpha1 = np.array([WEIGHT, 0, 0, 0, 1]) # (1,1)
    alpha2 = np.array([0, 0, WEIGHT, 0, 1]) # (1,4)
    alpha3 = np.array([0, 0, 0, WEIGHT, 1]) # (4,4)
    alpha4 = np.array([0, WEIGHT, 0, 0, 1])
    
    '''
    alpha1 = np.array([0, WEIGHT, 0, 0, 1]) # (1,1)
    alpha2 = np.array([0, 0, WEIGHT, 0, 1]) # (1,4)
    alpha3 = np.array([0, WEIGHT, 0, 0, 1]) # (4,4)
    alpha4 = np.array([0, 0, WEIGHT, 0, 1])
    '''
    
    p = alpha1.shape[0]
    m = 4
    
    '''
    sigsq1 = 25
    sigsq2 = 25
    sigsq3 = 25
    sigsq4 = 25
    '''
    
    '''
    sigsq1 = 2
    sigsq2 = 2
    sigsq3 = 2
    sigsq4 = 2
    '''
    sigsq1 = 0.1#0.01
    sigsq2 = 0.1#0.01
    sigsq3 = 0.1#0.01
    sigsq4 = 0.1#0.01
    
    ex_alphas = np.stack([alpha1, alpha2, alpha3, alpha4])
    ex_sigsqs = np.array([sigsq1, sigsq2, sigsq3, sigsq4])
    
    init_det_policy = np.random.choice([0,1,2,3], size=(D,D))
    init_policy = stoch_policy(init_det_policy, action_space)
    init_Q = np.random.rand(D,D,4)
    opt_policy, Q = value_iter(state_space, action_space, rewards, TP, 0.9, 1e-5)
    #Qlearn(0.5, 0.9, 0.1, Q_ITERS, 20, state_space,
     #         action_space, rewards, init_policy, init_Q)
    # takes about 1 min
    '''
    NOTE: SWITCHED GAM to 0.9!
    '''
    #visualize_policy(rewards, opt_policy)
    
    phi = np.random.rand(m,2)
    alpha = np.random.normal(size=(m,p), scale=0.05)
    sigsq = np.random.rand(m)
    beta = np.random.rand(m)
    theta = np.random.normal(size=d) #np.zeros_like(theta_true)
    
    traj_data = make_data(ex_alphas, ex_sigsqs, rewards, N, Ti, state_space, action_space,
                         init_state_sample, TP, m)
    boltz_data = make_data(ex_alphas, ex_sigsqs, rewards, N, Ti, state_space, action_space,
                         init_state_sample, TP, m, Q)
    # first index is n=1 to N
    # second index is expert
    # third is states, actions
    
    true_tot, det_tot_p, unif_tot_p = evaluate_general(theta, alpha, sigsq, phi, beta, traj_data,
                         TP, state_space,
                         action_space, B, m, M, Ti, learn_rate, reps, opt_policy, T,
                         rewards, init_policy, init_Q, J, centers_x, centers_y,
                         cr_reps, AR_AEVB, MEIRL_unif)
    print('true_tot = ' + str(true_tot))
    print('mean ar_tot_p = ' + str(np.mean(det_tot_p)))
    print('mean unif_tot_p = ' + str(np.mean(unif_tot_p)))
    plt.show()

'''
How is the unif model so robust???


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

ETA_COEF = 0.01, J=20, AR vs unif:
    * seed 20: AR SUCKS
'''

#regular
theta_star, phi_star, alpha_star, sigsq_star = ann_AEVB(theta, alpha, sigsq, phi, beta, traj_data, TP, state_space,
         action_space, B, m, M, Ti, learn_rate, reps, centers_x, centers_y)
theta_star_2, phi_star_2, alpha_star_2, sigsq_star_2 = AEVB(theta, alpha, sigsq, phi, beta, traj_data, TP, state_space,
         action_space, B, m, M, Ti, learn_rate, reps, centers_x, centers_y)
theta_star_p, alpha_star_p = MEIRL_det_pos(theta, alpha, sigsq, phi, beta, traj_data, TP, state_space,
         action_space, B, m, M, Ti, learn_rate, reps, centers_x, centers_y)
theta_star_AR, phi_star_AR, alpha_star_AR, sigsq_star_AR = AR_AEVB(theta, alpha,
  sigsq, phi, beta, traj_data, TP, state_space, action_space, B, m, M, Ti,
  learn_rate, reps, centers_x, centers_y)
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
theta_star_u, beta_star_u = MEIRL_unif(theta, alpha, sigsq, phi, beta, traj_data, TP, state_space,
         action_space, B, m, M, Ti, learn_rate, reps, centers_x, centers_y)
# these all seem to do terribly on seed 10, except when sigsq is 0.01 rather than 2 -
# then MEIRL_det_pos works quite well 

#boltz - FIX THIS
phi_star_b, theta_star_b, alpha_star_b, sigsq_star_b = ann_AEVB(theta, alpha, sigsq, phi, beta, boltz_data, TP, state_space,
         action_space, B, m, M, Ti, learn_rate, reps, centers_x, centers_y)
phi_star_2, theta_star_2, alpha_star_2, sigsq_star_2 = AEVB(theta, alpha, sigsq, phi, beta, boltz_data, TP, state_space,
         action_space, B, m, M, Ti, learn_rate, reps, centers_x, centers_y)
theta_star_p, alpha_star_p = MEIRL_det_pos(theta, alpha, sigsq, phi, beta, boltz_data, TP, state_space,
         action_space, B, m, M, Ti, learn_rate, reps, centers_x, centers_y)
phi_star_AR, theta_star_AR, alpha_star_AR, sigsq_star_AR = AR_AEVB(theta, alpha, sigsq, phi, beta, boltz_data, TP, state_space,
         action_space, B, m, M, Ti, learn_rate, reps, centers_x, centers_y)
theta_star_u, beta_star_u = MEIRL_unif(theta, alpha, sigsq, phi, beta, boltz_data, TP, state_space,
         action_space, B, m, M, Ti, learn_rate, reps, centers_x, centers_y)

# see_trajectory(rewards, np.array(traj_data[0])[0,0])

# this works p well, when true sigsq is set to 2 and the trajectories come
# from the truly specified model
dumb_data = make_data_myopic(alpha_star_2, sigsq_star_2, lin_rew_func(theta_star_2,
                            state_space, centers_x, centers_y), N, Ti, state_space, action_space,
                            init_state_sample, TP, m)

#sns.heatmap(lin_rew_func(theta_true, state_space, centers_x, centers_y))

'''
Testing against constant-beta model
'''

theta_s, beta_s = MEIRL_unif(theta, beta, traj_data, TP, state_space,
         action_space, B, m, M, Ti, learn_rate, 1, GD_unif, centers_x, centers_y)

true_tot, AEVB_tot, unif_tot = evaluate_vs_uniform(theta, alpha, sigsq, phi, beta, TP, 1, opt_policy, 30,
                        state_space, action_space, rewards, init_policy,
                        init_Q, J, B, m, M, Ti, learn_rate, traj_data, centers_x, centers_y, cr_reps)
# Using AR_AEVB as the inner loop, this works p well on D=8

true_tot, AEVB_tot, det_tot = evaluate_vs_det(theta, alpha, sigsq, phi, beta, TP, 1, opt_policy, 30,
                        state_space, action_space, rewards, init_policy,
                        init_Q, J, B, m, M, Ti, learn_rate, traj_data, centers_x, centers_y, cr_reps)

true_tot, det_tot_p, unif_tot_p = evaluate_det_vs_unif(theta, alpha, sigsq, phi, beta, TP, 1, opt_policy, T,
                        state_space, action_space, rewards, init_policy,
                        init_Q, J, B, m, M, Ti, learn_rate, traj_data,
                        centers_x, centers_y, cr_reps)

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

true_tot_b, AEVB_tot_b, unif_tot_b = evaluate_vs_uniform(theta, alpha, sigsq, phi, beta, TP, 1, opt_policy, 30,
                        state_space, action_space, rewards, init_policy,
                        init_Q, J, B, m, M, Ti, learn_rate, boltz_data, centers_x, centers_y, cr_reps)
# Works robustly well! This is on data where the demonstrators are Q-softmaxing,
# not next-step reward!

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

tr_tot, det_tot, ra_tot = evaluate_det_vs_random(theta, alpha, sigsq, phi, TP, 1, opt_policy, T,
                        state_space, action_space, rewards, init_policy,
                        init_Q, J, B, m, M, Ti, learn_rate, traj_data, centers_x, centers_y, cr_reps)

'''
To do:
    * misspec of reward function?
    * misspec of beta mean?
    * batch GD over trajectories?
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
     * Write function that stores the hyperparams/settings and saves results from
     evaluation
     * Compare performance for different split of steps in Ti vs N (length vs
     number of trajectories)
     
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
'''
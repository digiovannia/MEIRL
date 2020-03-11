import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import random

'''
A very simple dxd gridworld to test out multiple-experts IRL.
Agent can move in cardinal directions; moves in intended direction
with prob 90%, else uniform direction. If moves in illegal
direction, stays in place.

Will find optimal Q values by Q-learning.

Actions: 0 = up, 1 = right, 2 = down, 3 = left
'''

np.random.seed(1)

# Global params
D=6
MOVE_NOISE = 0.05
INTERCEPT = 10
WEIGHT = 2
N = 10
Ti = 50
B = 20

# Making gridworld
state_space = np.array([(i,j) for i in range(D) for j in range(D)])
action_space = list(range(4))
rewards = np.ones((D,D))
rewards[0,0] = 5
rewards[D-1,D-1] = 5
rewards[0,D-1] = -5
rewards[D-1,0] = -5
#sns.heatmap(rewards)

def manh_dist(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def act_to_coord(a):
    d = {0: (-1, 0), 1: (0, 1), 2: (1, 0),
         3: (0,-1)}
    return d[a]

def state_index(tup):
    return D*tup[0] + tup[1]

def multi_state_index(states):
    return D*states[:,0]+states[:,1]

def state_tuples_to_num(state_space, s):
    '''
    will need to fix
    '''
    return [state_space.index(state) for state in s]

def transition(state_space, action_space):
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

TP = transition(state_space, action_space)

def coord(x):
    return min(max(x, 0), D-1)

def grid_step(s, a):
    '''
    Given current state s and action a, returns resulting
    state.
    '''
    flip = np.random.rand()
    if flip < MOVE_NOISE:
        a = np.random.choice([0,1,2,3])
    '''
    if a == 0:
        new_state = (coord(s[0]-1), s[1])
    if a == 1:
        new_state = (s[0], coord(s[1]+1))
    if a == 2:
        new_state = (coord(s[0]+1), s[1])
    if a == 3:
        new_state = (s[0], coord(s[1]-1))
    '''
    new_state = s + act_to_coord(a)
    return np.minimum(np.maximum(new_state, 0), D-1)

def episode(s,T,policy,rewards,step_func,a=-1):
    '''
    Given any generic state, time horizon, policy, and
    reward structure indexed by the state, generates an
    episode starting from that state.

    step_func defines how a resulting state is generated
    from current state and action
    '''
    states = [s]
    if a < 0:
        a = np.random.choice(action_space, p=policy[tuple(s)])
    actions = [a]
    reward_list = [0]
    for _ in range(T-1):
        s = step_func(s,a)
        states.append(s)
        r = rewards[tuple(s)]
        reward_list.append(r)
        a = np.random.choice(action_space, p=policy[tuple(s)])
        actions.append(a)
    reward_list.append(rewards[tuple(s)])
    return np.array(states), actions, reward_list

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

# may delete this...
def MCES(gam, tol, T, state_space, action_space, rewards,
         policy, Q):
    '''
    Every-visit Monte Carlo w/ exploring starts to estimate Q*.
    Returns the corresponding optimal policy and Q*.
    '''
    counts = np.zeros((D,D,4))
    change = np.Inf
    while change > tol:
        change = 0
        # Random start
        # Given state space represented as a list
        s = state_space[np.random.choice(len(state_space))]
        a = np.random.choice(action_space)
        states, actions, reward_list = episode(s,T,policy,
          rewards,grid_step,a=a)
        G = 0
        for t in range(T-1,-1,-1):
            G = gam*G + reward_list[t+1]
            st = states[t]
            at = actions[t]
            old_Q = Q[st[0],st[1],at]
            Q[st[0],st[1],at] *= counts[st[0],st[1],at]
            counts[st[0],st[1],at] += 1
            Q[st[0],st[1],at] += G
            Q[st[0],st[1],at] /= counts[st[0],st[1],at]
            change = max(change, abs(Q[st[0],st[1],at] - old_Q))
            policy[st[0],st[1]] *= 0
            policy[st[0],st[1], np.argmax(Q[st[0],st[1]])] = 1
    policy *= 0
    for i in range(D):
        for j in range(D):
            policy[i,j,np.argmax(Q[i,j])] = 1
    return policy, Q

def visualize_policy(rewards, policy):
    '''
    Heatmap representing the input policy in the state
    space with the input rewards.
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

def eta(st):
    return np.array([abs(st[0]-1), abs(st[1]-1), abs(st[0]-4),
                     abs(st[1]-4), INTERCEPT])

def mu(s, alpha):
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
    Computes beta for whole state space.
    '''
    noise = np.random.normal(loc=0,scale=np.sqrt(sigsq),
                             size=(D,D))
    muvals = mu_all(alpha)
    return muvals + noise

def locally_opt(Q_star, alpha, sigsq):
    '''
    Given optimal Q* and some "state of expertise",
    returns a stochastic policy for an agent whose
    beta is ideal centered around that state.

    high beta = high expertise

    CHANGE THIS
    '''
    beta = beta_func(alpha, sigsq)
    policy = softmax(Q_star, beta, 2)
    det_pol = np.argmax(policy, axis=2)
    return policy, det_pol


'''
Going to test on trajectories from Q* model rather than one-step,
will see how bad...
'''

# Misspecified reward bases?

# Alpha vectors for the centers of the grid world
# where each expert is closest to optimal.
alpha1 = np.array([-WEIGHT, -WEIGHT, 0, 0, 1]) # (1,1)
alpha2 = np.array([-WEIGHT, 0, 0, -WEIGHT, 1]) # (1,4)
alpha3 = np.array([0, -WEIGHT, -WEIGHT, 0, 1]) # (4,1)
alpha4 = np.array([0, 0, -WEIGHT, -WEIGHT, 1]) # (4,4)
p = alpha1.shape[0]
d = 5
m = 4

sigsq1 = 25
sigsq2 = 25
sigsq3 = 25
sigsq4 = 25

ex_alphas = np.stack([alpha1, alpha2, alpha3, alpha4])
ex_sigsqs = np.array([sigsq1, sigsq2, sigsq3, sigsq4])

# Example of policy for expert who does
# well in bottom right corner but not
# elsewhere.

np.random.seed(1)
init_det_policy = np.random.choice([0,1,2,3], size=(D,D))
init_policy = stoch_policy(init_det_policy, action_space)
init_Q = np.random.rand(D,D,4)
policy, Q = Qlearn(0.5, 0.8, 0.1, 10000, 20, state_space,
          action_space, rewards, init_policy, init_Q)
          # This seems to converge robustly to optimal policy,
          # although Q values have some variance in states where
          # it doesn't really matter
#policy, Q = doubleQ(0.5, 0.8, 0.05, 5000, 20, state_space,
#          action_space, rewards, init_policy, init_Q)
visualize_policy(rewards, policy)
pol1 = locally_opt(Q, alpha1, sigsq1)[0]
pol2 = locally_opt(Q, alpha2, sigsq2)[0]
pol3 = locally_opt(Q, alpha3, sigsq3)[0]
pol4 = locally_opt(Q, alpha4, sigsq4)[0]







###### Phase 2 ######

# Initializations
phi = np.random.rand(m,2)
alpha = np.random.normal(size=(m,p))
sigsq = np.random.rand(m)
theta = np.random.normal(size=d)

def radial(s, c):
    return np.exp(-5*((s[0]-c[0])**2+(s[1]-c[1])**2))

def arr_radial(s, c):
    return np.exp(-5*((s[:,0]-c[0])**2+(s[:,1]-c[1])**2))

def linear_reward(theta, st):
    psi = np.array([radial(st, (0,0)),
                    radial(st,(D-1,D-1)),
                    radial(st, (0,D-1)),
                    radial(st, (D-1,0)),
                     INTERCEPT])
    return np.dot(psi, theta)

def psi_all_states(state_space):
    # d x D**2
    return np.array([arr_radial(state_space, (0,0)),
                    arr_radial(state_space,(D-1,D-1)),
                    arr_radial(state_space, (0,D-1)),
                    arr_radial(state_space, (D-1,0)),
                     INTERCEPT*np.ones(len(state_space))])

def vec_linear_reward(theta, s):
    pass

def lin_rew_func(theta, state_space):
    '''
    Using radial features above and theta below,
    gives a very close approximation to the true
    RF.

    May want to vectorize?
    '''
    return np.reshape(theta.dot(psi_all_states(state_space)), (D, D))

def expect_reward(rewards, st, at, TP, state_space):
    '''
    fix?
    '''
    si = state_space.index(st)
    return TP[si, at].dot(np.ravel(rewards))

def vec_expect_reward(rewards, s, a, TP, state_space):
    '''
    s is a list of state indices
    a is list of actions
    '''
    return TP[s, a].dot(np.ravel(rewards))
    
def arr_expect_reward(rewards, data, TP, state_space):
    '''
    data is m x 2 x Ti
    '''
    return TP[data[:,0], data[:,1]].dot(np.ravel(rewards)) #m x Ti

def grad_lin_rew(data, state_space):
    '''
    Input (data) is m x 2 x Ti

    Output should be m x d x Ti

    Each column is E(s'|st,at)(psi(s')) where psi is
    feature

    Looks correct!
    '''
    probs = TP[data[:,0], data[:,1]] # m x Ti x D**2
    return np.swapaxes(probs.dot(psi_all_states(state_space).transpose()), 1, 2)

def init_state_sample(state_space):
    '''
    Returns initial state; uniform
    '''
    s = state_space[np.random.choice(len(state_space))]
    return s

def Q_est(theta_h, gam, T, state_space,
          action_space, init_policy, init_Q, Qlearn=False,
          tol=0.0001, rate=0.5, eps=0.05, K=10000):
    '''
    Based on an estimate of theta, estimates Q by doing MCES or
    Q-learning with respect to reward induced by theta.
    '''
    reward_str = lin_rew_func(theta_h, state_space)
    if Qlearn:
        Q_h = Qlearn(rate, gam, eps, K, T, state_space,
          action_space, reward_str, init_policy, init_Q)[1]
    else:
        Q_h = MCES(gam, tol, T, state_space, action_space,
                   reward_str, init_policy, init_Q)[1]
    return rewards, Q_h

def synthetic_traj(rewards, policy, Ti, state_space, action_space, init_state_sample):
    '''
    Generate a fake trajectory based on given
    policy. Need to separately generate
    variational distribution parameters if want to
    use those.
    '''
    s = init_state_sample(state_space)
    a = np.random.choice(action_space, p=policy[tuple(s)])
    states, actions, reward_list = episode(s,Ti,policy,rewards,
                              grid_step,a=a)
    states = list(multi_state_index(states))
    return states, actions, reward_list

def see_trajectory(reward_map, state_seq):
    '''
    Input state seq is in index form; can transform to tuples
    '''
    state_seq = [state_space[s] for s in state_seq]
    for s in state_seq:
        sns.heatmap(reward_map)
        plt.annotate('*', (s[1]+0.2,s[0]+0.7), color='b', size=24)
        plt.show()

def make_data(Q, alphas, sigsqs, reward_str, N, Ti):
    '''
    Creates N trajectories each for local experts, given Q,
    alphas, sigsqs, rewards (which may be "true" or
    based on current guesses!)
    '''
    policies = [locally_opt(Q, alphas[i],
                            sigsqs[i])[0] for i in range(m)]
    trajectories = [
        [synthetic_traj(reward_str, policies[i], Ti, state_space,
                        action_space, init_state_sample)[:2] for i in range(m)]
        for _ in range(N)
    ]
    # first index is N
    # second is m
    return trajectories 

traj_data = make_data(Q, ex_alphas, ex_sigsqs, rewards, N, Ti)
data = np.array(traj_data[0]) # for testing
# first index is n=1 to N
# second index is expert
# third is states, actions

def sample_MV_beta(i, phi, alpha, sigsq, theta, s, a, TP,
                   state_space, B):
    '''
    s is list of indices

    may delete
    '''
    reward_est = lin_rew_func(theta, state_space)
    R = vec_expect_reward(reward_est, s, a, TP, state_space)
    E = eta_mat(state_space[s])
    mui = sigsq[i]*R + E.transpose().dot(alpha[i]) + phi[i,0]*np.ones(R.shape[0])
    Covi = (sigsq[i]+phi[i,1])*np.eye(R.shape[0])
    return np.random.multivariate_normal(mui, Covi, B)

def eta_mat(data):
    #VECTORIZE!
    arr = np.array([abs(data[:,0,:] // 6 - 1),
                    abs(data[:,0,:] % 6 - 1),
                    abs(data[:,0,:] // 6 - 4),
                    abs(data[:,0,:] % 6 - 4),
                    INTERCEPT*np.ones(data[:,0,:].shape)])

    return np.swapaxes(arr, 0, 1)

def RE_all(theta, data, TP, state_space, m):
    reward_est = lin_rew_func(theta, state_space)
    R_all = arr_expect_reward(reward_est, data, TP, state_space) # m x Ti
    E_all = eta_mat(data)
    return R_all, E_all

def sample_all_MV_beta(phi, alpha, sigsq, theta, R_all, E_all, data, TP,
                       state_space, B, m):
    '''
    For each expert, get B samples of betas from variational posterior.
    Assumes the input data consists of one trajectory for each expert.

    Output is m x B x Ti
    '''
    mus = sigsq[:,None]*R_all + np.einsum('ijk,ij->ik', E_all, alpha) + phi[:,0][:,None]*np.ones(R_all.shape[:2])
    Covs = np.einsum('i,jk->ijk', sigsq+phi[:,1], np.eye(R_all.shape[1]))

    #mus = np.array([sigsq[i]*R_all[i] + E_all[i].transpose().dot(alpha[i]) + phi[i,0]*np.ones(R_all[i].shape[0]) for i in range(m)])
    #Covs = np.array([(sigsq[i]+phi[i,1])*np.eye(R_all[i].shape[0]) for i in range(m)])
    return np.array([np.random.multivariate_normal(mus[i], Covs[i], B) for i in range(m)])

'''
Need to average over sample betas in the
following four functions
'''

def rho(state_space):
    return 1/len(state_space)

def uniform_action_sample(action_space, M):
    return list(np.random.choice(action_space, M))

def imp_samp_data(data, impa, j, m, Ti):
    actions = impa[j]*np.ones((m,Ti))
    out = np.stack((data[:,0,:], actions))
    return np.swapaxes(out, 0, 1)

def logZ(betas, impa, theta, data, M, TP, action_space):
    '''
    Importance sampling approximation of logZ
    and grad logZ

    May want to vectorize vec_expect_reward better to avoid 2 for-loops here
    '''
    reward_est = lin_rew_func(theta, state_space)

    #R_all = np.array([[vec_expect_reward(reward_est, data[i][0],
    #                  [impa[j]]*Ti, TP, state_space) for j in range(M)]
    #                  for i in range(m)])
    R_Z = np.swapaxes(np.array([arr_expect_reward(reward_est,
                      imp_samp_data(data, impa, j, m, Ti).astype(int),
                      TP, state_space) for j in range(M)]), 0, 1)
    '''
    R_all = np.array([vec_expect_reward(reward_est, data[i][0],
                      data[i][1], TP, state_space) for i in range(m)])
    '''
    expo = np.exp(np.einsum('ijk,ilk->ijlk', betas, R_Z))
    volA = len(action_space) # m x N x Ti
    lvec = np.log(volA*np.mean(expo,axis=2)) # for all times
    logZvec = lvec.sum(axis=2)
    # This appears to approximate the true logZ for the
    # grid world with 4 actions very well!
    '''
    gradlinrew

    probs = TP[data[:,0], data[:,1]] # m x Ti x D**2
    return np.swapaxes(probs.dot(psi_all_states(state_space).transpose()), 1, 2)
    '''
    return logZvec, glogZ # m x N; Not averaged over beta!

def traj_TP(data, TP, Ti, m):
    '''
    data = m x 2 x Ti
    output = m x (Ti-1)

    Computes TPs for (s1, a1) to s2, ..., st-1, at-1 to st
    '''
    s2_thru_sTi = TP[data[:,0,:(Ti-1)],data[:,1,:(Ti-1)]]
    return s2_thru_sTi[np.arange(m)[:,None], np.arange(Ti-1), data[:,0,1:]]

impa = uniform_action_sample(action_space, M)
R_all, E_all = RE_all(theta, data, TP, state_space, m)
betas = sample_all_MV_beta(phi, alpha, sigsq, theta, R_all, E_all,
                           data, TP, state_space, B, m)
logZvec = logZ(betas, impa, theta, data, M, TP, action_space)

def grad_terms(betas, phi, alpha, sigsq, theta, data, R_all,
             E_all, Ti, logZvec, m):
    '''
    Computes quantities used in multiple computations of gradients.
    '''
    one = np.ones(Ti)
    aE = np.einsum('ij,ijk->ik', alpha, E_all)
    vec = betas - (sigsq[:,None]*R_all + aE + phi[:,0][:,None]*np.ones((m,Ti)))[:,None,:]
    denom = sigsq + phi[:,1]
    vecnorm = np.einsum('ijk,ijk->ij', vec, vec)  
    gvec = betas - aE[:,None,:]
    gnorm = np.einsum('ijk,ijk->ij', gvec, gvec)
    return one, aE, vec, denom, vecnorm, gvec, gnorm

def logp(state_space, Ti, sigsq, gnorm, data, TP, m, betas, R_all, logZvec):
    p1 = np.log(rho(state_space)) - Ti/2*np.log(2*np.pi*sigsq)[:,None] - 1/(2*sigsq)[:,None]*gnorm
    logT = np.log(traj_TP(data, TP, Ti, m))
    p2 = np.einsum('ijk,ik->ij', betas, R_all) - logZvec + np.sum(logT, axis=1)[:,None]
    return p1 + p2

def logq(Ti, denom, vecnorm):
    return -Ti/2*np.log(2*np.pi*denom)[:,None] - vecnorm/((2*denom)[:,None])

def phi_grad(vec, one, denom, vecnorm, lp, lq):
    '''
    Output is m x 2; expectation is applied

    Will need to make sure feasible
    '''
    num1 = vec.dot(one)
    phigrad1 = num1/denom[:,None]
    phigrad2 = -Ti/(2*denom)[:,None] + vecnorm/((2*denom**2)[:,None])
    logdens = lp - lq
    return np.array([(phigrad1 * logdens).mean(axis=1), (phigrad2 * logdens).mean(axis=1)]).transpose()

def alpha_grad(E_all, gvec, vec, denom, lp, lq):
    '''
    Output m x p
    '''
    p1 = 1/sigsq[:,None,None]*np.einsum('ijk,ilk->ilj', E_all, gvec)
    p2 = 1/denom[:,None,None]*np.einsum('ijk,ilk->ilj', E_all, vec)
    return np.mean(p1 + p2*(lp - lq)[:,:,None], axis=1)

def sigsq_grad(Ti, sigsq, gnorm, denom, R_all, vec, lp, lq, vecnorm):
    '''
    Output m x 1

    Feasible!
    '''
    p1 = -Ti/(2*sigsq[:,None]) + gnorm/(2*sigsq[:,None]**2)
    p2 = -Ti/(2*denom[:,None]) + np.einsum('ik,ilk->il', R_all, vec)/denom[:,None] + vecnorm/(2*denom[:,None]**2)
    return np.mean(p1 + p2*(lp - lq), axis=1)

def theta_grad(data, state_space, denom, vec, logZvec):
    '''
    Output m x d
    '''
    gradR = grad_lin_rew(data, state_space) # m x d x Ti 
    p1 = ...
    p2 = (sigsq/denom)[:,None,None] * np.einsum('ijk,ilk->ijl', gradR, vec)
    return np.mean(p1 + p2*(lp - lq)[:,:,None], axis=1)

def AEVB(theta, alpha, sigsq, phi, traj_data, TP, state_space,
         action_space, B, m, M):
    '''
    Need the expert trajectories

    1) Init theta, alpha, sigsq, phi
    2) 
    '''
    impa = uniform_action_sample(action_space, M)
    for n in range(N):
        data = np.array(traj_data[n]) # m x 2 x Ti
        R_all, E_all = RE_all(theta, data, TP, state_space, m)
        betas = sample_all_MV_beta(phi, alpha, sigsq, theta, R_all, E_all,
                                          data, TP, state_space, B, m)
        logZvec, glogZ = logZ(betas, impa, theta, data, M, TP, action_space)
    pass

def grad_log_Z(theta, ):
    '''
    Uses a vectorized form to compute gradient of 
    log Z when tractable, e.g. in this case where we 
    can sum over actions.
    '''
    # Compute Num by summing matrices over actions,
    # where each col of the matrix corresponds to a time
    # step
    Num = np.array(d, len()) 
    den = ... 
    Ti = len(...)
    return (Num/den).dot(np.ones(Ti))

theta = np.array([4, 4, -6, -6, 0.1])
sns.heatmap(lin_rew_func(theta, state_space))

# Testing
theta_h = np.random.rand(5)
alpha_h = np.random.rand(5)
sigsq_h = np.random.rand()
# MCES(0.8, 0.00001, 50, state_space, action_space,
#                 rewards, init_policy, init_Q)
reward_est, Q_h = Q_est(theta_h, 0.8, 50, state_space, action_space,
                               init_policy, init_Q, tol=0.0001)
                        # FIX THIS 
test_policy = locally_opt(Q_h, alpha_h, sigsq_h)[0]
synth = synthetic_traj(reward_est, test_policy,
                       10, state_space, action_space, init_state_sample)
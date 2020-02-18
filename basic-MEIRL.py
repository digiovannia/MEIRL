import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import random

'''
A very simple dxd gridworld to test out multiple-experts IRL.
Agent can move in cardinal directions; moves in intended direction
with prob 90%, else uniform direction. If moves in illegal
direction, stays in place.

Will find optimal Q values by MC.

Actions: 0 = up, 1 = right, 2 = down, 3 = left
'''

D=6
MOVE_NOISE = 1

state_space = [(i,j) for i in range(D) for j in range(D)]
action_space = list(range(4))

#rewards = np.array([[1,3,-1],[-1,0,0],[-1,5,2]])
#rewards = np.random.rand(d,d)
rewards = np.ones((D,D))
rewards[0,0] = 5
rewards[D-1,D-1] = 5
rewards[0,D-1] = -5
rewards[D-1,0] = -5
#fig, ax = plt.subplots()
#im = ax.imshow(rewards)
#fig.tight_layout()
#sns.heatmap(rewards)

def coord(x):
    return min(max(x, 0), D-1)

def grid_step(s, a):
    '''
    s is a tuple for coords
    '''
    flip = np.random.rand()
    if flip > MOVE_NOISE: #0.9:
        a = np.random.choice([0,1,2,3])
    if a == 0:
        new_state = (coord(s[0]-1), s[1])
    if a == 1:
        new_state = (s[0], coord(s[1]+1))
    if a == 2:
        new_state = (coord(s[0]+1), s[1])
    if a == 3:
        new_state = (s[0], coord(s[1]-1))
    return new_state

def episode(s,T,policy,rewards,step_func,a=-1):
    '''
    Given any generic state, time horizon, policy, and reward
    structure indexed by the state, generates an episode starting
    from that state.

    step_func defines how a resulting state is generated from current
    state and action
    '''
    states = [s]
    if a >= 0:
        actions = [a]
    else:
        actions = [np.random.choice(action_space, p=policy[s])]
    reward_list = [0]
    for _ in range(1,T):
        s = step_func(s,a)
        states.append(s)
        r = rewards[s]
        reward_list.append(r)
        a = np.random.choice(action_space, p=policy[s])
        actions.append(a)
    reward_list.append(rewards[s])
    return states, actions, reward_list

def stoch_policy(det_policy, action_space):
    x = np.repeat(range(D), D)
    y = np.tile(range(D), D)
    z = np.ravel(det_policy)
    out = np.zeros((D,D,len(action_space)))
    out[x,y,z] = 1
    return out

def draw_state(state_space):
    pass

def MCES(gam, tol, T, state_space, action_space, rewards, policy, Q):
    '''
    Monte Carlo with exploring starts to estimate Q*. Might want
    to make this more modular, to allow for general state space?
    Currently only works for 2d states + 1d actions
    '''
    counts = np.zeros((D,D,4))
    change = np.Inf
    while change > tol:
        change = 0
        # random start
        # Given state space represented as a list
        s = state_space[np.random.choice(len(state_space))]
        a = action_space[np.random.choice(len(action_space))]
        states, actions, reward_list = episode(s,T,policy,rewards,grid_step,a=a)
        G = 0
        for t in range(T-1,-1,-1):
            G = gam*G + reward_list[t+1]
            st = states[t]
            at = actions[t]
            # leaving out the first-visit check
            old_Q = Q[st[0],st[1],at]
            Q[st[0],st[1],at] *= counts[st[0],st[1],at]
            counts[st[0],st[1],at] += 1
            Q[st[0],st[1],at] += G
            Q[st[0],st[1],at] /= counts[st[0],st[1],at]
            change = max(change, abs(Q[st[0],st[1],at] - old_Q))
            #policy[st[0],st[1]] = np.argmax(Q[st[0],st[1]])
            policy[st[0],st[1]] *= 0
            policy[st[0],st[1], np.argmax(Q[st[0],st[1]])] = 1
    policy *= 0
    for i in range(D):
        for j in range(D):
            #policy[i,j] = np.argmax(Q[i,j])
            policy[i,j,np.argmax(Q[i,j])] = 1
    return policy, Q

def visualize_policy(rewards, policy):
    pol = np.argmax(policy, axis=2)
    sns.heatmap(rewards)
    for x in range(D):
        for y in range(D):
            # draw arrows
            dx = (pol[x,y] % 2)*0.2*(2 - pol[x,y])
            dy = ((pol[x,y]-1) % 2)*0.2*(pol[x,y] - 1)
            plt.arrow(y+0.5,x+0.5,dx,dy,head_width=0.1,color="black",alpha=0.9)

# Some notes from the MC experiments:
# * longer time horizon isn't always better for learning
#   optimal policy, oddly
# * seems decreasing gamma makes learning more efficient, which
#   I guess makes sense by introducing some artificial urgency;
#   but there is such a thing as too low as well

def eps_greedy(Q, eps, action_space):
    best = np.argmax(Q, axis=2)
    hard = stoch_policy(best, action_space)
    hard += eps/(len(action_space)-1)*(1 - hard) - eps*hard
    return hard

def Qlearn(rate, gam, eps, K, T, state_space,
          action_space, rewards, policy, Q):
    #beta = 5*np.ones((d,d))
    for _ in range(K):
        st = state_space[np.random.choice(len(state_space))]
#       a = action_space[np.random.choice(len(action_space))]
        for _ in range(T):
            policy = eps_greedy(Q, eps, action_space) # may replace with eps-greedy?
            at = np.random.choice(action_space, p=policy[st])    
            sp = grid_step(st,at)
            rt = rewards[sp]
            Q[st[0],st[1],at] += rate*(rt + gam*np.max(Q[sp[0],sp[1]]) - Q[st[0],st[1],at])
            st = sp
    policy *= 0
    for i in range(D):
        for j in range(D):
            #policy[i,j] = np.argmax(Q[i,j])
            policy[i,j,np.argmax(Q[i,j])] = 1
    return policy, Q

def doubleQ(rate, gam, eps, K, T, state_space,
          action_space, rewards, policy, Q):
    #beta = 5*np.ones((d,d))
    Q1 = Q
    Q2 = Q.copy()
    for _ in range(K):
        st = state_space[np.random.choice(len(state_space))]
#       a = action_space[np.random.choice(len(action_space))]
        for _ in range(T):
            policy = eps_greedy(Q1 + Q2, eps, action_space) # may replace with eps-greedy?
            at = np.random.choice(action_space, p=policy[st])    
            sp = grid_step(st,at)
            rt = rewards[sp]
            if np.random.rand() < 0.5:
                Q1[st[0],st[1],at] += rate*(rt + gam*Q2[sp[0],sp[1],np.argmax(Q1[sp[0],sp[1]])] - Q1[st[0],st[1],at])
            else:
                Q2[st[0],st[1],at] += rate*(rt + gam*Q1[sp[0],sp[1],np.argmax(Q2[sp[0],sp[1]])] - Q2[st[0],st[1],at])                
            st = sp
    policy *= 0
    for i in range(D):
        for j in range(D):
            #policy[i,j] = np.argmax(Q[i,j])
            policy[i,j,np.argmax(Q1[i,j]+Q2[i,j])] = 1
    return policy, (Q1+Q2)/2


# Misspecified reward bases?
INTERCEPT = 10
WEIGHT = 2#5
# weight of 5 gives *very* suboptimal...

def eta(s):
    return np.array([abs(s[0]-1),
                     abs(s[1]-1),
                     abs(s[0]-4),
                     abs(s[1]-4),
                     INTERCEPT])

# Alpha vectors for each of the centers.
# (1,1)
alpha1 = np.array([-WEIGHT, -WEIGHT, 0, 0, 1])
# (1,4)
alpha2 = np.array([-WEIGHT, 0, 0, -WEIGHT, 1])
# (4,1)
alpha3 = np.array([0, -WEIGHT, -WEIGHT, 0, 1])
# (4,4)
alpha4 = np.array([0,0,-WEIGHT, -WEIGHT, 1])
p = alpha1.shape[0]
d = 5
m = 4

# seems to work best for getting decent noise
# without losing the signal totally
sigma1 = 5
sigma2 = 5
sigma3 = 5
sigma4 = 5

exp_alphas = np.stack([alpha1, alpha2, alpha3, alpha4])
exp_sigmas = np.array([sigma1, sigma2, sigma3, sigma4])

def mu(s, alpha):
    return np.dot(eta(s), alpha)

def mu_function(alpha):
    muvals = np.zeros((D,D))
    for i in range(D):
        for j in range(D):
            muvals[i,j] = mu((i,j), alpha)
    return muvals

def plot_mus(alpha):
    return sns.heatmap(mu_function(alpha))

def softmax(v, beta, axis=False):
    if axis:
        # w = np.exp(beta[:,:,None]*v)
        x = beta[:,:,None]*v
        w = np.exp(x - np.max(x, axis=axis)[:,:,None])
        # could try subtracting max?
        z = np.sum(w, axis=axis)
        return np.divide(w, z[:,:,None])
    else:
        w = np.exp(beta*v)
        z = np.sum(w)
        return w / z

def beta_func(alpha, sigma):
    noise = np.random.normal(loc=0,scale=sigma,
                             size=(D,D))
    muvals = mu_function(alpha)
    return muvals + noise

def locally_opt(Q_star, alpha, sigma):
    '''
    Given optimal Q* and some "state of expertise",
    returns a stochastic policy for an agent whose
    beta is ideal centered around that state.

    high beta = high expertise
    '''
    beta = beta_func(alpha, sigma)
    policy = softmax(Q_star, beta, 2)
    det_pol = np.argmax(policy, axis=2)
    return policy, det_pol

# Example of policy for expert who does
# well in bottom right corner but not
# elsewhere.

#policy, Q = MCES(0.8, 0.0001, 50, state_space, action_space,
#                 rewards, init_policy, init_Q)
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
pol4 = locally_opt(Q, alpha4, sigma4)[1]
visualize_policy(rewards, pol4)







###### Phase 2 ######

def variational(psi):
    '''
    Based on parameters psi_a, psi_s, and psi_t,
    generates alpha, sigma, and theta.

    m = number of experts
    p = dim(alpha)
    d = dim(theta)

    psi_a = mean and variance of distribution of alphas
    psi_s = vector of 1/means of exponential distribution of sigmas

    psi_s = (m x 1)
    psi_a1 = (m x p)
    psi_a2 = (m x p x p)
    psi_t1 = (d x 1)
    psi_t2 = (d x d)
    '''
    psi_t1, psi_t2, psi_a1, psi_a2, psi_s = psi

    thetas = np.random.multivariate_normal(psi_t1, psi_t2)
    alphas = np.array([np.random.multivariate_normal(psi_a1[i],
                       psi_a2[i]) for i in range(m)])
    sigmas = np.random.exponential(1/psi_s, m)
    return thetas, alphas, sigmas

def radial(s, c):
    return np.exp(-5*((s[0]-c[0])**2+(s[1]-c[1])**2))

def linear_reward(theta, s):
    varphi = np.array([radial(s, (0,0)),
                    radial(s,(D-1,D-1)),
                    radial(s, (0,D-1)),
                    radial(s, (D-1,0)),
                     INTERCEPT])
    return np.dot(varphi, theta)

def lin_rew_func(theta):
    '''
    Using radial features above and theta below,
    gives a very close approximation to the true
    RF.
    '''
    rvals = np.zeros((D,D))
    for i in range(D):
        for j in range(D):
            rvals[i,j] = linear_reward(theta, (i,j))
    return rvals

theta = np.array([4, 4, -6, -6, 0.1])
sns.heatmap(lin_rew_func(theta))

def rho(state_space):
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
    reward_str = lin_rew_func(theta_h)
    if Qlearn:
        Q_h = Qlearn(rate, gam, eps, K, T, state_space,
          action_space, reward_str, init_policy, init_Q)[1]
    else:
        Q_h = MCES(gam, tol, T, state_space, action_space,
                   reward_str, init_policy, init_Q)[1]
    return rewards, Q_h

def synthetic_traj(rewards, policy, Ti, state_space, action_space, rho):
    '''
    Generate a fake trajectory based on given
    policy. Need to separately generate
    variational distribution parameters if want to
    use those.
    '''
    s = rho(state_space)
    a = np.random.choice(action_space, p=policy[s])
    states, actions, reward_list = episode(s,Ti,policy,rewards,
                              grid_step,a=a)
    return states, actions, reward_list

def see_trajectory(reward_map, state_seq):
    for s in state_seq:
        sns.heatmap(reward_map)
        plt.annotate('*', (s[1]+0.2,s[0]+0.7), color='b', size=24)
        plt.show()

def make_data(Q, alphas, sigmas, reward_str, N, Ti):
    '''
    Creates N trajectories each for local experts, given Q,
    alphas, sigmas, rewards (which may be "true" or
    based on current guesses!)
    '''
    policies = [locally_opt(Q, alphas[i],
                            sigmas[i])[0] for i in range(m)]
    trajectories = [
        [synthetic_traj(reward_str, policies[i], Ti, state_space,
                        action_space, rho)[:2] for _ in range(N)]
        for i in range(m)
    ]
    return trajectories
    
np.random.seed(1)
psi_t1 = np.random.normal(size=d)
pt2 = np.random.normal(size=(d,d))
psi_t2 = pt2.dot(pt2.transpose())
psi_a1 = np.random.normal(size=(m,p))
pa2 = [np.random.normal(size=(p,p)) for i in range(m)]
psi_a2 = np.stack([pa2[i].dot(pa2[i].transpose()) for i in range(m)])
psi_s = np.random.rand(m)
prior = (psi_t1, psi_t2, psi_a1, psi_a2, psi_s)
thetas, alphas, sigmas = variational(prior)

def psi_gradient(thetas, alphas, sigmas, psi):
    '''
    Given samples of params from the variational distribution,
    computes gradient wrt each component of psi.
    '''
    psi_t1, psi_t2, psi_a1, psi_a2, psi_s = psi
    psi_t2_inv = np.linalg.inv(psi_t2)
    psi_a2_inv = np.linalg.inv(psi_a2) # stack of inverses
    diff_t1 = psi_t1 - thetas
    diff_a1 = psi_a1 - thetas # (m x p)
    psi_t1_g = 2*psi_t2_inv.dot(diff_t1)
    psi_t2_g = psi_t2_inv.dot(1/2*np.outer(diff_t1, diff_t1).dot(psi_t2_inv) - np.eye(p))
    psi_a1_g = np.einsum('ijk,ik->ij', psi_a2_inv, diff_a1) #2*psi_a2_inv.dot(diff_a1)
    a_out = 1/2*np.einsum('ij,ik->ijk', diff_a1, diff_a1)
    a_in = np.einsum('ijk,ikl->ijl', a_out, psi_a2_inv)
    psi_a2_g = np.einsum('ijk,ikl->ijl', psi_a2_inv, (a_in - np.eye(p)))
    psi_s_g = 1/psi_s - sigmas


def AVO(psi, Q_star, exp_alphas, exp_sigmas, rewards, state_space,
        action_space, init_policy, init_Q, gam, delta_tol, rate,
        eps, K, M, N, Ti, T):
    '''
    Init:
    * theta, alpha, sigma, phi, psi

    Inputs:
    * psi = prior on psi, 5-tuple
    ###* true_data = m-list of N-lists of state-seq, action-seq tuples

    Hyperparams:
    * M = minibatch size (32?)
    '''
    delta = np.Inf
    thetas, alphas, sigmas = variational(psi)
    while delta > eps:
        true_data = make_data(Q_star, exp_alphas, exp_sigmas,
                              rewards, N, Ti)
        reward_str, Qh = Q_est(thetas, gam, T,
                               state_space, action_space,
                               init_policy, init_Q, rate=rate,
                               eps=eps, K=K, Qlearn=True)
        fake_data = make_data(Qh, alphas, sigmas, reward_str, N, Ti)
    pass

# Testing
theta_h = np.random.rand(5)
alpha_h = np.random.rand(5)
sigma_h = np.random.rand()
# MCES(0.8, 0.00001, 50, state_space, action_space,
#                 rewards, init_policy, init_Q)
reward_est, Q_h = Q_est(theta_h, 0.8, 50, state_space, action_space,
                               init_policy, init_Q, tol=0.0001)
                        # FIX THIS 
test_policy = locally_opt(Q_h, alpha_h, sigma_h)[0]
synth = synthetic_traj(reward_est, test_policy,
                       10, state_space, action_space, rho)
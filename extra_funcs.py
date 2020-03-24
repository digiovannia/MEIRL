#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 21:04:49 2020

@author: adigi
"""


def theta_grad_det_reg(lam, theta, data, beta, state_space, glogZ_theta, centers_x, centers_y):
    gradR = grad_lin_rew(data, state_space, centers_x, centers_y) # m x d x Ti 
    penalty = lam*theta
    return -glogZ_theta + np.einsum('ij,ikj->k', beta, gradR) - penalty

def MEIRL_det(theta, alpha, traj_data, TP, state_space,
         action_space, B, m, M, Ti, learn_rate, reps, y_t, update, centers_x,
         centers_y, plot=True):
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
    # while error > eps:
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
            lik.append(loglikelihood)
              
            g_theta = theta_grad_det(data, beta, state_space, glogZ_theta, centers_x, centers_y)
            g_alpha = alpha_grad_det(glogZ_alpha, R_all, E_all)
          
            theta_m, alpha_m, = theta, alpha
            theta = y_theta + learn_rate*g_theta
            alpha = y_alpha + learn_rate*g_alpha
            
            mult = (tm - 1)/t
            y_theta = theta + mult*(theta - theta_m)
            y_alpha = alpha + mult*(alpha - alpha_m)
            
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



# possible alternative for MEIRL_det_pos:
    
    '''
    impa = list(np.random.choice(action_space, M))
    N = len(traj_data)
    lik = []
    theta_m = np.zeros_like(theta)
    alpha_m = np.zeros_like(alpha)
    #y_theta = theta.copy()
    #y_alpha = alpha.copy()
    #best = -np.inf
    #best_theta = theta_m.copy()
    #best_alpha = alpha_m.copy()
    #tm = 1
    t = 1 #UNIF
    last_lik = -np.inf
    for _ in range(reps):
        permut = list(np.random.permutation(range(N)))
        for n in permut:
            #t = 1/2*(1 + np.sqrt(1 + 4*tm**2))
            y_theta, y_alpha = y_t(theta, theta_m, alpha, alpha_m, t) #UNIF
            
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
            alpha = np.maximum(0, y_alpha + learn_rate*g_alpha) # UNIF
            # alpha = y_alpha + learn_rate*g_alpha
            
            #mult = (tm - 1)/t
            #y_theta = theta + mult*(theta - theta_m)
            #y_alpha = np.maximum(alpha + mult*(alpha - alpha_m), 0)
            
            learn_rate *= 0.99
            t += 1 # UNIF
            #tm = t
            
            #if loglikelihood > best:
            #    best = loglikelihood
            #    best_theta = y_theta.copy()
            #    best_alpha = y_alpha.copy()
            #elif loglikelihood < last_lik:
            #    y_theta = theta.copy()
            #    y_alpha = alpha.copy()
            #    tm = 1
                
            #last_lik = loglikelihood
    if plot:
        plt.plot(lik)
    return theta, alpha #best_theta, best_alpha
    '''
    
def MEIRL_det_reg(lam, theta, alpha, traj_data, TP, state_space,
         action_space, B, m, M, Ti, learn_rate, reps, y_t, update, centers_x,
         centers_y, plot=True):
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
    # while error > eps:
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
            lik.append(loglikelihood)
              
            g_theta = theta_grad_det_reg(lam, y_theta, data, beta, state_space, glogZ_theta, centers_x, centers_y)
            g_alpha = alpha_grad_det(glogZ_alpha, R_all, E_all)
          
            theta_m, alpha_m, = theta, alpha
            theta = y_theta + learn_rate*g_theta
            alpha = y_alpha + learn_rate*g_alpha
            
            mult = (tm - 1)/t
            y_theta = theta + mult*(theta - theta_m)
            y_alpha = alpha + mult*(alpha - alpha_m)
            
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




##### Evaluation funcs #####
    
def evaluate_vs_random(theta, alpha, sigsq, phi, beta, TP, reps, policy, T,
                        state_space, action_space, rewards, init_policy,
                        init_Q, J, B, m, M, Ti, learn_rate, traj_data,
                        centers_x, centers_y, cr_reps):
    s_list = [state_space[np.random.choice(len(state_space))] for _ in range(cr_reps)]
    true_rew = cumulative_reward(s_list, cr_reps, policy, T, state_space, rewards)
    plt.plot(np.cumsum(true_rew), color='b') 
    true_total = np.sum(true_rew)
    AEVB_total = []
    random_total = []
    for _ in range(J):
        theta_star = AR_AEVB(theta, alpha, sigsq, phi, traj_data, TP, state_space,
         action_space, B, m, M, Ti, learn_rate, reps, plot=False)[0]
        reward_est = lin_rew_func(theta_star, state_space, centers_x, centers_y)
        est_policy = value_iter(state_space, action_space, reward_est, TP, 0.9, 1e-5)[0]
         #Qlearn(0.5, 0.8, 0.1, Q_ITERS, 20, state_space,
          #action_space, reward_est, init_policy, init_Q)[0]
        est_rew = cumulative_reward(s_list, cr_reps, est_policy, T, state_space, rewards)
        plt.plot(np.cumsum(est_rew), color='r')
        AEVB_total.append(np.sum(est_rew))
        
        reward_est = lin_rew_func(np.random.normal(size=d), state_space,
                                  centers_x, centers_y)
        est_policy = value_iter(state_space, action_space, reward_est, TP, 0.9, 1e-5)[0]
        #Qlearn(0.5, 0.8, 0.1, Q_ITERS, 20, state_space,
          #action_space, reward_est, init_policy, init_Q)[0]
        est_rew = cumulative_reward(s_list, cr_reps, est_policy, T, state_space, rewards)
        plt.plot(np.cumsum(est_rew), color='g')
        random_total.append(np.sum(est_rew))
        
        print('.')
    return true_total, AEVB_total, random_total

def evaluate_vs_uniform(theta, alpha, sigsq, phi, beta, TP, reps, policy, T,
                        state_space, action_space, rewards, init_policy,
                        init_Q, J, B, m, M, Ti, learn_rate, traj_data,
                        centers_x, centers_y, cr_reps):
    s_list = [state_space[np.random.choice(len(state_space))] for _ in range(cr_reps)]
    true_rew = cumulative_reward(s_list, cr_reps, policy, T, state_space, rewards)
    plt.plot(np.cumsum(true_rew), color='b') 
    true_total = np.sum(true_rew)
    AEVB_total = []
    unif_total = []
    for _ in range(J):
        theta_star = AR_AEVB(theta, alpha, sigsq, phi, traj_data, TP, state_space,
         action_space, B, m, M, Ti, learn_rate, 5, GD, plot=False)[0]
        reward_est = lin_rew_func(theta_star, state_space, centers_x, centers_y)
        est_policy = value_iter(state_space, action_space, reward_est, TP, 0.9, 1e-5)[0]
        #Qlearn(0.5, 0.8, 0.1, Q_ITERS, 20, state_space,
          #action_space, reward_est, init_policy, init_Q)[0]
        est_rew = cumulative_reward(s_list, cr_reps, est_policy, T, state_space, rewards)
        plt.plot(np.cumsum(est_rew), color='r')
        AEVB_total.append(np.sum(est_rew))
        
        theta_star = MEIRL_unif(theta, beta, traj_data, TP, state_space,
         action_space, B, m, M, Ti, learn_rate, 5, GD, centers_x, centers_y)[0]
        reward_est = lin_rew_func(theta_star, state_space,
                                  centers_x, centers_y)
        est_policy = value_iter(state_space, action_space, reward_est, TP, 0.9, 1e-5)[0]
         #Qlearn(0.5, 0.8, 0.1, Q_ITERS, 20, state_space,
          #action_space, reward_est, init_policy, init_Q)[0]
        est_rew = cumulative_reward(s_list, cr_reps, est_policy, T, state_space, rewards)
        plt.plot(np.cumsum(est_rew), color='g')
        unif_total.append(np.sum(est_rew))
        
        print('.')
    return true_total, AEVB_total, unif_total

def evaluate_det_vs_unif(theta, alpha, sigsq, phi, beta, TP, reps, policy, T,
                        state_space, action_space, rewards, init_policy,
                        init_Q, J, B, m, M, Ti, learn_rate, traj_data,
                        centers_x, centers_y, cr_reps):
    s_list = [state_space[np.random.choice(len(state_space))] for _ in range(cr_reps)]
    true_rew = cumulative_reward(s_list, cr_reps, policy, T, state_space, rewards)
    plt.plot(np.cumsum(true_rew), color='b') 
    true_total = np.sum(true_rew)
    det_total = []
    unif_total = []
    for _ in range(J):
        theta_star = MEIRL_det_pos(theta, alpha, traj_data, TP, state_space,
         action_space, B, m, M, Ti, learn_rate, reps,
         centers_x, centers_y, plot=False)[0]
        reward_est = lin_rew_func(theta_star, state_space,
                                  centers_x, centers_y)
        est_policy = value_iter(state_space, action_space, reward_est, TP, 0.9, 1e-5)[0]
        #Qlearn(0.5, 0.8, 0.1, Q_ITERS, 20, state_space,
          #action_space, reward_est, init_policy, init_Q)[0]
        est_rew = cumulative_reward(s_list, cr_reps, est_policy, T, state_space, rewards)
        plt.plot(np.cumsum(est_rew), color='r')
        det_total.append(np.sum(est_rew))

        theta_star = MEIRL_unif(theta, beta, traj_data, TP, state_space,
         action_space, B, m, M, Ti, learn_rate, reps, GD_unif, centers_x, centers_y)[0]
        reward_est = lin_rew_func(theta_star, state_space,
                                  centers_x, centers_y)
        est_policy = value_iter(state_space, action_space, reward_est, TP, 0.9, 1e-5)[0]
        #Qlearn(0.5, 0.8, 0.1, Q_ITERS, 20, state_space,
          #action_space, reward_est, init_policy, init_Q)[0]
        est_rew = cumulative_reward(s_list, cr_reps, est_policy, T, state_space, rewards)
        plt.plot(np.cumsum(est_rew), color='g')
        unif_total.append(np.sum(est_rew))
        
        print('.')
    return true_total, det_total, unif_total

def evaluate_unif_vs_random(theta, alpha, sigsq, phi, beta, TP, reps, policy, T,
                        state_space, action_space, rewards, init_policy,
                        init_Q, J, B, m, M, Ti, learn_rate, traj_data,
                        centers_x, centers_y, cr_reps):
    s_list = [state_space[np.random.choice(len(state_space))] for _ in range(cr_reps)]
    true_rew = cumulative_reward(s_list, cr_reps, policy, T, state_space, rewards)
    plt.plot(np.cumsum(true_rew), color='b') 
    true_total = np.sum(true_rew)
    unif_total = []
    ra_total = []
    for _ in range(J):
        theta_star = MEIRL_unif(theta, beta, traj_data, TP, state_space,
         action_space, B, m, M, Ti, learn_rate, reps, GD_unif, centers_x, centers_y)[0]
        reward_est = lin_rew_func(theta_star, state_space,
                                  centers_x, centers_y)
        est_policy = value_iter(state_space, action_space, reward_est, TP, 0.9, 1e-5)[0]
        #Qlearn(0.5, 0.8, 0.1, Q_ITERS, 20, state_space,
          #action_space, reward_est, init_policy, init_Q)[0]
        est_rew = cumulative_reward(s_list, cr_reps, est_policy, T, state_space, rewards)
        plt.plot(np.cumsum(est_rew), color='r')
        unif_total.append(np.sum(est_rew))

        theta_star = np.random.normal(size=d)
        reward_est = lin_rew_func(theta_star, state_space,
                                  centers_x, centers_y)
        est_policy = value_iter(state_space, action_space, reward_est, TP, 0.9, 1e-5)[0]
        #Qlearn(0.5, 0.8, 0.1, Q_ITERS, 20, state_space,
          #action_space, reward_est, init_policy, init_Q)[0]
        est_rew = cumulative_reward(s_list, cr_reps, est_policy, T, state_space, rewards)
        plt.plot(np.cumsum(est_rew), color='g')
        ra_total.append(np.sum(est_rew))
        
        print('.')
    return true_total, unif_total, ra_total

def evaluate_det_vs_random(theta, alpha, sigsq, phi, TP, reps, policy, T,
                        state_space, action_space, rewards, init_policy,
                        init_Q, J, B, m, M, Ti, learn_rate, traj_data,
                        centers_x, centers_y, cr_reps):
    s_list = [state_space[np.random.choice(len(state_space))] for _ in range(cr_reps)]
    true_rew = cumulative_reward(s_list, cr_reps, policy, T, state_space, rewards)
    plt.plot(np.cumsum(true_rew), color='b') 
    true_total = np.sum(true_rew)
    det_total = []
    unif_total = []
    for _ in range(J):
        theta_star = MEIRL_det_pos(theta, alpha, traj_data, TP, state_space,
         action_space, B, m, M, Ti, learn_rate, reps,
         centers_x, centers_y, plot=False)[0]
        reward_est = lin_rew_func(theta_star, state_space,
                                  centers_x, centers_y)
        est_policy = value_iter(state_space, action_space, reward_est, TP, 0.9, 1e-5)[0]
        #Qlearn(0.5, 0.8, 0.1, Q_ITERS, 20, state_space,
         # action_space, reward_est, init_policy, init_Q)[0]
        est_rew = cumulative_reward(s_list, cr_reps, est_policy, T, state_space, rewards)
        plt.plot(np.cumsum(est_rew), color='r')
        det_total.append(np.sum(est_rew))

        reward_est = lin_rew_func(np.random.normal(size=d), state_space,
                                  centers_x, centers_y)
        est_policy = value_iter(state_space, action_space, reward_est, TP, 0.9, 1e-5)[0]
        #Qlearn(0.5, 0.8, 0.1, Q_ITERS, 20, state_space,
         # action_space, reward_est, init_policy, init_Q)[0]
        est_rew = cumulative_reward(s_list, cr_reps, est_policy, T, state_space, rewards)
        plt.plot(np.cumsum(est_rew), color='g')
        unif_total.append(np.sum(est_rew))
        
        print('.')
    return true_total, det_total, unif_total

def evaluate(reps, policy, T, state_space, rewards, theta_est, init_policy,
             init_Q, centers_x, centers_y):
    reward_est = lin_rew_func(theta_est, state_space, centers_x, centers_y)
    est_policy = Qlearn(0.5, 0.8, 0.1, Q_ITERS, 20, state_space,
          action_space, reward_est, init_policy, init_Q)[0]
    true_rew = cumulative_reward(reps, policy, T, state_space, rewards)
    est_rew = cumulative_reward(reps, est_policy, T, state_space, rewards)
    plt.plot(np.cumsum(true_rew), color='b')
    plt.plot(np.cumsum(est_rew), color='r')
    return np.sum(true_rew), np.sum(est_rew)

def evaluate_vs_det(theta, alpha, sigsq, phi, beta, TP, reps, policy, T,
                        state_space, action_space, rewards, init_policy,
                        init_Q, J, B, m, M, Ti, learn_rate, traj_data,
                        centers_x, centers_y, cr_reps):
    s_list = [state_space[np.random.choice(len(state_space))] for _ in range(cr_reps)]
    true_rew = cumulative_reward(s_list, cr_reps, policy, T, state_space, rewards)
    plt.plot(np.cumsum(true_rew), color='b') 
    true_total = np.sum(true_rew)
    AEVB_total = []
    det_total = []
    for _ in range(J):
        theta_star = AR_AEVB(theta, alpha, sigsq, phi, traj_data, TP, state_space,
         action_space, B, m, M, Ti, learn_rate, 5, plot=False)[0]
        reward_est = lin_rew_func(theta_star, state_space, centers_x, centers_y)
        est_policy = value_iter(state_space, action_space, reward_est, TP, 0.9, 1e-5)[0]
        #Qlearn(0.5, 0.8, 0.1, Q_ITERS, 20, state_space,
        #action_space, reward_est, init_policy, init_Q)[0]
        est_rew = cumulative_reward(s_list, cr_reps, est_policy, T, state_space, rewards)
        plt.plot(np.cumsum(est_rew), color='r')
        AEVB_total.append(np.sum(est_rew))
        
        '''
        Changing to MEIRL_det_pos here!
        '''
        theta_star = MEIRL_det_pos(theta, alpha, traj_data, TP, state_space,
         action_space, B, m, M, Ti, learn_rate, 5, GD,
         centers_x, centers_y, plot=False)[0]
        reward_est = lin_rew_func(theta_star, state_space,
                                  centers_x, centers_y)
        est_policy = value_iter(state_space, action_space, reward_est, TP, 0.9, 1e-5)[0]
        #Qlearn(0.5, 0.8, 0.1, Q_ITERS, 20, state_space,
         # action_space, reward_est, init_policy, init_Q)[0]
        est_rew = cumulative_reward(s_list, cr_reps, est_policy, T, state_space, rewards)
        plt.plot(np.cumsum(est_rew), color='g')
        det_total.append(np.sum(est_rew))
        
        print('.')
    return true_total, AEVB_total, det_total
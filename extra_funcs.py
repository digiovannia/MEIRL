#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 21:04:49 2020

@author: adigi
"""

def save_results(id_num,  algo_a=AR_AEVB, algo_b=MEIRL_unif, random=False,
                 test_data='myo', hyparams=HYPARAMS):
    '''
    Compares two algorithms with each other (and with ground truth reward)
    with varying seed values, including varying reward functions
    '''
    
    # Global params
    global D, MOVE_NOISE, INTERCEPT_ETA, WEIGHT, RESCALE, COEF
    global ETA_COEF, GAM, M, N, J, T, Ti, B, INTERCEPT_REW, TP
    (D, MOVE_NOISE, INTERCEPT_ETA, WEIGHT, COEF, ETA_COEF, GAM, M, N, J,
      T, Ti, B, INTERCEPT_REW, learn_rate, cr_reps, reps,
      sigsq_list) = (hyparams['D'], hyparams['MOVE_NOISE'],
      hyparams['INTERCEPT_ETA'], hyparams['WEIGHT'],
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

def ann_AEVB(theta, alpha, sigsq, phi, beta, traj_data, TP, state_space,
         action_space, B, m, M, Ti, N, learn_rate, reps, centers_x, centers_y,
         plot=True):
    '''
    Autoencoder with simulated annealing and Nesterov
    '''
    impa = list(np.random.choice(action_space, M))
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
    time_since_best = 0
    tm = 1
    start_lr = learn_rate
    normals = np.random.multivariate_normal(np.zeros(Ti), np.eye(Ti), (m, B))
    
    for _ in range(reps):
        permut = list(np.random.permutation(range(N)))
        for n in permut:
            time_since_best += 1
            t = 1/2*(1 + np.sqrt(1 + 4*tm**2))

            data = np.array(traj_data[n])
            reward_est = lin_rew_func(y_theta, state_space, centers_x,
              centers_y)
            R_all, E_all = RE_all(reward_est, data, TP, state_space, m,
              centers_x, centers_y)
            meanvec, denom, gvec, gnorm = grad_terms(normals, y_phi, y_alpha,
              y_sigsq, y_theta, data, R_all, E_all, Ti, m)
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
                time_since_best = 0
              
            g_phi = phi_grad_ae(y_phi, m, Ti, normals, denom, y_sigsq, gZ_phi)
            g_theta = theta_grad_ae(gZ_theta, data, state_space, R_all, E_all,
              y_sigsq, y_alpha, centers_x, centers_y)
            g_alpha = alpha_grad_ae(gZ_alpha, E_all, R_all)
            g_sigsq = sigsq_grad_ae(gZ_sigsq, normals, Ti, y_sigsq, gnorm,
              denom, R_all, gvec)
            
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
            #y_alpha = alpha + mult*(alpha - alpha_m)
            y_alpha = np.maximum(alpha + mult*(alpha - alpha_m), 0)
            y_sigsq = np.maximum(sigsq + mult*(sigsq - sigsq_m), 0.01)
            
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
    if plot:
        plt.plot(elbos)
    return best_theta, best_phi, best_alpha, best_sigsq

def AEVB(theta, alpha, sigsq, phi, beta, traj_data, TP, state_space,
         action_space, B, m, M, Ti, N, learn_rate, reps, centers_x, centers_y,
         plot=True):
    impa = list(np.random.choice(action_space, M))
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
    normals = np.random.multivariate_normal(np.zeros(Ti), np.eye(Ti), (m, B))
    for _ in range(reps):
        permut = list(np.random.permutation(range(N)))
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
            
            g_phi = g_phi / np.linalg.norm(g_phi)
            g_theta = g_theta / np.linalg.norm(g_theta)
            g_alpha = g_alpha / np.linalg.norm(g_alpha, 'f')
            g_sigsq = g_sigsq / np.linalg.norm(g_sigsq)
          
            phi_m, theta_m, alpha_m, sigsq_m = phi, theta, alpha, sigsq
            phi, theta, alpha, sigsq = GD(y_phi, y_theta, y_alpha, y_sigsq,
              g_phi, g_theta, g_alpha, g_sigsq, learn_rate)
            
            mult = (tm - 1)/t
            y_phi = phi + mult*(phi - phi_m)
            y_theta = theta + mult*(theta - theta_m)
            y_alpha = np.maximum(alpha + mult*(alpha - alpha_m), 0)
            y_sigsq = sigsq + mult*(sigsq - sigsq_m)
            
            learn_rate *= 0.99
            tm = t
    if plot:
        plt.plot(elbos)
    return best_theta, best_phi, best_alpha, best_sigsq

def arr_radial(s, c, coef):
    '''
    Radial basis functions for the reward function, applied to an array of
    states. "c" is a fixed center point.
    '''
    return RESCALE*np.exp(-coef*((s[:,0]-c[0])**2+(s[:,1]-c[1])**2))

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
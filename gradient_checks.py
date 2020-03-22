#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 11:58:54 2020

@author: adigi
"""

def grad_check_phi_re(phi, alpha, sigsq, theta, data, Ti,
                     m, state_space, B, impa, ix):
    '''
    Good on phi1, not on phi2
    '''
    epsilon = 1e-4

    R_all, E_all = RE_all(theta, data, TP, state_space, m, centers_x, centers_y)
    normals = np.random.multivariate_normal(np.zeros(Ti), np.eye(Ti), (m,B))
    meanvec, denom, gvec, gnorm = grad_terms_re(normals,
      phi, alpha, sigsq, theta, data, R_all, E_all, Ti, m)
    logZvec, glogZ_theta, glogZ_alpha, glogZ_sigsq, glogZ_phi = logZ_re(normals, meanvec, denom, impa, theta, data, M, TP, R_all, E_all,
                    action_space, centers_x, centers_y)
    a_p_g = phi_grad_re(phi, m, Ti, normals, denom, sigsq, glogZ_phi)

    left = phi.copy()
    left[:,ix] += epsilon
    right = phi.copy()
    right[:,ix] -= epsilon
    meanvec_l, denom_l, gvec_l, gnorm_l = grad_terms_re(normals,
      left, alpha, sigsq, theta, data, R_all, E_all, Ti, m)
    meanvec_r, denom_r, gvec_r, gnorm_r = grad_terms_re(normals,
      right, alpha, sigsq, theta, data, R_all, E_all, Ti, m)
    logZvec_l, glogZ_theta_l, glogZ_alpha_l, glogZ_sigsq_l, glogZ_phi_l = logZ_re(normals,
      meanvec_l, denom_l, impa, theta, data, M, TP, R_all, E_all, action_space, centers_x, centers_y)
    logZvec_r, glogZ_theta_r, glogZ_alpha_r, glogZ_sigsq_r, glogZ_phi_r = logZ_re(normals,
      meanvec_r, denom_r, impa, theta, data, M, TP, R_all, E_all, action_space, centers_x, centers_y)

    logprobdiff_l = logprobs(state_space, Ti, sigsq, gnorm_l, data, TP, m, normals, R_all,
      logZvec_l, meanvec_l, denom_l)
    logprobdiff_r = logprobs(state_space, Ti, sigsq, gnorm_r, data, TP, m, normals, R_all,
      logZvec_r, meanvec_r, denom_r)

    n_t_g = (logprobdiff_l - logprobdiff_r)/(2*epsilon)
    change = n_t_g.mean(axis=1)
    return a_p_g[:,ix], change

def grad_check_alpha_re(phi, alpha, sigsq, theta, data, Ti,
                     m, state_space, B, impa, ix):
    '''
    WORKS!
    '''
    epsilon = 1e-4

    R_all, E_all = RE_all(theta, data, TP, state_space, m, centers_x, centers_y)
    normals = np.random.multivariate_normal(np.zeros(Ti), np.eye(Ti), (m,B))
    meanvec, denom, gvec, gnorm = grad_terms_re(normals,
      phi, alpha, sigsq, theta, data, R_all, E_all, Ti, m)

    logZvec, glogZ_theta, glogZ_alpha, glogZ_sigsq, glogZ_phi = logZ_re(normals, meanvec, denom, impa, theta, data, M, TP, R_all, E_all,
                    action_space, centers_x, centers_y)
    a_a_g = alpha_grad_re(glogZ_alpha, E_all, R_all)

    left = alpha.copy()
    left[:,ix] += epsilon
    right = alpha.copy()
    right[:,ix] -= epsilon
    meanvec_l, denom_l, gvec_l, gnorm_l = grad_terms_re(normals,
      phi, left, sigsq, theta, data, R_all, E_all, Ti, m)
    meanvec_r, denom_r, gvec_r, gnorm_r = grad_terms_re(normals,
      phi, right, sigsq, theta, data, R_all, E_all, Ti, m)
    logZvec_l, glogZ_theta_l, glogZ_alpha_l, glogZ_sigsq_l, glogZ_phi_l = logZ_re(normals,
      meanvec_l, denom_l, impa, theta, data, M, TP, R_all, E_all, action_space, centers_x, centers_y)
    logZvec_r, glogZ_theta_r, glogZ_alpha_r, glogZ_sigsq_r, glogZ_phi_r = logZ_re(normals,
      meanvec_r, denom_r, impa, theta, data, M, TP, R_all, E_all, action_space, centers_x, centers_y)

    logprobdiff_l = logprobs(state_space, Ti, sigsq, gnorm_l, data, TP, m, normals, R_all,
      logZvec_l, meanvec_l, denom_l)
    logprobdiff_r = logprobs(state_space, Ti, sigsq, gnorm_r, data, TP, m, normals, R_all,
      logZvec_r, meanvec_r, denom_r)

    n_t_g = (logprobdiff_l - logprobdiff_r)/(2*epsilon)
    change = n_t_g.mean(axis=1)
    return a_a_g[:,ix], change

def grad_check_sigsq_re(phi, alpha, sigsq, theta, data, Ti,
                     m, state_space, B, impa, ix):
    '''
    Some variance still, but pretty good.
    '''
    epsilon = 1e-4

    R_all, E_all = RE_all(theta, data, TP, state_space, m, centers_x, centers_y)
    normals = np.random.multivariate_normal(np.zeros(Ti), np.eye(Ti), (m,B))
    meanvec, denom, gvec, gnorm = grad_terms_re(normals,
      phi, alpha, sigsq, theta, data, R_all, E_all, Ti, m)
    logZvec, glogZ_theta, glogZ_alpha, glogZ_sigsq, glogZ_phi = logZ_re(normals, meanvec, denom, impa, theta, data, M, TP,
                    R_all, E_all, action_space, centers_x, centers_y)
    a_s_g = sigsq_grad_re(glogZ_sigsq, normals, Ti, sigsq, gnorm, denom, R_all,
                  gvec)

    left = sigsq.copy()
    left += epsilon
    right = sigsq.copy()
    right -= epsilon
    meanvec_l, denom_l, gvec_l, gnorm_l = grad_terms_re(normals,
      phi, alpha, left, theta, data, R_all, E_all, Ti, m)
    meanvec_r, denom_r, gvec_r, gnorm_r = grad_terms_re(normals,
      phi, alpha, right, theta, data, R_all, E_all, Ti, m)
    logZvec_l, glogZ_theta_l, glogZ_alpha_l, glogZ_sigsq_l, glogZ_phi_l = logZ_re(normals,
      meanvec_l, denom_l, impa, theta, data, M, TP, R_all, E_all, action_space, centers_x, centers_y)
    logZvec_r, glogZ_theta_r, glogZ_alpha_r, glogZ_sigsq_r, glogZ_phi_r = logZ_re(normals,
      meanvec_r, denom_r, impa, theta, data, M, TP, R_all, E_all, action_space, centers_x, centers_y)
    
    logprobdiff_l = logprobs(state_space, Ti, sigsq, gnorm_l, data, TP, m, normals, R_all,
      logZvec_l, meanvec_l, denom_l)
    logprobdiff_r = logprobs(state_space, Ti, sigsq, gnorm_r, data, TP, m, normals, R_all,
      logZvec_r, meanvec_r, denom_r)

    n_t_g = (logprobdiff_l - logprobdiff_r)/(2*epsilon)
    change = n_t_g.mean(axis=1)
    return a_s_g, change

def grad_check_theta_re(phi, alpha, sigsq, theta, data, Ti,
                     m, state_space, B, impa, ix):
    '''
    Much less variance! but still biased...

    Deduced that the bias is in the grad log.
    logZ itself is approximated, so as long as
    consistent this should still work...?
    '''
    epsilon = 1e-4

    R_all, E_all = RE_all(theta, data, TP, state_space, m, centers_x, centers_y)
    normals = np.random.multivariate_normal(np.zeros(Ti), np.eye(Ti), (m,B))
    meanvec, denom, gvec, gnorm = grad_terms_re(normals,
      phi, alpha, sigsq, theta, data, R_all, E_all, Ti, m)

    logZvec, glogZ_theta, glogZ_alpha, glogZ_sigsq, glogZ_phi = logZ_re(normals, meanvec, denom, impa, theta, data, M, TP, R_all, E_all,
                    action_space, centers_x, centers_y)
    a_t_g = theta_grad_re(glogZ_theta, data, state_space, R_all, E_all, sigsq, alpha, centers_x, centers_y)


    left = theta.copy()
    left[ix] += epsilon
    right = theta.copy()
    right[ix] -= epsilon
    R_all_l, E_all_l = RE_all(left, data, TP, state_space, m, centers_x, centers_y)
    R_all_r, E_all_r = RE_all(right, data, TP, state_space, m, centers_x, centers_y)
    meanvec_l, denom_l, gvec_l, gnorm_l = grad_terms_re(normals,
      phi, alpha, sigsq, left, data, R_all_l, E_all_l, Ti, m)
    meanvec_r, denom_r, gvec_r, gnorm_r = grad_terms_re(normals,
      phi, alpha, sigsq, right, data, R_all_r, E_all_r, Ti, m)
    logZvec_l, glogZ_theta_l, glogZ_alpha_l, glogZ_sigsq_l, glogZ_phi_l = logZ_re(normals,
      meanvec_l, denom_l, impa, left, data, M, TP, R_all_l, E_all_l, action_space, centers_x, centers_y)
    logZvec_r, glogZ_theta_r, glogZ_alpha_r, glogZ_sigsq_r, glogZ_phi_r = logZ_re(normals,
      meanvec_r, denom_r, impa, right, data, M, TP, R_all_r, E_all_r, action_space, centers_x, centers_y)

    logprobdiff_l = logprobs(state_space, Ti, sigsq, gnorm_l, data, TP, m, normals, R_all,
      logZvec_l, meanvec_l, denom_l)
    logprobdiff_r = logprobs(state_space, Ti, sigsq, gnorm_r, data, TP, m, normals, R_all,
      logZvec_r, meanvec_r, denom_r)

    n_t_g = (logprobdiff_l - logprobdiff_r)/(2*epsilon)
    change = (n_t_g.mean(axis=1)).sum()
    return a_t_g[ix], change

def logp(state_space, Ti, sigsq, gnorm, data, TP, m, betas, R_all, logZvec):
    p1 = np.log(1/len(state_space)) - Ti/2*np.log(2*np.pi*sigsq)[:,None] - 1/(2*sigsq)[:,None]*gnorm
    logT = np.log(traj_TP(data, TP, Ti, m))
    p2 = np.einsum('ijk,ik->ij', betas, R_all) - logZvec + np.sum(logT, axis=1)[:,None]
    return p1 + p2

def logq(Ti, denom, vecnorm):
    return -Ti/2*np.log(2*np.pi*denom)[:,None] - vecnorm/((2*denom)[:,None])
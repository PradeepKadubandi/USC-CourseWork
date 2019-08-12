from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        # Edit here
        alpha[:, 0] = self.pi * self.B[:, self.obs_dict[Osequence[0]]]
        for t in range(1, L):
            alpha[:, t] = np.matmul(self.A.T, alpha[:, t-1]) * self.B[:, self.obs_dict[Osequence[t]]]
        ###################################################
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        # Edit here
        beta[:, L-1] = 1.0
        for t in range(L-2, -1, -1):
            beta[:, t] = np.matmul(self.A, (beta[:, t+1] * self.B[:, self.obs_dict[Osequence[t+1]]]))
        ###################################################
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0
        ###################################################
        # Edit here
        alpha = self.forward(Osequence)
        prob = sum(alpha[:, len(Osequence) - 1])
        ###################################################
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        ###################################################
        # Edit here
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        seq_prob = sum(alpha[:, len(Osequence) - 1])
        prob = (alpha * beta) / seq_prob
        ###################################################
        return prob
    #TODO:
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        ###################################################
        # Edit here
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        seq_prob = sum(alpha[:, len(Osequence) - 1])
        
        # for t in range(L-1):
            # obs_index = self.obs_dict[Osequence[t+1]]
            # prob[:, :, t] = (alpha[:, t] * self.A * self.B[:, obs_index].T * beta[:, t+1].T) / seq_prob
            
        for t in range(L-1):
            obs_index = self.obs_dict[Osequence[t+1]]
            for s_b in range(S):
                for s_e in range(S):
                    prob[s_b, s_e, t] = (alpha[s_b, t] * self.A[s_b, s_e] * self.B[s_e, obs_index] * beta[s_e, t+1]) / seq_prob
        ###################################################
        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        # Q3.3 Edit here
        S = len(self.pi)
        L = len(Osequence)
        doe = np.zeros([S, L])
        delta = np.zeros([S, L], dtype=int)
        doe[:, 0] = self.pi * self.B[:, self.obs_dict[Osequence[0]]]
        for t in range(1, L):
            prod = self.A.T * doe[:, t-1]
            delta[:, t] = np.argmax(prod, axis=1)
            doe[:, t] = np.max(prod, axis=1) * self.B[:, self.obs_dict[Osequence[t]]]
        
        sid_to_st = {}
        for k, v in self.state_dict.items():
            sid_to_st[v] = k
            
        st = np.argmax(doe[:, L-1])
        path.append(sid_to_st[st])
        for t in range(L-1, 0, -1):
            st = delta[st, t]
            path.append(sid_to_st[st])
            
        path.reverse() 
        ###################################################
        return path

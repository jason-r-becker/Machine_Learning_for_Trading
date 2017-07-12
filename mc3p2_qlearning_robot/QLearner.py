"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand


class QLearner(object):
    def author(self):
        return 'Jason R Becker'

    def __init__(self, \
         num_states=100, \
         num_actions=4, \
         alpha=0.2, \
         alphar=0.2, \
         gamma=0.9, \
         rar=0.5, \
         radr=0.99, \
         dyna=0, \
         verbose=False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0

        self.rar = rar
        self.radr = radr
        self.alpha = alpha
        self.gamma = gamma
        self.dyna = dyna
        self.num_states = num_states
        self.itter = 0

        self.q = {}
        self.R = {}

        # Initialize Q table
        for s, a in np.ndindex((num_states, num_actions)):
            self.q[(s, a)] = np.random.random_integers(-1, 1)

        # Initialize T and R tables for dyna
        if self.dyna > 0:
            self.Tc = np.empty(num_states**2 * num_actions).reshape(num_states, num_actions, num_states)
            self.Tc.fill(0.00001)
            self.T = self.Tc / self.Tc.sum(axis=2, keepdims=True)

            for s, a in np.ndindex((num_states, num_actions)):
                self.R[(s, a)] = -1

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """

        self.s = s
        actions = [self.q.get((s, a), 0.0) for a in range(self.num_actions)]
        action = np.random.choice([i for i, a in enumerate(actions) if a == max(actions)])

        if self.verbose:
            print "s =", s, "a =", action

        return action

    def query(self, s_prime, r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """

        # Update Q table
        Qmax = max([self.q.get((s_prime, a_prime), 0.0) for a_prime in range(self.num_actions)])
        self.q[(self.s, self.a)] = (1 - self.alpha) * self.q[(self.s, self.a)] + self.alpha * (r + self.gamma * Qmax)

        if np.random.random() < self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            actions = [self.q.get((s_prime, a), 0.0) for a in range(self.num_actions)]
            action = np.random.choice([i for i, a in enumerate(actions) if a == max(actions)])

        # Update random action with decay rate
        self.rar *= self.radr
        self.itter += 1

        # Implement dyna
        if self.dyna > 0:
            self.Tc[self.s, self.a, s_prime] += 1
            self.T = self.Tc / self.Tc.sum(axis=2, keepdims=True)
            self.R[(self.s, self.a)] = (1 - self.alpha) * self.R[(self.s, self.a)] + self.alpha * r

            dyna_s = np.random.random_integers(0, self.num_states - 1, self.dyna)
            dyna_a = np.random.random_integers(0, self.num_actions - 1, self.dyna)

            if self.itter > 10:  # Seeds first 10 iterations before performing dyna
                for i in range(self.dyna):
                    dyna_s_prime = np.random.multinomial(1, self.T[dyna_s[i], dyna_a[i], ]).argmax()
                    dyna_r = self.R.get((dyna_s[i], dyna_a[i]), 0.0)
                    dyna_Qmax = max([self.q.get((dyna_s_prime, dyna_a_prime), 0.0)
                                     for dyna_a_prime in range(self.num_actions)])
                    self.q[(dyna_s[i], dyna_a[i])] = (1 - self.alpha) * self.q[(dyna_s[i], dyna_a[i])] + \
                                                     self.alpha * (dyna_r + self.gamma * dyna_Qmax)

        self.a = action
        self.s = s_prime

        if self.verbose:
            print "s =", s_prime, "a =", action, "r =", r

        return action


if __name__ == "__main__":
    print "Remember Q from Star Trek? Well, this isn't him"

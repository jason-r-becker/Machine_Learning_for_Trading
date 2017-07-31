"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand


class QLearner(object):
    def author(self):
        return 'Jason R Becker'

    def __init__(self, \
                 num_states=100,
                 num_actions=3,
                 alpha=0.2,
                 gamma=0.9,
                 rar=0.5,
                 radr=0.99,
                 dyna=0,
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

        # Initialize Q table
        self.q = np.random.uniform(-1, 1, size=(num_states, num_actions))

        # Initialize T and R tables for dyna
        if self.dyna != 0:
            self.Tc = np.ndarray(shape=(num_states, num_actions, num_states))
            self.Tc.fill(0.00001)
            self.T = self.Tc / self.Tc.sum(axis=2, keepdims=True)
            self.R = np.ndarray(shape=(num_states, num_actions))
            self.R.fill(-1.0)

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """

        self.s = s
        action = rand.randint(0, self.num_actions - 1)
        if rand.random() > self.rar:
            action = np.argmax(self.q[s,])
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
        self.q[self.s, self.a] = (1 - self.alpha) * self.q[self.s, self.a] + self.alpha * (
            r + self.gamma * np.max(self.q[s_prime,]))

        if rand.random() <= self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.q[s_prime,])

        # Update random action with decay rate
        self.rar = self.rar * self.radr

        # Implement dyna
        if self.dyna != 0:
            # increment Tc, update T and R
            self.Tc[self.s, self.a, s_prime] = self.Tc[self.s, self.a, s_prime] + 1
            self.T = self.Tc / self.Tc.sum(axis=2, keepdims=True)
            self.R[self.s, self.a] = (1 - self.alpha) * self.R[self.s, self.a] + (self.alpha * r)

            # iterate through the dyna simulations
            for i in range(0, self.dyna):
                # select a random a and s
                a_dyna = np.random.randint(low=0, high=self.num_actions)
                s_dyna = np.random.randint(low=0, high=self.num_states)
                # infer s' from T
                s_prime_dyna = np.random.multinomial(1, self.T[s_dyna, a_dyna,]).argmax()
                # compute R from s and a
                r = self.R[s_dyna, a_dyna]
                # update Q
                self.q[s_dyna, a_dyna] = (1 - self.alpha) * self.q[s_dyna, a_dyna] + \
                                         self.alpha * (r + self.gamma * np.max(self.q[s_prime_dyna,]))

        self.s = s_prime
        self.a = action

        if self.verbose:
            print "s =", s_prime, "a =", action, "r =", r

        return action


if __name__ == "__main__":
    print "Remember Q from Star Trek? Well, this isn't him"

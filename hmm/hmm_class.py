
import numpy as np

class Hmm:

    def __init__(self, state_dict, obs_dict, init_prob, trans_prob, emit_prob, given_obs):
        self.obs_dict = obs_dict
        self.state_dict = state_dict
        self.init_prob = init_prob
        self.trans_prob = trans_prob
        self.emit_prob = emit_prob
        self.given_obs = given_obs
        self.time = len(given_obs)
        self.state = len(init_prob)

    def forward_algorithim(self):
        forward = np.zeros((self.time, self.state))
        # loop through all the time sequence of the given observations
        for t in range(self.time):
            # loop through the current hidden self.state
            for s in range(self.state):
                path_probs = []  # list that stores path probabilities
                # if the time is the first time use the initial probability
                # and we have no previous forward probability
                if t == 0:
                    # the probability of each path
                    obs_indx = next((indx for indx, obs in self.obs_dict.items() if obs == self.given_obs[t]), None)
                    prob = self.init_prob[s] * self.emit_prob[s][obs_indx]
                    path_probs.append(prob)

                # if the time is not the first time we use the transition probability,
                # also we use the previous forward probability
                else:
                    # loop through the previous self.state,
                    for i in range(self.state):
                        # the path probability is multiplication of the transition prob times the emission probability
                        obs_indx = next((indx for indx, obs in self.obs_dict.items() if obs == self.given_obs[t]), None)
                        prob = self.trans_prob[i][s] * self.emit_prob[s][obs_indx]

                        # the forward probability from previous self.state is the multiplication of path probability times
                        # the forward probability of that previous self.state

                        fwrd_prob = forward[t - 1][i] * prob

                        # then append the calculated forward probability through each path to the path_probs
                        path_probs.append(fwrd_prob)

                # the forward probability of current state, with in a current time, is the sum of the total path probabilities
                forward[t][s] = sum(path_probs)

        return forward

    def viterbi_algorithim(self):
       pass

    def backward_algorithim(self):
        pass

    def do_estep(self):
        pass

    def do_mstep(self):
        pass

def main():

    state_dict = {0: "Hot", 1: "Cold"}
    obs_dict = {0: "1", 1: "2", 2: "3"}  # eating 1 or 2 or 3 ice cream based on the weather condition

    # state_dict = {0:"s", 1:"t"}
    # obs_dict = {0:"A", 1:"B"}

    init_prob = [0.8, 0.2]

    # init_prob = [0.85, 0.15]

    # Hot to Hot and Cold
    # Cold to Hot and Cold

    trans_prob = [[0.6, 0.4],
                  [0.5, 0.5]]

    # s to s and t
    # t to s and t

    # trans_prob = [[0.3, 0.7],
    #             [0.1, 0.9]]

    # Hot then prob of 1, 2, 3
    # Cold then prob of 1, 2, 3

    emit_prob = [[0.2, 0.4, 0.4],
                 [0.5, 0.4, 0.1]]

    # s then prob of A, B
    # t then prob of A, B

    # emit_prob = [[0.4, 0.6],
    #            [0.5, 0.5]]

    given_obs = ["1", "1"]

    # given_obs = ["A", "B", "B", "A"]

    hmm = Hmm(state_dict, obs_dict, init_prob, trans_prob, emit_prob, given_obs)

    foward = hmm.forward_algorithim()

    print(foward)



__end__ = '__end__'

if __name__ == '__main__':
    main()

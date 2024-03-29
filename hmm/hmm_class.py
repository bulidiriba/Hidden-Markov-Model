
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
        viterbi = np.zeros((self.time, self.state))
        backpointer = np.zeros((self.time))
        # loop through all the time sequence of the given observations
        for t in range(self.time):
            # loop through the current hidden state
            for s in range(self.state):
                su = []  # initialize the sum as zero
                # if the time is the first time use the initial probability
                # and we have no previous forward probability
                if t == 0:
                    # the probability
                    obs_indx = next((indx for indx, obs in self.obs_dict.items() if obs == self.given_obs[t]), None)
                    prob = self.init_prob[s] * self.emit_prob[s][obs_indx]
                    su.append(prob)

                # if the time is not the first time we use the transition probability,
                # also we use the previous forward probability
                else:
                    # loop through the previous state,
                    for i in range(self.state):
                        # the current forward probability is the multiplication of the previous forward probability
                        # with the path probability(the transition prob times the emission probability)
                        obs_indx = next((indx for indx, obs in self.obs_dict.items() if obs == self.given_obs[t]), None)
                        prob = viterbi[t - 1][i] * self.trans_prob[i][s] * self.emit_prob[s][obs_indx]
                        su.append(prob)

                # append the calculated forward probability to forward array
                viterbi[t][s] = max(su)

            viterbi_tolist = viterbi.tolist()
            indx = viterbi_tolist[t].index(max(viterbi_tolist[t]))
            backpointer[t] = indx

        return viterbi, backpointer

    def backward_algorithim(self):
        backward = np.zeros((self.time, self.state))
        # loop through all the time sequence of the given observations
        t = self.time - 1
        while t >= 0:
            # loop through the current hidden state
            for i in range(self.state):
                path_probs = []  # list that stores path probabilities
                # if the time is the first time use the initial probability
                # and we have no previous forward probability
                if t == self.time - 1:
                    # the probability of each path
                    prob = 1
                    path_probs.append(prob)

                # if the time is not the first time we use the transition probability,
                # also we use the previous forward probability
                else:
                    # loop through the previous state,
                    for j in range(self.state):
                        # the path probability is multiplication of the transition prob times the emission probability
                        obs_indx = next((indx for indx, obs in self.obs_dict.items() if obs == self.given_obs[t + 1]),
                                        None)
                        # prob = trans_prob[i][s] * emit_prob[s][obs_indx]
                        prob = self.trans_prob[i][j] * self.emit_prob[j][obs_indx]
                        # the forward probability from previous state is the multiplication of path probability times
                        # the forward probability of that previous state
                        bwrd_prob = backward[t + 1][j] * prob
                        # then append the calculated forward probability through each path to the path_probs
                        path_probs.append(bwrd_prob)

                # the forward probability of current state, with in a current time, is the sum of the total path probabilities
                backward[t][i] = sum(path_probs)
            t -= 1

        return backward

    def e_step(self):
        e = np.zeros((2 ** self.state, self.time - 1))

        forward = self.forward_algorithim()
        backward = self.backward_algorithim()

        for t in range(self.time - 1):
            prob = []
            for i in range(self.state):
                for j in range(self.state):
                    # print(state_dict.get(i), state_dict.get(j))
                    obs_indx = next((indx for indx, obs in self.obs_dict.items() if obs == self.given_obs[t + 1]), None)

                    numerator = forward[t][i] * self.trans_prob[i][j] * self.emit_prob[j][obs_indx] * backward[t + 1][j]
                    # denominator = sum(forward[time-1])
                    denominator = sum(forward[self.time - 1]) * sum(backward[0])
                    value = numerator / denominator
                    prob.append(value)

            e[:, t] = prob

        return e

    def m_step(self):
        new_trans_prob = np.zeros((2 ** self.state))
        e = self.e_step()

        num = sum(e[0])
        den = sum(e[0]) + sum(e[1])
        value = num / den
        new_trans_prob[0] = value

        num = sum(e[1])
        den = sum(e[0]) + sum(e[1])
        value = num / den
        new_trans_prob[1] = value

        num = sum(e[2])
        den = sum(e[2]) + sum(e[3])
        value = num / den
        new_trans_prob[2] = value

        num = sum(e[3])
        den = sum(e[2]) + sum(e[3])
        value = num / den
        new_trans_prob[3] = value

        return new_trans_prob

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

    viterbi, backpointer = hmm.viterbi_algorithim()

    print(viterbi)

    backward = hmm.backward_algorithim()

    print(backward)

    e = hmm.e_step()

    print(e)

    m = hmm.m_step()

    print(m.reshape(2, 2))





__end__ = '__end__'

if __name__ == '__main__':
    main()

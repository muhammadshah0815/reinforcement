# analysis.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    answerDiscount = 0.9
    answerNoise = 0.2
    answerNoise = 0.0
    return answerDiscount, answerNoise

def question3a():
    answerDiscount = 0.9  # Emphasize immediate rewards
    answerNoise = 0.2     # Maintain some risk-taking
    answerLivingReward = -2  # Penalize each step slightly to encourage taking the shortest path
    return answerDiscount, answerNoise, answerLivingReward

def question3b():
    answerDiscount = 0.5  # Value future safety
    answerNoise = 0.2     # Encourage careful movement
    answerLivingReward = -2  # Motivate ending the episode safely
    return answerDiscount, answerNoise, answerLivingReward

def question3c():
    answerDiscount = 0.9  # Far-sighted, values distant future rewards
    answerNoise = 0.001      # Some risk tolerance
    answerLivingReward = 0.002  # Penalize each step to strongly discourage taking risks
    return answerDiscount, answerNoise, answerLivingReward

def question3d():
    answerDiscount = 0.9  # Far-sighted
    answerNoise = 0.1    # Highly cautious
    answerLivingReward = 0.01  # Strongly discourage taking risks
    return answerDiscount, answerNoise, answerLivingReward


def question3e():
    answerDiscount = 1    # Future rewards still matter
    answerNoise = 0       # Actions are reliable
    answerLivingReward = 2  # High living reward to avoid ending the episode
    return answerDiscount, answerNoise, answerLivingReward

def question8():
    return 'NOT POSSIBLE'


if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))

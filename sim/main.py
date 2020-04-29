import logging

from decision_maker import ThresholdCheapDecisionMaker, FixedDecisionMaker
from environment import DecisionEnvironment

logging.basicConfig(level=logging.INFO)

default = DecisionEnvironment()


fdm = FixedDecisionMaker(default)
fdm.decide_all(num_gen=4, num_cheap_est=2, num_expensive_est=1)

# threshold
tcdm = ThresholdCheapDecisionMaker(default)
tcdm.decide_all(threshold=default.opt_mean * 1.2)

# threshold=None -> take the first heuristic
ttfdm = ThresholdCheapDecisionMaker(default)
ttfdm.decide_all()

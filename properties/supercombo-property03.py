from dnnv.properties import *
import numpy as np
N = Network("N")
x = Image(Parameter("image", type=str))
desire = np.zeros((1, 8)).astype('float32')
traffic_convention = np.zeros((1, 2)).astype('float32')
initial_state = np.zeros((1, 512)).astype('float32')

img_epsilon = float(Parameter("epsilon", type=str))
percent_margin = 0.1
output = N(x, desire, traffic_convention, initial_state)
lb0_sig = np.clip((1 / (1+np.exp(-output[0, 5857]))) - percent_margin, 0.001, 0.999)
lb0 = np.log(lb0_sig / (1-lb0_sig))
ub0_sig = np.clip(1 / (1+np.exp(-output[0, 5857])) + percent_margin, 0.001, 0.999)
ub0 = np.log(ub0_sig / (1-ub0_sig))
lb1_sig = np.clip((1 / (1+np.exp(-output[0, 5858]))) - percent_margin, 0.001, 0.999)
lb1 = np.log(lb1_sig / (1-lb1_sig))
ub1_sig = np.clip(1 / (1+np.exp(-output[0, 5858])) + percent_margin, 0.001, 0.999)
ub1 = np.log(ub1_sig / (1-ub1_sig))
lb2_sig = np.clip(1 / (1+np.exp(-output[0, 5859])) - percent_margin, 0.001, 0.999)
lb2 = np.log(lb2_sig / (1-lb2_sig))
ub2_sig = np.clip(1 / (1+np.exp(-output[0, 5859])) + percent_margin, 0.001, 0.999)
ub2 = np.log(ub2_sig / (1-ub2_sig))
Forall(
    x_,
    Implies(
        (0 <= x_ <= 255) &
        ((x - img_epsilon) < x_ < (x + img_epsilon)),
        And(lb0 <= N(x_, desire, traffic_convention, initial_state)[0, 5857] <= ub0,
            lb1 <= N(x_, desire, traffic_convention, initial_state)[0, 5858] <= ub1,
            lb2 <= N(x_, desire, traffic_convention, initial_state)[0, 5859] <= ub2,
        ),
    ),
)
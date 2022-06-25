from dnnv.properties import *
import numpy as np

N = Network("N")
x = Image(Parameter("image", type=str))
desire = np.zeros((1, 8)).astype('float32')
traffic_convention = np.zeros((1, 2)).astype('float32')
initial_state = np.zeros((1, 512)).astype('float32')

img_epsilon = float(Parameter("epsilon", type=str))
y_epsilon = 3.7 / 4
output = N(x, desire, traffic_convention, initial_state)

Forall(
    x_,
    Implies(
        (0 <= x_ <= 255) &
        ((x - img_epsilon) < x_ < (x + img_epsilon)),
        And(-y_epsilon < (N(x_, desire, traffic_convention, initial_state)[0, 5556] - output[0, 5556]) < y_epsilon,
            -y_epsilon < (N(x_, desire, traffic_convention, initial_state)[0, 5217] - output[0, 5217]) < y_epsilon,
        ),
    ),
)
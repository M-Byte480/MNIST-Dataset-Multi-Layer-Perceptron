import numpy as np
import sklearn as sk

# Exercise 1 : Hard code weights (ie: no learning)
def unitStep(activation):
    if activation >= 0:
        return 1
    else:
        return 0


def perceptronModel(inputs, weights, bias):
    activation_array = np.dot(inputs, weights) + bias
    signal = unitStep(activation_array)
    return signal


# OR Function
def logicFunctionOR(inputs):
    weights = [1, 1]  # 2 nodes
    bias = -.5

    return perceptronModel(inputs, weights, bias)


def logicFunctionAND(inputs):
    weights = [0.5, 0.5]
    bias = -1

    return perceptronModel(inputs, weights, bias)


def logicFunctionXOR(inputs):
    gate1_and = logicFunctionAND(inputs)
    gate2_or = logicFunctionOR(inputs)

    return perceptronModel(inputs, weights, bias)



test1 = np.array([0, 0])
test2 = np.array([0, 1])
test3 = np.array([1, 0])
test4 = np.array([1, 1])

#
# print("OR({}, {}) = {}".format(0, 0, logicFunctionOR(test1)))
# print("OR({}, {}) = {}".format(0, 1, logicFunctionOR(test2)))
# print("OR({}, {}) = {}".format(1, 0, logicFunctionOR(test3)))
# print("OR({}, {}) = {}".format(1, 1, logicFunctionOR(test4)))
#


# Exercise 2: Learning the AND gate
def learningAND():
    weights = np.random.rand(1,3) * 10
    w_1 = np.round(weights[0][0], 1)
    w_2 = np.round(weights[0][1], 1)
    bias = np.round(weights[0][2], 1)

    inputs = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]

    inputs_array = np.asarray(inputs)

    expected_output = inputs_array[:,1] * inputs_array[:,0] # Exercise 2 : AND
    # expected_output = [0, 1, 1, 0] # Exercise 3: XOR
    print(expected_output)

    error_vector = np.array([0,0,0,0])
    for i in range(len(inputs)):
        input_values = inputs[i]
        output = perceptronModel(input_values, np.asarray([w_1, w_2]), bias)
        error_vector[i] = expected_output[i] - output

    E = np.sum(error_vector)

    print(E)

    max_iteration = 1_000
    t = 1
    learning_rate = 0.1
    #Animation
    vals = [[w_1, w_2, bias]]

    ## Do we reach our iter count first or Error count gets fixed
    while t < max_iteration and E != 0:
        for i in range(len(inputs)):
            output = perceptronModel(inputs[i], np.asarray([w_1, w_2]), bias)
            error_vector[i] = -1 if expected_output[i] != output else 0
            w_1 =   w_1  + learning_rate * error_vector[i] * inputs[i][0] # first input
            w_2 =   w_2  + learning_rate * error_vector[i] * inputs[i][1] # second input
            bias =  bias + learning_rate * error_vector[i]

        vals.append([w_1, w_2, bias])
        E = np.sum(error_vector)
        # print('Sum of errors: ', E)
        print(f"{t} W1: {w_1}, W2: {w_2}, bias: {bias}")
        t += 1
    print(E)
    print(vals)

    for i in range(len(inputs)):
        print(unitStep(np.dot(np.asarray([w_1, w_2]), inputs[i]) + bias))
    print(w_1, w_2, bias)

## Exercise 2
learningAND()

## Exercise 3


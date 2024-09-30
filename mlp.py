import network as n
from house import *

multi_layer_perceptron = n.Network([2, 6, 10, 1])

data_set = [
    House(2_000, 5, 10_000),
    House(3_000, 7, 25_000),
    House(1_000, 3, 8_000),
    House(2_000, 6, 12_500),
    House(3_000, 4, 18_000),
    House(6_000, 12, 40_000),
    House(2_000, 7, 20_000),
    House(4_000, 6, 27_000),
]

cross_validation = [
    House(2_500, 5, 15_000),
    House(3_500, 8, 30_000),
    House(5_000, 9, 35_000)
]

training_data = [(( i.area,i.rooms), i.price) for i in data_set]
test_data = [((i.area, i.rooms), i.price) for i in cross_validation]

mini_batch_size = [ i[0] for i in training_data]

learning_rate = 0.2
epoch = 1_000 #iterations

multi_layer_perceptron.SGD(
    training_data = training_data,
    epochs = epoch,
    mini_batch_size= 4,
    learning_rate= learning_rate,
    test_data = test_data
)
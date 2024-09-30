import mnist_loader
import network
# from numba import jit, cuda
# from timeit import default_timer as timer

# @jit(nopython=True)
def running_on_gpu():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    net = network.Network([784, 30, 10])
    learning_rate = 3.0
    epoch = 30          #iterations
    mini_batch_size = 5

    net.SGD(
        training_data = training_data,
        epochs = epoch,
        mini_batch_size= mini_batch_size,
        learning_rate= learning_rate,
        test_data = test_data
    )

running_on_gpu()

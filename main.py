import tensorflow as tf
import numpy as np
import Perceptron as Per


def skalyr():
    rank_0_tensor = tf.constant(4)
    print(rank_0_tensor)


def rank_tensor_1():
    rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
    print(rank_1_tensor)


def rank_tensor_2():
    rank_2_tensor = tf.constant([[1, 2],
                                 [3, 4],
                                 [5, 6]], dtype=tf.float16)
    print(rank_2_tensor)
    np.array(rank_2_tensor)
    rank_2_tensor.numpy()


def rank_tensor_3():
    rank_3_tensor = tf.constant([
        [[0, 1, 2, 3, 4],
         [5, 6, 7, 8, 9],
         [5, 6, 7, 8, 9]],
        [[10, 11, 12, 13, 14],
         [10, 11, 12, 13, 14],
         [15, 16, 17, 18, 19]],
        [[20, 21, 22, 23, 24],
         [20, 21, 22, 23, 24],
         [25, 26, 27, 28, 29]], ])

    print(rank_3_tensor)

def rank_tensor_():
    a = tf.constant([[1, 6],
                     [3, 2]])
    b = tf.constant([[2, 1],
                     [5, 6]])

    # print(tf.add(a, b), "\n")
    print(tf.multiply(a, b), "\n")
    # print(tf.matmul(a, b), "\n")


if __name__ == "__main__":
    cool_perceptron = Per.Perceptron()

    # Define your inputs as a single list
    inputs = [24, 55]

    # Calculate the weighted sum
    print(cool_perceptron.weighted_sum([24, 55]))
    print(cool_perceptron.activation(52))
    # print(cool_perceptron.activation(52))



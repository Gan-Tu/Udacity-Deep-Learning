"""
This script builds and runs a graph with miniflow.

(x + y) + y
"""

from __future__ import print_function
import numpy as np
from miniflow import *


############################ TEST 1 ############################ 
print("Test 1 .........")

x, y = Input(), Input()

f = Add(x, y)

feed_dict = {x: 10, y: 5}

sorted_nodes = topological_sort(feed_dict)
output = forward_pass(f, sorted_nodes)

# NOTE: because topological_sort set the values for the `Input` nodes we could also access
# the value for x with x.value (same goes for y).
print("{} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], output))



############################ TEST 2 ############################ 
print("Test 2 .........")

inputs, weights, bias = Input(), Input(), Input()

f = Linear(inputs, weights, bias)

feed_dict = {
    inputs: [6, 14, 3],
    weights: [0.5, 0.25, 1.4],
    bias: 2
}

graph = topological_sort(feed_dict)
output = forward_pass(f, graph)

print("should be 12.7 with this example: " + str(output))



############################ TEST 3 ############################ 
print("Test 3 .........")


X, W, b = Input(), Input(), Input()

f = Linear(X, W, b)

X_ = np.array([[-1., -2.], [-1, -2]])
W_ = np.array([[2., -3], [2., -3]])
b_ = np.array([-3., -5])

feed_dict = {X: X_, W: W_, b: b_}

graph = topological_sort(feed_dict)
output = forward_pass(f, graph)

print("Output should be:\nExpected: \n[[-9., 4.],\n[-9., 4.]]\nActual:")
print(output)

############################ TEST 4 ############################ 
print("Test 4 .........")


X, W, b = Input(), Input(), Input()

f = Linear(X, W, b)
g = Sigmoid(f)

X_ = np.array([[-1., -2.], [-1, -2]])
W_ = np.array([[2., -3], [2., -3]])
b_ = np.array([-3., -5])

feed_dict = {X: X_, W: W_, b: b_}

graph = topological_sort(feed_dict)
output = forward_pass(g, graph)

print("Output should be:\nExpected:\n[[  1.23394576e-04   9.82013790e-01]\n \
    [  1.23394576e-04   9.82013790e-01]]\nActual:")
print(output)

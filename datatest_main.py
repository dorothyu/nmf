#!/usr/bin/env python

from numpy import dot
from numpy import random
from nmf import nmf

# Define constants
Matrix_ORDER = 5
Matrix_RANK = 2

# Generate a target Matrix
origin_W = random.randint(0,9,size=(Matrix_RANK, Matrix_ORDER))
origin_H = random.randint(0,9,size=(Matrix_ORDER, Matrix_RANK))
v = dot(origin_W, origin_H) 

# Generate initial Matrix W and H (Random generation)
w = random.randint(0,9,size=(Matrix_RANK, Matrix_ORDER))
h = random.randint(0,9,size=(Matrix_ORDER, Matrix_RANK))

# Calculate W&H by multiple iteration
(output_W, output_H) = nmf(v, w, h, 0.001, 50, 100)

# Show Results
print ("\n==Original_Matrix==")
print ("Matrix_W :\n", origin_W) 
print ("Matrix_H :\n", origin_H)
print ("Target_V :\n", v)

print ("\n===Output_Matrix===")
print ("Matrix_W :\n", output_W)
print ("Matrix_H :\n", output_H)
print ("Output_V :\n", dot(output_W, output_H))

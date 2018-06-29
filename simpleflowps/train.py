#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Optimizer classes for parameters optimization.
'''
from WorkerClient import WorkerClient
from .operations import Operation, compute_gradients

class GradientDescentOptimizer(object):
    ''' Optimizer that implements the gradient descent algorithm.
    '''
    def __init__(self, learning_rate):
        ''' Construct a new gradient descent optimizer

        :param learning_rate: learning rate of optimizier.
        :type learning_rate: float
        '''
        self.learning_rate = learning_rate

    def minimize(self, loss):
        ''' Generate an gradient descent optimization operation for loss.

        :param loss: The loss operation to be optimized.
        :type loss: Object of `Operation`
        '''
        learning_rate = self.learning_rate

        class MinimizationOperation(Operation):
            def compute_output(self):
                # Get gradient table.
                grad_table = compute_gradients(loss)
                should_update = False

                # Iterate all trainable variables in graph.
                for var in DEFAULT_GRAPH.trainable_variables:
                    if var in grad_table:
                        grad = grad_table[var]
                        ret = DEFAULT_PS.push(var.name, grad)
                        if 'behind' in ret:
                            should_update = True
                            break

                if should_update:
                    for var in DEFAULT_GRAPH.trainable_variables:
                        var.output_value = DEFAULT_PS.pull(var.name)

        return MinimizationOperation()

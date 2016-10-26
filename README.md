# Neural Networks

## Overview

This project is contains:
- a model for a neural network
- a trainer for that neural network
- a helper class to use easily use and train the neural network
- a loader class to manipulate CSV data files for training/testing the network
- and a demo of the functionality

The core of the neural network and the trainer are modeled after the content from the tutorial, ["Neural Networks Demystified"](https://www.youtube.com/watch?v=bxe2T-V8XRs). I would definitely reccomend the series for getting started with neural networks!

---

## Demo

You will need numpy and scipy to run this project.

To run the demo, navigate to the /code/ folder and type 'python demo.py' in your console.

You'll see some optimization jargon in the terminal, then a window will pop up like this:

![screenshot][screenshot]

What am I looking at?

The neural network has just pulled data from the /data/concrete.csv data file in the project. The file contains data for concrete (age, water content, etc.) as well as its strength. 

The network trains itself using the first half of the data. The output column for this is the concrete's strength while the other columns serve as input fields.

Using the second half of the input data, the network then attempts to make calculations for the output data. It then compares its guesses with the actual output data and displays them in that graph. You can see that in this trial, the network had an average error of 15.71%.

[screenshot]: https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Visualization"


---

## About this neural network

This Neural Network has one input layer, one hidden layer, and on output layer
of nodes, each with a customizable quantity of nodes. You can also set a penalty
for the neural networks model complexity which will help reduce overfitting.

When the network is first made, all the connections' weights are set to random
values between 0 and 1. To train the network, instantiate a Trainer object with
the network to be trained as a parameter. Run Trainer.train() with valid input
and output data to train the network.

---

## Restrictions and problems

* To the best of my knowledge, it appears that input and output values must be
fed in between 0 and 1.
* Neural network can only handle fully numerical data.

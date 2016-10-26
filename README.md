#Neural Networks

This project is closely modeled after the content from the tutorial "Neural
Networks Demystified" - https://www.youtube.com/watch?v=bxe2T-V8XRs

---

You will need numpy and scipy for this project.

To run the demo just type 'python demo.py' in a console.

---

This Neural Network has one input layer, one hidden layer, and on output layer
of nodes, each with a customizable quantity of nodes. You can also set a penalty
for the neural networks model complexity which will help reduce overfitting.

When the network is first made, all the connections' weights are set to random
values between 0 and 1. To train the network, instantiate a Trainer object with
the network to be trained as a parameter. Run Trainer.train() with valid input
and output data to train the network.

---

Restrictions and Problems:

* To the best of my knowledge, it appears that input and output values must be
fed in between 0 and 1.

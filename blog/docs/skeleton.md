# [WIP] Towards efficient multi-GPU training in Keras over TensorFlow

In this article I'd like to summarize the current state of attempts for data-parallel multi-GPU training in Keras over TensorFlow and a roadmap of steps to make it efficient and usable in practice.

We will cover two problems: efficiently training on a single GPU and extending training to multiple GPUs on the same machine.

We started our experiments in July 2017 and in the meanwhile there appeared some open-source experiments that seem to solve the scaling problem, either on different backend (MXNet, CNTK) or even using TensorFlow.

So let's try to dive in the problem, see the landscape of of solutions and review a bunch of measurements in practice to see what's working and what doesn't.

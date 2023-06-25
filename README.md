# Synchronous vs Asynchronous Reinforcement Learning in a Real-World Robot

## Abstract
In recent times, reinforcement learning (RL) with physical
robots has attracted the attention of a wide range of researchers.
However, state-of-the-art RL algorithms do not consider that
physical environments do not wait for the RL agent to
make decisions or updates. RL agents learn by periodically
conducting computationally expensive gradient updates. When
decision-making and gradient update tasks are carried out
sequentially by the RL agent in a physical robot, it significantly
increases the agent’s response time. In a rapidly changing
environment, this increased response time may be detrimental to
the performance of the learning agent. Asynchronous RL methods,
which separate the computation of decision-making and gradient
updates, are a potential solution to this problem. However, only
a few comparisons between asynchronous and synchronous RL
have been made with physical robots. For this reason, the exact
performance benefits of using asynchronous RL methods over
synchronous RL methods are still unclear. In this study,we provide a
performance comparison between asynchronous and synchronous
RL using a physical robotic arm called Franka Emika Panda.
Our experiments show that the agents learn faster and attain
significantly more returns using asynchronous RL. Our experiments
also demonstrate that the learning agent with a faster response
time performs better than the agent with a slower response time,
even if the agent with a slower response time performs a higher
number of gradient updates.

## RESULTS

![image](https://github.com/fahimfss/rl655_franka/assets/8725869/bb4d6411-5fc0-48b5-8654-d308a19daabe)


We show the learning curves of Asynchronous Baseline, Asynchronous
High-Resolution, Synchronous Baseline, and Synchronous
High-Resolution settings in Figure 6. We conducted twenty-five independent
runs for each setting, resulting in 100 independent runs
and 200 hours of learning with the Franka Emika Panda robotic arm.
The experiments show that the performance of Asynchronous Baseline
and Asynchronous High-Resolution is significantly better than
the synchronous counterparts. The Asynchronous High-Resolution
even managed to perform better than the Synchronous Baseline.
This fact becomes more interesting if we consider Figure 7. Figure
7 shows that the Asynchronous High-Resolution performed fewer
learning updates than the synchronous variants. We conclude that
the better performance of asynchronous versions is primarily due
to lower action cycle time (or faster response time) and a higher
number of samples collected for learning. The number of gradient
updates has a less significant effect on performance.  

**Please check out the full report here:** [Report](https://github.com/fahimfss/rl655_franka/blob/main/results/RL655.pdf) 

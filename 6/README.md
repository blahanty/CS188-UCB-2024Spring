Project 6: Reinforcement Learning
=================================================================================

Due: **Friday, April 26, 11:59 PM PT**.

* * *

Table of contents
--------------------------------------------------

*   Introduction
*   Spring 2024 Change
*   MDPs
*   Question 1 (5 points): Value Iteration
*   Question 2 (5 points): Policies
*   Question 3 (5 points): Q-Learning
*   Question 4 (2 points): Epsilon Greedy
*   Question 5 (1 point): Q-Learning and Pacman
*   Question 6 (3 points): Approximate Q-Learning
*   Question 7 (4 points): Deep Q-Learning
*   Submission

* * *

Introduction
----------------------------------------

In this project, you will implement value iteration and Q-learning. You will test your agents first on Gridworld (from class), then apply them to a simulated robot controller (Crawler) and Pacman.

As in previous projects, this project includes an autograder for you to grade your solutions on your machine. This can be run on all questions with the command:

```
python autograder.py
```

It can be run for one particular question, such as q2, by:

```
python autograder.py -q q2
```

It can be run for one particular test by commands of the form:

```
python autograder.py -t test_cases/q2/1-bridge-grid
```

Spring 2024 Change
----------------------------------------------------

This semester, just like the previous project, we will be having two versions available. They will be identical except for the final question, which is to train a deep q learning model. You will have the option of doing this question in Pytorch, or the custom neural network library that was used in the original version of the ML project. Both solutions will be very similar, so it will most likely be easier for you to continue with the version that you used for the previous project. The code for this project contains the following files, available as a [zip archive](https://inst.eecs.berkeley.edu/~cs188/sp24/assets/projects/reinforcement.zip), and the pytorch version can be found [here](https://inst.eecs.berkeley.edu/~cs188/sp24/assets/projects/reinforcement_pytorch.zip).

<table><tbody><tr><td colspan="2"><b>Files you'll edit:</b></td></tr><tr><td><code>valueIterationAgents.py</code></td><td>A value iteration agent for solving known MDPs.</td></tr><tr><td><code>qlearningAgents.py</code></td><td>Q-learning agents for Gridworld, Crawler and Pacman.</td></tr><tr><td><code>analysis.py</code></td><td>A file to put your answers to questions given in the project.</td></tr><tr><td colspan="2"><b>Files you might want to look at:</b></td></tr><tr><td><code>mdp.py</code></td><td>Defines methods on general MDPs.</td></tr><tr><td><code>learningAgents.py</code></td><td>Defines the base classes <code>ValueEstimationAgent</code> and <code>QLearningAgent</code>, which your agents will extend.</td></tr><tr><td><code>util.py</code></td><td>Utilities, including <code>util.Counter</code>, which is particularly useful for Q-learners.</td></tr><tr><td><code>gridworld.py</code></td><td>The Gridworld implementation.</td></tr><tr><td><code>featureExtractors.py</code></td><td>Classes for extracting features on (state, action) pairs. Used for the approximate Q-learning agent (in <code>qlearningAgents.py</code>).</td></tr><tr><td colspan="2"><b>Supporting files you can ignore:</b></td></tr><tr><td><code>environment.py</code></td><td>Abstract class for general reinforcement learning environments. Used by <code>gridworld.py</code>.</td></tr><tr><td><code>graphicsGridworldDisplay.py</code></td><td>Gridworld graphical display.</td></tr><tr><td><code>graphicsUtils.py</code></td><td>Graphics utilities.</td></tr><tr><td><code>textGridworldDisplay.py</code></td><td>Plug-in for the Gridworld text interface.</td></tr><tr><td><code>crawler.py</code></td><td>The crawler code and test harness. You will run this but not edit it.</td></tr><tr><td><code>graphicsCrawlerDisplay.py</code></td><td>GUI for the crawler robot.</td></tr><tr><td><code>autograder.py</code></td><td>Project autograder</td></tr><tr><td><code>testParser.py</code></td><td>Parses autograder test and solution files</td></tr><tr><td><code>testClasses.py</code></td><td>General autograding test classes</td></tr><tr><td><code>test_cases/</code></td><td>Directory containing the test cases for each question</td></tr><tr><td><code>reinforcementTestClasses.py</code></td><td>Project 3 specific autograding test classes</td></tr></tbody></table>

**Files to Edit and Submit**: You will fill in portions of `valueIterationAgents.py`, `qlearningAgents.py`, and `analysis.py` during the assignment. Once you have completed the assignment, you will submit these files to Gradescope (for instance, you can upload all `.py` files in the folder). Please do not change the other files in this distribution.

**Evaluation**: Your code will be autograded for technical correctness. Please do not change the names of any provided functions or classes within the code, or you will wreak havoc on the autograder. However, the correctness of your implementation - not the autograder's judgements - will be the final judge of your score. If necessary, we will review and grade assignments individually to ensure that you receive due credit for your work.

**Academic Dishonesty**: We will be checking your code against other submissions in the class for logical redundancy. If you copy someone else's code and submit it with minor changes, we will know. These cheat detectors are quite hard to fool, so please don't try. We trust you all to submit your own work only; please don't let us down. If you do, we will pursue the strongest consequences available to us.

**Getting Help**: You are not alone! If you find yourself stuck on something, contact the course staff for help. Office hours, section, and the discussion forum are there for your support; please use them. If you can't make our office hours, let us know and we will schedule more. We want these projects to be rewarding and instructional, not frustrating and demoralizing. But, we don't know when or how to help unless you ask.

**Discussion**: Please be careful not to post spoilers.

* * *

MDPs
------------------------

To get started, run Gridworld in manual control mode, which uses the arrow keys:

```
python gridworld.py -m
```

You will see the two-exit layout from class. The blue dot is the agent. Note that when you press up, the agent only actually moves north 80% of the time. Such is the life of a Gridworld agent!

You can control many aspects of the simulation. A full list of options is available by running:

```
python gridworld.py -h
```

The default agent moves randomly

```
python gridworld.py -g MazeGrid
```

You should see the random agent bounce around the grid until it happens upon an exit. Not the finest hour for an AI agent.

Note: The Gridworld MDP is such that you first must enter a pre-terminal state (the double boxes shown in the GUI) and then take the special 'exit' action before the episode actually ends (in the true terminal state called `TERMINAL_STATE`, which is not shown in the GUI). If you run an episode manually, your total return may be less than you expected, due to the discount rate (`-d` to change; 0.9 by default).

Look at the console output that accompanies the graphical output (or use `-t` for all text). You will be told about each transition the agent experiences (to turn this off, use `-q`).

As in Pacman, positions are represented by `(x, y)` Cartesian coordinates and any arrays are indexed by `[x][y]`, with `'north'` being the direction of increasing `y`, etc. By default, most transitions will receive a reward of zero, though you can change this with the living reward option (`-r`).

* * *

Question 1 (5 points): Value Iteration
-----------------------------------------------------------------------------------------

Recall the value iteration state update equation:

$$V_{k+1}(s) \leftarrow \max_a \sum_{s'} T(s,a,s')\left[R(s,a,s') + \gamma V_k(s')\right]$$

Write a value iteration agent in `ValueIterationAgent`, which has been partially specified for you in `valueIterationAgents.py`. Your value iteration agent is an offline planner, not a reinforcement learning agent, and so the relevant training option is the number of iterations of value iteration it should run (option `-i`) in its initial planning phase. `ValueIterationAgent` takes an MDP on construction and runs value iteration for the specified number of iterations before the constructor returns.

Value iteration computes $k$-step estimates of the optimal values, $V_k$. In addition to `runValueIteration`, implement the following methods for `ValueIterationAgent` using $V_k$:

*   `computeActionFromValues(state)` computes the best action according to the value function given by self.values.
*   `computeQValueFromValues(state, action)` returns the Q-value of the (state, action) pair given by the value function given by `self.values`.

These quantities are all displayed in the GUI: values are numbers in squares, Q-values are numbers in square quarters, and policies are arrows out from each square.

Important: Use the "batch" version of value iteration where each vector $V_k$ is computed from a fixed vector $V_{k-1}$ (like in lecture), not the "online" version where one single weight vector is updated in place. This means that when a state's value is updated in iteration $k$ based on the values of its successor states, the successor state values used in the value update computation should be those from iteration $k-1$ (even if some of the successor states had already been updated in iteration $k$). The difference is discussed in [Sutton & Barto](https://web.archive.org/web/20230417150626/https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) in Chapter 4.1 on page 91.

_Note_: A policy synthesized from values of depth $k$ (which reflect the next $k$ rewards) will actually reflect the next $k+1$ rewards (i.e. you return $\pi_{k+1}$). Similarly, the Q-values will also reflect one more reward than the values (i.e. you return $Q_{k+1}$).

You should return the synthesized policy $\pi_{k+1}$.

_Hint_: You may optionally use the `util.Counter` class in `util.py`, which is a dictionary with a default value of zero. However, be careful with `argMax`: the actual argmax you want may be a key not in the counter!

_Note_: Make sure to handle the case when a state has no available actions in an MDP (think about what this means for future rewards).

To test your implementation, run the autograder:

```
python autograder.py -q q1
```

The following command loads your `ValueIterationAgent`, which will compute a policy and execute it 10 times. Press a key to cycle through values, Q-values, and the simulation. You should find that the value of the start state (`V(start)`, which you can read off of the GUI) and the empirical resulting average reward (printed after the 10 rounds of execution finish) are quite close.

```
python gridworld.py -a value -i 100 -k 10
```

_Hint_: On the default `BookGrid`, running value iteration for 5 iterations should give you this output:

```
python gridworld.py -a value -i 5
```

![](https://inst.eecs.berkeley.edu/~cs188/sp24/assets/projects/value_iter_diagram.png)

_Grading_: Your value iteration agent will be graded on a new grid. We will check your values, Q-values, and policies after fixed numbers of iterations and at convergence (e.g. after 100 iterations).

* * *

Question 2 (5 points): Policies
---------------------------------------------------------------------------

Consider the `DiscountGrid` layout, shown below. This grid has two terminal states with positive payoff (in the middle row), a close exit with payoff +1 and a distant exit with payoff +10. The bottom row of the grid consists of terminal states with negative payoff (shown in red); each state in this "cliff" region has payoff -10. The starting state is the yellow square. We distinguish between two types of paths: (1) paths that "risk the cliff" and travel near the bottom row of the grid; these paths are shorter but risk earning a large negative payoff, and are represented by the red arrow in the figure below. (2) paths that "avoid the cliff" and travel along the top edge of the grid. These paths are longer but are less likely to incur huge negative payoffs. These paths are represented by the green arrow in the figure below.

![](https://inst.eecs.berkeley.edu/~cs188/sp24/assets/projects/value_2_paths.png)

In this question, you will choose settings of the discount, noise, and living reward parameters for this MDP to produce optimal policies of several different types. **Your setting of the parameter values for each part should have the property that, if your agent followed its optimal policy without being subject to any noise, it would exhibit the given behavior.** If a particular behavior is not achieved for any setting of the parameters, assert that the policy is impossible by returning the string `'NOT POSSIBLE'`.

Here are the optimal policy types you should attempt to produce:

1.  Prefer the close exit (+1), risking the cliff (-10)
2.  Prefer the close exit (+1), but avoiding the cliff (-10)
3.  Prefer the distant exit (+10), risking the cliff (-10)
4.  Prefer the distant exit (+10), avoiding the cliff (-10)
5.  Avoid both exits and the cliff (so an episode should never terminate)

To see what behavior a set of numbers ends up in, run the following command to see a GUI:

```
python gridworld.py -g DiscountGrid -a value --discount [YOUR_DISCOUNT] --noise [YOUR_NOISE] --livingReward [YOUR_LIVING_REWARD]
```

To check your answers, run the autograder:

```
python autograder.py -q q2
```

`question2a()` through `question2e()` should each return a 3-item tuple of `(discount, noise, living reward)` in `analysis.py`.

_Note_: You can check your policies in the GUI. For example, using a correct answer to 3(a), the arrow in (0,1) should point east, the arrow in (1,1) should also point east, and the arrow in (2,1) should point north.

_Note_: On some machines you may not see an arrow. In this case, press a button on the keyboard to switch to qValue display, and mentally calculate the policy by taking the arg max of the available qValues for each state.

_Grading_: We will check that the desired policy is returned in each case.

* * *

Question 3 (5 points): Q-Learning
-------------------------------------------------------------------------------

Note that your value iteration agent does not actually learn from experience. Rather, it ponders its MDP model to arrive at a complete policy before ever interacting with a real environment. When it does interact with the environment, it simply follows the precomputed policy (e.g. it becomes a reflex agent). This distinction may be subtle in a simulated environment like a Gridworld, but it's very important in the real world, where the real MDP is not available.

You will now write a Q-learning agent, which does very little on construction, but instead learns by trial and error from interactions with the environment through its `update(state, action, nextState, reward)` method. A stub of a Q-learner is specified in `QLearningAgent` in `qlearningAgents.py`, and you can select it with the option `'-a q'`. For this question, you must implement the `update`, `computeValueFromQValues`, `getQValue`, and `computeActionFromQValues` methods.

_Note_: For `computeActionFromQValues`, you should break ties randomly for better behavior. The `random.choice()` function will help. In a particular state, actions that your agent hasn't seen before still have a Q-value, specifically a Q-value of zero, and if all of the actions that your agent has seen before have a negative Q-value, an unseen action may be optimal.

_Important_: Make sure that in your `computeValueFromQValues` and `computeActionFromQValues` functions, you only access Q values by calling `getQValue`. This abstraction will be useful for question 10 when you override `getQValue` to use features of state-action pairs rather than state-action pairs directly.

With the Q-learning update in place, you can watch your Q-learner learn under manual control, using the keyboard:

```
python gridworld.py -a q -k 5 -m
```

Recall that `-k` will control the number of episodes your agent gets to learn. Watch how the agent learns about the state it was just in, not the one it moves to, and "leaves learning in its wake." Hint: to help with debugging, you can turn off noise by using the `--noise 0.0` parameter (though this obviously makes Q-learning less interesting). If you manually steer Pacman north and then east along the optimal path for four episodes, you should see the following Q-values:

![](https://inst.eecs.berkeley.edu/~cs188/sp24/assets/projects/q_learning.png)

_Grading_: We will run your Q-learning agent and check that it learns the same Q-values and policy as our reference implementation when each is presented with the same set of examples. To grade your implementation, run the autograder:

```
python autograder.py -q q3
```

* * *

Question 4 (2 points): Epsilon Greedy
---------------------------------------------------------------------------------------

Complete your Q-learning agent by implementing epsilon-greedy action selection in `getAction`, meaning it chooses random actions an epsilon fraction of the time, and follows its current best Q-values otherwise. Note that choosing a random action may result in choosing the best action - that is, you should not choose a random sub-optimal action, but rather any random legal action.

You can choose an element from a list uniformly at random by calling the `random.choice` function. You can simulate a binary variable with probability `p` of success by using `util.flipCoin(p)`, which returns `True` with probability `p` and `False` with probability `1-p`.

After implementing the `getAction` method, observe the following behavior of the agent in `GridWorld` (with epsilon = 0.3).

```
python gridworld.py -a q -k 100
```

Your final Q-values should resemble those of your value iteration agent, especially along well-traveled paths. However, your average returns will be lower than the Q-values predict because of the random actions and the initial learning phase.

You can also observe the following simulations for different epsilon values. Does that behavior of the agent match what you expect?

```
python gridworld.py -a q -k 100 --noise 0.0 -e 0.1
```

```
python gridworld.py -a q -k 100 --noise 0.0 -e 0.9
```

To test your implementation, run the autograder:

```
python autograder.py -q q4
```

With no additional code, you should now be able to run a Q-learning crawler robot:

```
python crawler.py
```

If this doesn't work, you've probably written some code too specific to the `GridWorld` problem and you should make it more general to all MDPs.

This will invoke the crawling robot from class using your Q-learner. Play around with the various learning parameters to see how they affect the agent's policies and actions. Note that the step delay is a parameter of the simulation, whereas the learning rate and epsilon are parameters of your learning algorithm, and the discount factor is a property of the environment.

* * *

Question 5 (1 point): Q-Learning and Pacman
---------------------------------------------------------------------------------------------------

Time to play some Pacman! Pacman will play games in two phases. In the first phase, _training_, Pacman will begin to learn about the values of positions and actions. Because it takes a very long time to learn accurate Q-values even for tiny grids, Pacman's training games run in quiet mode by default, with no GUI (or console) display. Once Pacman's training is complete, he will enter _testing_ mode. When testing, Pacman's `self.epsilon` and `self.alpha` will be set to 0.0, effectively stopping Q-learning and disabling exploration, in order to allow Pacman to exploit his learned policy. Test games are shown in the GUI by default. Without any code changes you should be able to run Q-learning Pacman for very tiny grids as follows:

```
python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid
```

Note that `PacmanQAgent` is already defined for you in terms of the `QLearningAgent` you've already written. `PacmanQAgent` is only different in that it has default learning parameters that are more effective for the Pacman problem (`epsilon=0.05, alpha=0.2, gamma=0.8`). You will receive full credit for this question if the command above works without exceptions and your agent wins at least 80% of the time. The autograder will run 100 test games after the 2000 training games.

_Hint_: If your `QLearningAgent` works for `gridworld.py` and `crawler.py` but does not seem to be learning a good policy for Pacman on `smallGrid`, it may be because your `getAction` and/or `computeActionFromQValues` methods do not in some cases properly consider unseen actions. In particular, because unseen actions have by definition a Q-value of zero, if all of the actions that have been seen have negative Q-values, an unseen action may be optimal. Beware of the `argMax` function from `util.Counter`!

To grade your answer, run:

```
python autograder.py -q q5
```

_Note_: If you want to experiment with learning parameters, you can use the option `-a`, for example `-a epsilon=0.1,alpha=0.3,gamma=0.7`. These values will then be accessible as `self.epsilon`, `self.gamma` and `self.alpha` inside the agent.

_Note_: While a total of 2010 games will be played, the first 2000 games will not be displayed because of the option `-x 2000`, which designates the first 2000 games for training (no output). Thus, you will only see Pacman play the last 10 of these games. The number of training games is also passed to your agent as the option `numTraining`.

_Note_: If you want to watch 10 training games to see what's going on, use the command:

```
python pacman.py -p PacmanQAgent -n 10 -l smallGrid -a numTraining=10
```

During training, you will see output every 100 games with statistics about how Pacman is faring. Epsilon is positive during training, so Pacman will play poorly even after having learned a good policy: this is because he occasionally makes a random exploratory move into a ghost. As a benchmark, it should take between 1000 and 1400 games before Pacman's rewards for a 100 episode segment becomes positive, reflecting that he's started winning more than losing. By the end of training, it should remain positive and be fairly high (between 100 and 350).

Make sure you understand what is happening here: the MDP state is the exact board configuration facing Pacman, with the now complex transitions describing an entire ply of change to that state. The intermediate game configurations in which Pacman has moved but the ghosts have not replied are not MDP states, but are bundled in to the transitions.

Once Pacman is done training, he should win very reliably in test games (at least 90% of the time), since now he is exploiting his learned policy.

However, you will find that training the same agent on the seemingly simple `mediumGrid` does not work well. In our implementation, Pacman's average training rewards remain negative throughout training. At test time, he plays badly, probably losing all of his test games. Training will also take a long time, despite its ineffectiveness.

Pacman fails to win on larger layouts because each board configuration is a separate state with separate Q-values. He has no way to generalize that running into a ghost is bad for all positions. Obviously, this approach will not scale.

* * *

Question 6 (3 points): Approximate Q-Learning
-------------------------------------------------------------------------------------------------------

Implement an approximate Q-learning agent that learns weights for features of states, where many states might share the same features. Write your implementation in `ApproximateQAgent` class in `qlearningAgents.py`, which is a subclass of `PacmanQAgent`.

_Note_: Approximate Q-learning assumes the existence of a feature function $f(s,a)$ over state and action pairs, which yields a vector $[f_1(s,a), \dots, f_i(s,a), \dots, f_n(s,a)]$ of feature values. We provide feature functions for you in `featureExtractors.py`. Feature vectors are `util.Counter` (like a dictionary) objects containing the non-zero pairs of features and values; all omitted features have value zero. So, instead of an vector where the index in the vector defines which feature is which, we have the keys in the dictionary define the identity of the feature.

The approximate Q-function takes the following form:

$$Q(s,a)=\sum_{i=1}^n f_i(s,a) w_i$$

where each weight $w_i$ is associated with a particular feature $f_i(s,a)$. In your code, you should implement the weight vector as a dictionary mapping features (which the feature extractors will return) to weight values. You will update your weight vectors similarly to how you updated Q-values:

$$w_i \leftarrow w_i + \alpha \cdot \text{difference} \cdot f_i(s,a)$$
$$\text{difference} = \left( r + \gamma \max_{a'}Q(s',a') \right) - Q(s,a)$$

Note that the $\text{difference}$ term is the same as in normal Q-learning, and $r$ is the experienced reward.

By default, `ApproximateQAgent` uses the `IdentityExtractor`, which assigns a single feature to every `(state,action)` pair. With this feature extractor, your approximate Q-learning agent should work identically to `PacmanQAgent`. You can test this with the following command:

```
python pacman.py -p ApproximateQAgent -x 2000 -n 2010 -l smallGrid
```

_Important_: `ApproximateQAgent` is a subclass of `QLearningAgent`, and it therefore shares several methods like `getAction`. Make sure that your methods in `QLearningAgent` call `getQValue` instead of accessing Q-values directly, so that when you override `getQValue` in your approximate agent, the new approximate q-values are used to compute actions.

Once you're confident that your approximate learner works correctly with the identity features, run your approximate Q-learning agent with our custom feature extractor, which can learn to win with ease:

```
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumGrid
```

Even much larger layouts should be no problem for your `ApproximateQAgent` (_warning_: this may take a few minutes to train):

```
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumClassic
```

If you have no errors, your approximate Q-learning agent should win almost every time with these simple features, even with only 50 training games.

_Grading_: We will run your approximate Q-learning agent and check that it learns the same Q-values and feature weights as our reference implementation when each is presented with the same set of examples. To grade your implementation, run the autograder:

```
python autograder.py -q q6
```

* * *

Question 7 (4 points): Deep Q-Learning
-----------------------------------------------------------------------------------------

For the final project question of the semester, you will combine concepts from Q-learning earlier in this project and ML from the previous project. In model.py, you will implement DeepQNetwork, which is a neural network that predicts the Q values for all possible actions given a state.

You will implement the following functions:

1.  `__init__()`: Just like in Project 5, you will initialize all the parameters of your neural network here. You must also initialize the following variables:
    1.  `self.parameters`: A list containing all your parameters in order of your forward pass.
    2.  `self.learning_rate`: You will use this in gradient\_update().
    3.  `self.numTrainingGames`: The number of games that Pacman will play to collect transitions from and learn its Q values; note that this should be greater than 1000, since roughly the first 1000 games are used for exploration and are not used to update the Q network.
    4.  `self.batch_size`: The number of transitions the model should use for each gradient update. The autograder will use this variable; you should not need to access this variable after setting it.
2.  `get_loss()`: Return the square loss between predicted Q values (outputted by your network), and the Q\_targets (which you will treat as the ground truth).
3.  `run()`/`forward()`: Similar to the method of the same name in Project 5, where you will return the result of a forward pass through your Q network. (The output should be a vector of size (batch\_size, num\_actions), since we want to return the Q value for all possible actions given a state.) If you are using pytorch, you will need to fill out the `forward()` method, while the original project will have you use `run()`
4.  `gradient_update()` : Iterate through your self.parameters and update each of them according to the computed gradients. However, unlike project 5, you are not iterating over the entire dataset in this function, nor are you repeatedly updating the parameters until convergence. This function should only perform a single gradient update for each parameter. The autograder will repeatedly call this function to update your network.

For the `gradient_update()`, if you choose to use the pytorch version, we recommend using the `SGD` optimizer rather than `Adam` as you did in the ML project. Both are used exactly the same way, but `SGD` tends to perform a bit better in this instance. You are also more than welcome to try out different optimizers for your solution, but `SGD` is what the staff solution uses and performs relatively well.

Grading: We will run your Deep Q learning Pacman agent for 10 games after your agent trains on `self.numTrainingGames` games. If your agent wins at least 6/10 of the games, then you will receive full credit. If your agent wins at least 8/10 of the games, then you will receive 1 extra credit point (5/4). Please note that deep Q learning is not known for its stability, despite some of the tricks we have implemented in the backend training loop. The number of games your agent wins may vary for each run. To achieve the extra credit point, your implementation should consistently beat the 80% threshold.

```
python autograder.py -q q7
```

Submission
------------------------------------

In order to submit your project upload the Python files you edited. For instance, use Gradescope's upload on all `.py` files in the project folder.

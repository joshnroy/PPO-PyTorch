# Textworld RL Agent

## Goal

- Learn to play the [textworld](https://www.microsoft.com/en-us/research/project/textworld/) games (maybe _tabla rusa_) using Deep RL (specifically PPO)

## Algorithm Sketch

While the game continues:

1. Read input
2. Use language model to make into a real-valued observation
3. Use LSTM to make sequential observations into a state
4. Use PPO to decide on real-valued action based on current state
5. Use language model to translate real-valued action into text action
6. Submit to game, get new observation and reward

## Steps

- [ ] Very small, single deterministic textworld
- [ ] Large, single deterministic textworld
- [ ] Very small, set of deterministic textworlds
- [ ] Large, set of deterministic textworlds
- [ ] Do better on the MSR challenge (if they released the validation set)

## Questions

- What is the language model to use?
- Do I need pre-training?
- Can I use the language model to get a real valued "state"?


## PPO-PyTorch (Previous Readme)
Minimal PyTorch implementation of Proximal Policy Optimization with clipped objective for OpenAI gym environments.

## Usage

- To test a preTrained network : run `test.py` or `test_continuous.py`
- To train a new network : run `PPO.py` or `PPO_continuous.py`
- All the hyperparameters are in the `PPO.py` or `PPO_continuous.py` file
- If you are trying to train it on a environment where action dimension = 1, make sure to check the tensor dimensions in the update function of PPO class, since I have used `torch.squeeze()` quite a few times. `torch.squeeze()` squeezes the tensor such that there are no dimensions of length = 1 ([more info](https://pytorch.org/docs/stable/torch.html?highlight=torch%20squeeze#torch.squeeze)).
- Number of actors for collecting experience = 1. This could be easily changed by creating multiple instances of ActorCritic networks in the PPO class and using them to collect experience (like A3C and standard PPO).

## Dependencies
Trained and tested on:
```
Python 3.6
PyTorch 1.0
NumPy 1.15.3
gym 0.10.8
Pillow 5.3.0
```

## Results

PPO Discrete LunarLander-v2 (1200 episodes)           |  PPO Continuous BipedalWalker-v2 (4000 episodes)
:-------------------------:|:-------------------------:
![](https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/gif/PPO_LunarLander-v2.gif) |  ![](https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/gif/PPO_BipedalWalker-v2.gif)


## References

- PPO [paper](https://arxiv.org/abs/1707.06347)
- [OpenAI Spinning up](https://spinningup.openai.com/en/latest/)

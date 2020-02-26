import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import gym
import numpy as np
import textworld.gym
import re

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class StateExtractor(nn.Module):
    def __init__(self, max_words, state_dim):
        super(StateExtractor, self).__init__()
        self.embedding = nn.Embedding(max_words, state_dim)
        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=state_dim//2, batch_first=True, num_layers=1, bidirectional=True)
    
    def forward(self, x):
        out = self.embedding(x)
        out, hidden = self.lstm(out)
        out = torch.sum(out, dim=1)
        return out

class ActionGenerator(nn.Module):
    def __init__(self, max_words, state_dim):
        super(ActionGenerator, self).__init__()
        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=state_dim, batch_first=True, num_layers=1)
        self.out = nn.Linear(state_dim, max_words)
    
    def forward(self, x):
        # TODO: What should the input be here?
        out = x.unsqueeze(1).expand(x.shape[0], 10, x.shape[1])
        out, hidden = self.lstm(out)
        out = F.softmax(self.out(out), dim=-1)

        return out

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std, max_words):
        super(ActorCritic, self).__init__()
        self.max_words = max_words

        action_generator = ActionGenerator(max_words, state_dim)
        state_extractor = StateExtractor(max_words, state_dim)

        # action mean range -1 to 1
        self.actor =  nn.Sequential(
                state_extractor,
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, action_dim),
                nn.ReLU(),
                action_generator
                )
        # critic
        self.critic = nn.Sequential(
                state_extractor,
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
                )
        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)
        
    def forward(self):
        raise NotImplementedError
    
    # TODO: This needs to be changed to fit the NLP thing
    def act(self, state, memory):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        
        return action.detach()
    
    # TODO: This needs to be changed to fit the NLP thing
    def evaluate(self, state, action):   
        action_mean = self.actor(state)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.max_words = 10_000
        
        self.policy = ActorCritic(state_dim, action_dim, action_std, max_words=self.max_words).to(device)
        self.optimizer = torch.optim.SparseAdam(self.policy.parameters(), lr=lr, betas=betas)
        
        self.policy_old = ActorCritic(state_dim, action_dim, action_std, max_words=self.max_words).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

        self.word_to_idx = {"<END>": 0, "<START>": 1}
        self.idx_to_word = {0: "<END>", 1: "<START>"}
    
    def select_action(self, state, memory):
        state = state.lower() + " <STOP>"
        state = re.findall(r"[\w]+|[.,!?;']", state)
        # state = [x for x in state if x != '']
        idx_state = []
        for s in state:
            if s not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx[s] = idx
                self.idx_to_word[idx] = s
            idx_state.append(self.word_to_idx[s])
        idx_state = np.array(idx_state)
        idx_state = torch.LongTensor(idx_state.reshape(1, -1)).to(device)
        # onehot_state = F.one_hot(idx_state, self.max_words)
        return self.policy_old.act(idx_state, memory).cpu().data.numpy().flatten()
    
    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = reward
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
def main():
    ############## Hyperparameters ##############
    # Register a text-based game as a new Gym's environment.
    # env_id = textworld.gym.register_game("tw_games/custom_game.ulx", max_episode_steps=50, request_infos=textworld.core.EnvInfos(admissible_commands=True))
    env_id = textworld.gym.register_game("tw_games/custom_game.ulx", max_episode_steps=50)
    render = False
    solved_reward = 1         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 10000        # max training episodes
    max_timesteps = 50        # max timesteps in one episode
    
    update_timestep = max_timesteps * 4      # update policy every n timesteps
    action_std = 0.5            # constant std for action distribution (Multivariate Normal)
    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor
    
    lr = 0.0003                 # parameters for Adam optimizer
    betas = (0.9, 0.999)
    
    random_seed = None
    #############################################
    
    # creating environment
    env = gym.make(env_id)
    state_dim = 128
    action_dim = 128
    
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    
    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0
    
    # training loop
    for i_episode in range(1, max_episodes+1):
        state, _ = env.reset()
        for t in range(max_timesteps):
            time_step +=1
            # Running policy_old:
            action = ppo.select_action(state, memory)
            state, reward, done, _ = env.step(action)
            
            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            # update if its time
            if time_step % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
            running_reward += reward
            if render:
                env.render()
            if done:
                break
        
        avg_length += t
        
        # stop training if avg_reward > solved_reward
        if running_reward >= (log_interval*solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_continuous_solved_{}.pth'.format(env_name))
            break
        
        # save every 500 episodes
        if i_episode % 500 == 0:
            torch.save(ppo.policy.state_dict(), './PPO_continuous_{}.pth'.format(env_name))
            
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))
            
            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
            
if __name__ == '__main__':
    main()
    

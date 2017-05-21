import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
from threading import Thread

class Config:
    beta = .1
    lr = .1

class PolicyNetwork():
    def __init__(self, config, scope):
        self.config = config
        
        config.state_shape, num_actions
        self.beta = .1
        self.lr = .1
        self.num_actions = num_actions
        self.grad_clip = 10
        self.scope = scope
        self.rnn_hidden_size = 100
        
        init_params = {
            weights_initializer: tf.contrib.layers.xavier_initializer(),
            biases_initializer: tf.zeros_initializer()
        }
        
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.uint8, shape=(None,) + self.config.state_shape)
            self.rnn_state = tf.placeholder(tf.uint8, shape=(None,) + self.config.rnn_hidden_size) # hidden size?
            
            self.policy, self.value, self.new_rnn_state = make_architecture(state, rnn_state)
            self.optimizer, self.gradients = make_gradients(policy, value, state, action, future_reward)
    
    def make_architecture(self, state, rnn_state):
        out = state
        for i in range(4):
            out = tf.contrib.layers.conv2d(
                inputs=out,
                kernel_size=3,
                num_outputs=32,
                stride=2,
                padding="SAME",
                activation_fn=tf.nn.elu,
                **init_params)

        lstm = tf.contrib.rnn.BasicLSTMCell(256)
        out = tf.contrib.layers.flatten(inputs=out)
        out, new_state = lstm(rnn_state, out)

        policy = tf.contrib.layers.fully_connected(
            inputs=out,
            num_outputs=num_actions,
            activation_fn=tf.nn.softmax,
            **init_params
        )
        value = tf.contrib.layers.fully_connected(
            inputs=out,
            num_outputs=1,
            activation_fn=None,
            **init_params
        )
        return policy, value, new_state
    
    def make_gradients(self, policy, value, state, action, future_reward):
        log_policy = tf.log(policy)
        entropy = -tf.einsum('ij,ij->i', policy, log_policy)
        action_log_prob = tf.einsum('ij,ij->i', log_policy, tf.one_hot(action, self.config.num_actions))
        advantage = future_reward - value
        
        policy_loss = tf.reduce_mean(action_log_prob*advantage + self.config.beta*entropy)
        value_loss = tf.reduce_mean(advantage**2)
        loss = policy_loss + value_loss
    
        optimizer = tf.train.AdamOptimizer(self.lr)
        gradients, variables = zip(*optimizer.compute_gradients(loss, var_list=get_variables(self.scope)))
        gradients, _ = tf.clip_by_global_norm(grads, self.grad_clip)
        return optimizer, gradients
        
    def copy_from(self, sess, other_policy_network):
        src = get_variables(other_policy_network.scope)
        target = get_variables(self.scope)
        
        assignments = [tf.assign(t, s) for s, t in zip(src, target)]
        return sess.run(tf.group(*assignments))        
    
    def get_policy(self, sess, state, rnn_state):
        return sess.run([self.policy, self.value, self.new_rnn_state], {
            self.state: state,
            self.rnn_state: rnn_state
        })
    
    def apply_gradients(self, sess, grads):
        return sess.run(self.optimizer.apply_gradients(zip(grads, get_variables(self.scope))))

def get_variables(scope):
    return sorted(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope), key=lambda v: v.name)

def worker(coord, sess, histories, policy_network, max_episode_len=20):
    env = gym.make('CartPole-v0')
    local_policy_network = PolicyNetwork(policy_network.config)
    
    while not coord.should_stop():
        local_policy_network.copy_from(sess, policy_network)
        
        state = env.reset()
        rnn_state = np.zeros()
        
        history = []
        done = False
        value = 0
        for _ in xrange(max_episode_len):
            policy, value, rnn_state = local_policy_network.get_policy(sess, state, rnn_state)
            action = np.random.choice(num_actions, p=policy)
            obs, reward, done, _ = env.step(action)
            
            history.append((state, action, reward, value, rnn_state))
            if done:
                break
        
        future_reward = 0 if done else value
        gradients = []
        for state, action, reward, value, rnn_state in reversed(history):
            future_reward = reward + gamma * future_reward
            gradients.append(local_policy_network.get_gradients(value, state, action, future_reward))
        
        total_gradients = [sum(step[i] for step in gradients) for i in range(len(gradients[0]))]
        policy_network.apply_gradients(sess, total_gradients)

def main():
    env = gym.make('')
    env.action_space.n
    env.observation_space.n
    env = wrappers.Monitor(env, 'results/experiments')
    # preprocess frames?
    
    policy_network = PolicyNetwork(scope='global')
    sess = tf.Session()
    
    # Initialize workers
    coord = tf.train.Coordinator()

    threads = [Thread(target=worker, args=(coord, sess)) for i in xrange(10)]
    for t in threads:
        t.start()
    
    coord.join(threads)

main()

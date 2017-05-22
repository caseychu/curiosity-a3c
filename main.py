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
        self.config.rnn_hidden_size = 256
        
        init_params = {
            weights_initializer: tf.contrib.layers.xavier_initializer(),
            biases_initializer: tf.zeros_initializer()
        }
        
        with tf.variable_scope(scope):
            self.states = tf.placeholder(tf.uint8, shape=(None,) + self.config.state_shape)
            self.rnn_state = tf.placeholder(tf.uint8, shape=(None,) + self.config.rnn_hidden_size) # hidden size?
            
            self.policy, self.value, self.new_rnn_state = make_architecture(states, rnn_state)
            self.optimizer, self.gradients = make_gradients(policy, value, state, action, future_reward)
    
    def make_architecture(self, states, initial_rnn_state):
        out = states
        for i in range(4):
            out = tf.contrib.layers.conv2d(
                inputs=out,
                kernel_size=3,
                num_outputs=32,
                stride=2,
                padding="SAME",
                activation_fn=tf.nn.elu,
                **init_params)

        out = tf.contrib.layers.flatten(inputs=out)
        out, final_rnn_state = tf.nn.dynamic_rnn(
            tf.contrib.rnn.BasicLSTMCell(self.config.rnn_hidden_size),
            tf.expand_dims(states, 0),
            initial_state=tf.expand_dims(initial_rnn_state, 0))
        out = tf.squeeze(out, [0])
        
        policy_logits = tf.contrib.layers.fully_connected(
            inputs=out,
            num_outputs=num_actions,
            activation_fn=None,
            **init_params
        )
        values = tf.contrib.layers.fully_connected(
            inputs=out,
            num_outputs=1,
            activation_fn=None,
            **init_params
        )
        
        log_policies = tf.log_softmax(policy_logits)
        policies = tf.softmax(policy_logits)
        return policies, values, final_rnn_state
    
    def get_next_action(self, state, rnn_state):
       
    
    # Make a loss from a rollout; the arguments should all be placeholders
    def make_loss(self, states, actions, estimated_values, empirical_values):
        
        length = len(states)
        discounting_matrix = np.fromfunction(lambda i,j: self.gamma**(j-i)*np.less_equal(i,j), shape=(length,length))
        empirical_values = tf.matmul(discounting_matrix, rewards)
        
        policy_loss = tf.einsum('ij,ij,i->',
            log_policies,
            tf.one_hot(actions, self.config.num_actions),
            empirical_values - estimated_values)
        
        entropy_loss = -(-tf.einsum('ij,ij->', policies, log_policies))
        
        value_loss = tf.nn.l2_loss(empirical_values - values)
        
        return policy_loss + entropy_loss + value_loss
    
    
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

def worker(coord, sess, policy_network, max_episode_len=20):
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
            new_state, reward, done, _ = env.step(action)
            
            history.append((state, action, reward, value))
            state = new_state
            if done:
                break
        
        if done:
            future_reward = 0
        else:
            _, value, _ = local_policy_network.get_policy(sess, state, rnn_state)
            future_reward = value
        
        gradients = []
        for state, action, reward, value, rnn_state in reversed(history):
            future_reward = reward + gamma * future_reward
            gradients.append(local_policy_network.get_gradients(value, state, action, future_reward, rnn_state))
        
        total_gradients = map(sum, zip(*gradients))
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
    threads = [Thread(target=worker, args=(coord, sess, policy_network)) for i in xrange(10)]
    for t in threads:
        t.start()
    
    coord.join(threads)

main()

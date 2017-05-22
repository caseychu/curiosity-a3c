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
            self.states = states = tf.placeholder(tf.uint8, shape=(None,) + self.config.state_shape)
            self.rnn_state = rnn_state = tf.placeholder(tf.uint8, shape=(None,) + self.config.rnn_hidden_size)
            
            # The main architecture: 4 conv layers into an LSTM, into two fully connected layers.
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
            cell = tf.contrib.rnn.BasicLSTMCell(self.config.rnn_hidden_size)
            out, next_rnn_state = tf.nn.dynamic_rnn(
                cell,
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
            
            self.zero_rnn_state = cell.zero_state(1)
            self.next_rnn_state = next_rnn_state
            self.log_policies = tf.log_softmax(policy_logits)
            self.policies = tf.softmax(policy_logits)     
            self.values = values

            # Gradients.
            self.actions = actions = tf.placeholder(tf.uint8, shape=(None,))
            self.estimated_values = estimated_values = tf.placeholder(tf.float32, shape=(None,))
            self.empirical_values = empirical_values = tf.placeholder(tf.float32, shape=(None,))
            
            policy_loss = tf.einsum('ij,ij,i->',
                log_policies,
                tf.one_hot(actions, self.config.num_actions),
                empirical_values - estimated_values)
            entropy_loss = -(-tf.einsum('ij,ij->', policies, log_policies))
            value_loss = tf.nn.l2_loss(empirical_values - values)
            loss = policy_loss + entropy_loss + value_loss
            
            gradients = tf.gradients(loss, get_variables(self.scope))
            gradients, _ = tf.clip_by_global_norm(grads, self.grad_clip)
            
            self.gradients = gradients
            self.loss = loss
            self.optimizer = tf.train.AdamOptimizer(self.lr)
        
    def copy_from(self, sess, other_policy_network):
        src = get_variables(other_policy_network.scope)
        target = get_variables(self.scope)
        
        assignments = [tf.assign(t, s) for s, t in zip(src, target)]
        return sess.run(tf.group(*assignments))        
    
    def get_next_policy(self, sess, state, rnn_state):
        if rnn_state is None:
            rnn_state = self.zero_rnn_state
            
        policies, values, rnn_state = sess.run([self.policies, self.values, self.new_rnn_state], {
            self.state: [state],
            self.rnn_state: rnn_state
        })
        return policies[0], values[0], rnn_state
    
    def get_gradients_from_rollout(self, sess, states, actions, estimated_values, empirical_values):
        return sess.run(self.gradients, feed_dict={
            self.states: states,
            self.rnn_state: self.zero_rnn_state,
            self.actions: actions,
            self.estimated_values: estimated_values,
            self.empirical_values: empirical_values
        })
    
    def apply_gradients(self, sess, grads):
        return sess.run(self.optimizer.apply_gradients(zip(grads, get_variables(self.scope))))

def get_variables(scope):
    return sorted(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope), key=lambda v: v.name)

def worker(coord, sess, policy_network, max_episode_length=20):
    env = gym.make('CartPole-v0')
    local_policy_network = PolicyNetwork(policy_network.config)
    
    while not coord.should_stop():
        local_policy_network.copy_from(sess, policy_network)
        rnn_state = None
        state = env.reset()
        
        history = []
        done = False
        value = 0
        for _ in xrange(max_episode_length):
            policy, value, rnn_state = local_policy_network.get_next_policy(sess, state, rnn_state)
            action = np.random.choice(num_actions, p=policy)
            new_state, reward, done, _ = env.step(action)
            
            history.append((state, action, value, reward))
            state = new_state
            if done:
                break
        
        future_rewards = [0 if done else value]
        for _, _, reward, _, _ in reversed(history[:-1]):
            future_rewards.append(reward + policy_network.config.gamma * future_rewards[-1])
        
        empirical_values = np.array(reversed(future_rewards))
        states, actions, rewards, estimated_values = map(np.array, zip(*history))
        
        gradients = local_policy_network.get_gradients_from_rollout(sess, states, actions, estimated_values, empirical_values)
        policy_network.apply_gradients(sess, gradients)

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

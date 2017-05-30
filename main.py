import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
from threading import Thread
import sys

import time
from test_env import EnvTest

class Config:
    num_actions = 1
    state_shape = (1,)
    
    beta = .1
    lr = 1e-4
    grad_clip = 10
    rnn_hidden_size = 256
    gamma = .99

class PolicyNetwork():
    def __init__(self, config, scope, parent_network=None):
        self.config = config
        self.scope = scope

        if parent_network is None:
            parent_network = self
        
        init_params = {
            "weights_initializer": tf.contrib.layers.xavier_initializer(),
            "biases_initializer": tf.zeros_initializer()
        }
        
        with tf.variable_scope(scope):
            self.states = states = tf.placeholder(tf.float32, shape=(None,) + self.config.state_shape)
            
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
                
            cell = tf.contrib.rnn.BasicLSTMCell(self.config.rnn_hidden_size)
            rnn_state = cell.zero_state(1, tf.float32)
            
            out = tf.contrib.layers.flatten(inputs=out)
            out, next_rnn_state = tf.nn.dynamic_rnn(cell, tf.expand_dims(out, 0), initial_state=rnn_state)
            out = tf.squeeze(out, [0])

            policy_logits = tf.contrib.layers.fully_connected(
                inputs=out,
                num_outputs=self.config.num_actions,
                activation_fn=None,
                **init_params
            )
            log_policies = tf.nn.log_softmax(policy_logits)
            policies = tf.nn.softmax(policy_logits)
            values = tf.contrib.layers.fully_connected(
                inputs=out,
                num_outputs=1,
                activation_fn=None,
                **init_params
            )
            values = tf.squeeze(values, -1)
            
            self.rnn_state = rnn_state
            self.next_rnn_state = next_rnn_state
            self.log_policies = log_policies
            self.policies = policies
            self.values = values
            
            # Synchronizing variables.
            all_variables = sorted(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope), key=lambda v: v.name)
            self.all_variables = all_variables
            
            sync_op = tf.group(*[tf.assign(t, s) for s, t in zip(parent_network.all_variables, all_variables)])
            self.sync_op = sync_op

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
            
            grads = tf.gradients(loss, all_variables)
            grads, grad_norm = tf.clip_by_global_norm(grads, self.config.grad_clip)
            optimizer = tf.train.AdamOptimizer(self.config.lr)
            train_op = optimizer.apply_gradients(zip(grads, parent_network.all_variables))
            
            self.grads = grads
            self.grad_norm = grad_norm
            self.loss = loss
            self.optimizer = optimizer
            self.train_op = train_op
            
            # Summaries
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('gradient norm', grad_norm)
            self.summaries = tf.summary.merge_all()
        
    def synchronize(self, sess):
        return sess.run(self.sync_op)
    
    def get_next_policy(self, sess, state, rnn_state):
        if rnn_state is None:
            policies, values, next_rnn_state = sess.run([self.policies, self.values, self.next_rnn_state], {
                self.states: [state]
            })
        else:
            policies, values, next_rnn_state = sess.run([self.policies, self.values, self.next_rnn_state], {
                self.states: [state],
                self.rnn_state[0]: rnn_state[0],
                self.rnn_state[1]: rnn_state[1]
            })
            
        return policies[0], values[0], next_rnn_state
    
    def apply_gradients_from_rollout(self, sess, states, actions, estimated_values, empirical_values):
        g, gn, l, _ = sess.run([self.grads, self.grad_norm, self.loss, self.train_op], feed_dict={
            self.states: states,
            self.actions: actions,
            self.estimated_values: estimated_values,
            self.empirical_values: empirical_values
        })
        #print self.scope, 'grad norm: ', gn, '; loss: ', l
        sys.stdout.flush()
        return g

def worker(coord, sess, policy_network, env, max_episode_length=20):
    with coord.stop_on_exception():
        while not coord.should_stop():
            policy_network.synchronize(sess)
            rnn_state = None
            state = env.reset()

            history = []
            done = False
            value = 0
            for _ in xrange(max_episode_length):
                policy, value, rnn_state = policy_network.get_next_policy(sess, state, rnn_state)
                action = np.random.choice(policy_network.config.num_actions, p=policy)
                new_state, reward, done, _ = env.step(action)

                history.append((state, action, value, reward))
                state = new_state
                if done:
                    break

            states, actions, estimated_values, rewards = map(np.array, zip(*history))
            print policy_network.scope, 'true reward: ', np.sum(rewards)

            future_rewards = [0 if done else value]
            for reward in reversed(rewards.tolist()[:-1]):
                future_rewards.append(reward + policy_network.config.gamma * future_rewards[-1])
            empirical_values = np.array(list(reversed(future_rewards)))

            policy_network.apply_gradients_from_rollout(sess, states, actions, estimated_values, empirical_values)


def main():
    make_env = lambda: EnvTest((5, 5, 1))
    #gym.make('Pong-v0')
        
    env = make_env()
    config = Config()
    config.num_actions = env.action_space.n
    config.state_shape = env.observation_space.shape
    
    #env = wrappers.Monitor(env, 'results/experiments')
    # preprocess frames?
    
    policy_network = PolicyNetwork(config, 'global')
    
    sess = tf.Session()
    coord = tf.train.Coordinator()
    threads = []
    for i in xrange(2):
        local_env = make_env()
        local_policy_network = PolicyNetwork(config, 'worker' + str(i), parent_network=policy_network)
        threads.append(Thread(target=worker, args=(coord, sess, local_policy_network, local_env)))
    
    sess.run(tf.global_variables_initializer())
    
    try:
        for t in threads:
            t.start()
        coord.join(threads, stop_grace_period_secs=5)
    except KeyboardInterrupt:
        coord.request_stop()
        raise

main()

import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
from threading import Thread
import sys

class Config:
    num_actions = 1
    state_shape = (1,)
    
    beta = .1
    lr = .1
    grad_clip = 10
    rnn_hidden_size = 256
    gamma = .99

class PolicyNetwork():
    def __init__(self, config, scope):
        self.config = config
        self.scope = scope
        
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
            self.all_variables = sorted(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope), key=lambda v: v.name)
            print [v.name for v in self.all_variables]

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
            
            grads = tf.gradients(loss, self.all_variables)
            grads, grad_norm = tf.clip_by_global_norm(grads, self.config.grad_clip)
            
            self.grads = grads
            self.grad_norm = grad_norm
            self.loss = loss
            self.optimizer = tf.train.AdamOptimizer(self.config.lr)
            _ = self.optimizer.minimize(loss) # We need to create this op before tf.global_variables_initializer
            
            # Summaries
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('gradient norm', grad_norm)
            self.summaries = tf.summary.merge_all()
        
    def copy_from(self, sess, other_policy_network):
        src = other_policy_network.all_variables
        target = self.all_variables
        
        assignments = [tf.assign(t, s) for s, t in zip(src, target)]
        return sess.run(tf.group(*assignments))        
    
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
    
    def get_gradients_from_rollout(self, sess, states, actions, estimated_values, empirical_values):
        g, l = sess.run([self.grads, self.grad_norm], feed_dict={
            self.states: states,
            self.actions: actions,
            self.estimated_values: estimated_values,
            self.empirical_values: empirical_values
        })
        print 'loss: ', l
        sys.stdout.flush()
        return g
    
    def apply_gradients(self, sess, grads):
        return sess.run(self.optimizer.apply_gradients(zip(grads, self.all_variables)))

def worker(coord, worker_id, sess, graph, policy_network, max_episode_length=20):
    with graph.as_default():
        env = gym.make('Breakout-v0')
        local_policy_network = PolicyNetwork(policy_network.config, worker_id)
    
        while not coord.should_stop():
            local_policy_network.copy_from(sess, policy_network)
            rnn_state = None
            state = env.reset()

            history = []
            done = False
            value = 0
            for _ in xrange(max_episode_length):
                policy, value, rnn_state = local_policy_network.get_next_policy(sess, state, rnn_state)
                action = np.random.choice(local_policy_network.config.num_actions, p=policy)
                new_state, reward, done, _ = env.step(action)

                history.append((state, action, value, reward))
                state = new_state
                if done:
                    break

            states, actions, estimated_values, rewards = map(np.array, zip(*history))
            print 'true reward: ', np.sum(rewards)
            
            future_rewards = [0 if done else value]
            for reward in reversed(rewards.tolist()[:-1]):
                future_rewards.append(reward + policy_network.config.gamma * future_rewards[-1])
            empirical_values = np.array(list(reversed(future_rewards)))
            
            gradients = local_policy_network.get_gradients_from_rollout(sess, states, actions, estimated_values, empirical_values)
            policy_network.apply_gradients(sess, gradients)

def main():
    env = gym.make('Breakout-v0')
    
    config = Config()
    config.num_actions = env.action_space.n
    config.state_shape = env.observation_space.shape
    
    #env = wrappers.Monitor(env, 'results/experiments')
    # preprocess frames?
    
    policy_network = PolicyNetwork(config, 'global')
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # Initialize workers
    graph = tf.get_default_graph()
    coord = tf.train.Coordinator()
    
    threads = [Thread(target=worker, args=(coord, 'worker' + str(i), sess, graph, policy_network)) for i in xrange(1)]
    with coord.stop_on_exception():
        for t in threads:
            t.start()
        coord.join(threads)

main()

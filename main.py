import tensorflow as tf
import numpy as np

class PolicyNetwork():
    def __init__(self, state_shape, num_actions, scope, reuse=False):
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
        
        with tf.variable_scope(scope, reuse=reuse):
            self.state = tf.placeholder(tf.uint8, shape=(None,) + self.state_shape)
            self.rnn_state = tf.placeholder(tf.uint8, shape=(None,) + self.rnn_hidden_size)
            
            self.policy, self.value = make_architecture(state, rnn_state)
            self.gradients = make_gradients(policy, value, state, action, future_reward)
    
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
        return policy, value
    
    def make_gradients(self, policy, value, state, action, future_reward):
        log_policy = tf.log(policy)
        entropy = -tf.einsum('ij,ij->i', policy, log_policy)
        action_log_prob = tf.einsum('ij,ij->i', log_policy, tf.one_hot(action, self.num_actions))
        advantage = future_reward - value
        
        policy_loss = tf.reduce_mean(action_log_prob*advantage + self.beta*entropy)
        value_loss = tf.reduce_mean(advantage**2)
        loss = policy_loss + value_loss
    
        optimizer = tf.train.AdamOptimizer(self.lr)
        grads, variables = zip(*optimizer.compute_gradients(loss,
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)))
        grads, _ = tf.clip_by_global_norm(grads, self.grad_clip)
        return optimizer.apply_gradients(zip(grads, variables))
        
    def copy_from(self, sess, scope):
        src = sorted(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope), key=lambda v: v.name)
        target = sorted(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope), key=lambda v: v.name)
        
        assignments = [tf.assign(t, s) for s, t in zip(src, target)]
        return sess.run(tf.group(*assignments))
    
    def get_policy(self, sess, state, rnn_state=np.zeros()):
        return sess.run(self.policy, {
            self.state: state,
            self.rnn_state: rnn_state
        })
    
    def apply_gradients(self, sess, history):
        return sess.run(self.apply, gr)

class PolicyExplorer():
    pass

def main():
    
    # Initialize actor-critic network
    
    # Initialize workers
    
    # Apply async updates

    pass

main()

# Make Tensorflow behave like Numpy!
def lift(construct_graph):
    def construct(**placeholders):
        out = construct_graph(**placeholders)
        def run(sess, **kwargs):
            return sess.run(out, feed_dict=dict(zip(placeholders, kwargs)))
        return run    
    return construct
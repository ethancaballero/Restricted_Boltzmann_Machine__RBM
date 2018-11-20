import tensorflow as tf
import numpy as np
from dist_util import sample_bernoulli
from util import save_merged_images
import os


class RBM:
    def __init__(self, args):
        """  This function initializes an instance of the RBM class.
        Args:
            args: parse_args input command-line arguments (hyperparameters).
        """
        self.args = args

        self.n_visible = self.args.n_vis
        self.n_hidden = self.args.n_hid

        # define which distribution type v & h each sample from
        if self.args.dist_type_vis == "bernoulli":
            self.presample_v_distribution = tf.nn.sigmoid
            self.sample_v_distribution = sample_bernoulli
        else:
            raise NotImplementedError
        if self.args.dist_type_hid == "bernoulli":
            self.presample_h_distribution = tf.nn.sigmoid
            self.sample_h_distribution = sample_bernoulli
        else:
            raise NotImplementedError

        # create input placeholders
        self.vp = tf.placeholder(tf.float32, [None, self.n_visible], name="visible_input_placeholder")
        self.hp = tf.placeholder(tf.float32, [None, self.n_hidden], name="hidden_input_placeholder")

        # create parameters of rbm
        self.W = tf.Variable(tf.truncated_normal(
            [self.n_visible, self.n_hidden], mean=0.0, stddev=0.05, dtype=tf.float32), name="weight_matrix")
        self.a = tf.Variable(tf.truncated_normal(
            [self.n_visible], mean=self.args.vb_mean, stddev=0.02, dtype=tf.float32), name="visible_bias")
        self.b = tf.Variable(tf.truncated_normal(
            [self.n_hidden], mean=self.args.hb_mean, stddev=0.02, dtype=tf.float32), name="hidden_bias")

        # create optimizer
        if self.args.adam:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr, epsilon=1e-4)
        else:
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.args.lr)

        # construct graph(s)
        self.train_op = self._training(grads=self._contrastive_divergence())
        self.reconstruction = self._reconstruct(self.vp)
        self.reconstruction_error = tf.reduce_mean(tf.square(self.vp - self.reconstruction))
        self.inferred_hidden_activations = self.sample_h_distribution(self._prob_h_given_v(self.vp))
        self.v_marg = self._gibbs_sample_v_prime_given_v(self.vp, steps=self.args.v_marg_steps)

        self._idx_pll = 0  # index used for pll calculation
        self.pll = self._pseudo_log_likelihood()

        # init variables
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _prob_h_given_v(self, visible_probs):
        """ p(h|v)
        """
        return self.presample_h_distribution(tf.matmul(visible_probs, self.W) + self.b)

    def _prob_v_given_h(self, hidden_probs):
        """ p(v|h)
        """
        return self.presample_v_distribution(tf.matmul(hidden_probs, tf.transpose(self.W)) + self.a)

    def _contrastive_divergence(self):
        """ Defines the core operations that are used for training.

        Returns:
            grads: gradients obtained via contrastive divergence
        """
        input = self.vp
        # Positive phase begin
        positive_hidden_probs = self._prob_h_given_v(input)
        positive_hidden_activations = self.sample_h_distribution(positive_hidden_probs)
        positive_grads = tf.matmul(tf.transpose(input), positive_hidden_probs)

        # Negative phase begin
        hidden_activations = positive_hidden_activations

        # Contrastive Divergence iterations
        for step in range(self.args.cd_k):
            visible_probs = self._prob_v_given_h(hidden_activations)
            visible_activations = self.sample_v_distribution(visible_probs)
            hidden_probs = self._prob_h_given_v(visible_activations)
            hidden_activations = self.sample_h_distribution(hidden_probs)

        negative_visible_activations = visible_activations
        negative_hidden_activations = hidden_activations

        negative_grads = tf.matmul(tf.transpose(negative_visible_activations), negative_hidden_activations)

        # Calculate Gradients
        grad_w_new = -(positive_grads - negative_grads) / tf.to_float(tf.shape(input)[0])
        grad_visible_bias_new = -(tf.reduce_mean(input - negative_visible_activations, 0))
        grad_hidden_bias_new = -(tf.reduce_mean(positive_hidden_probs - negative_hidden_activations, 0))

        grads = [grad_w_new, grad_visible_bias_new, grad_hidden_bias_new]
        return grads

    def _training(self, loss=None, grads=None):
        """Sets up the training Ops.
        Applies the gradients to all trainable variables.
        If no loss is provided, gradients default to grads.

        Args:
            loss: loss to be minimized.
            grads: gradients to be minimized.
        Returns:
            train_op: the Op for training.
        """
        train_op = self.optimizer.apply_gradients(list(zip(grads, tf.trainable_variables())))
        return train_op

    def _reconstruct(self, visible_input):
        """ Reconstructs visible variables.

        Args:
            visible_input: visible_input to be reconstructed
        Returns:
            reconstruction
        """
        return self._prob_v_given_h(self.sample_h_distribution(self._prob_h_given_v(visible_input)))

    def _gibbs_sample_v_prime_given_v(self, visible_input, steps=500):
        """ Perform n-step Gibbs sampling chain in order to obtain the marginal distribution p(v|W,a,b) of the
        visible variables.

        Args:
            visible_input: visible_input to initialize gibbs chain
            steps: number of steps that Gibbs sampling chain is run for
        """
        v = visible_input
        for step in range(steps):
            v = self.sample_v_distribution(self._prob_v_given_h(self.sample_h_distribution(self._prob_h_given_v(v))))
        return v

    def _free_energy(self, v):
        """ FE(v) = −(aT)(v) − ∑_{i}log(1 + e^(b_{i} + W_{i}v))
        """
        return - tf.matmul(v, tf.expand_dims(self.a, -1)) \
            - tf.reduce_sum(tf.log(1 + tf.exp(self.b + tf.matmul(v, self.W))), axis=1)

    def _pseudo_log_likelihood(self):
        """ log(PL(v)) ≈ N * log(sigmoid(FE(v_{i}) − FE(v)))
        """
        v = self.sample_v_distribution(self.vp)
        vi = tf.concat(
            [v[:, :self._idx_pll + 1], 1 - v[:, self._idx_pll + 1:self._idx_pll + 2], v[:, self._idx_pll + 2:]], 1)
        self._idx_pll = (self._idx_pll + 1) % self.n_visible
        fe_x = self._free_energy(v)
        fe_xi = self._free_energy(vi)
        return tf.reduce_mean(tf.reduce_mean(
            self.n_visible * tf.log(tf.nn.sigmoid(tf.clip_by_value(fe_xi - fe_x, -20, 20))), axis=0))


    def update_model(self, visible_input):
        """ Updates model parameters via single step of optimizer train_op.

        Args:
            visible_input: visible_input 
        """
        self.sess.run(self.train_op, feed_dict={self.vp: visible_input})

    def eval_pll(self, visible_input):
        """ Evalulates pseudo_log_likelihood that model assigns to visible_input.

        Args:
            visible_input: visible_input 
        Returns:
            pseudo_log_likelihood
        """
        return self.sess.run(self.pll, feed_dict={self.vp: visible_input})

    def eval_rec_error(self, visible_input):
        """ Evalulates single step reconstruction error of reconstruction of visible_input.

        Args:
            visible_input: visible_input
        Returns:
            reconstruction error
        """
        return self.sess.run(self.reconstruction_error, feed_dict={self.vp: visible_input})   

    def sample_v_marg(self, n_samples=100, size=784, epoch=0):
        """ This function samples images via Gibbs sampling chain in order to inspect the marginal distribution of the
        visible variables.

        Args:
            num_samples: an integer value representing the number of samples that will be generated by the model.
            size: size of visible samples.
            epoch: how many training epochs have occured before taking this sample.
        """
        batch_v_noise = np.random.rand(n_samples, size)
        v_marg = self.sess.run(self.v_marg, feed_dict={self.vp: batch_v_noise})
        v_marg = v_marg.reshape([n_samples, 28, 28])
        save_merged_images(images=v_marg, size=(10, 10), path=os.path.join(*[
            self.args.log_dir, self.args.run_name, 'samples',
            'test_marg_ep' + str(epoch) + '_steps' + str(self.args.v_marg_steps) + '.png']))

    def save(self, file_path):
        """ Saves model.

        Args:
            file_path: path of file to save model in.
        """
        saver = tf.train.Saver()
        saver.save(self.sess, file_path)

    def load(self, file_path):
        """ Loads model.

        Args:
            file_path: path of file to load model from.
        """
        saver = tf.train.Saver()
        saver.restore(self.sess, file_path)
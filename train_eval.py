import numpy as np
import os


def evaluate(args, model, data):
    """ This function samples images via Gibbs sampling chain in order to inspect the marginal distribution of the
    visible variables.

    Args:
        args: parse_args input command-line arguments (hyperparameters).
        model: model to sample from.
        data: data to measure pseudo_log_likelihood (pll) that model assign to it (if pll is used).

    """
    for e in range(args.n_eval_samples):
        model.sample_v_marg(epoch=-(e + 1))


def train(args, model, data):
    """ This function trains model via Contrastive Divergence.

    Args:
        args: parse_args input command-line arguments (hyperparameters).
        model: model to train.
        data: data to train model with.
    """

    # calculate number of batches per epoch
    x_train = data.train.images
    if args.batch_size > 0:
        n_batches_train = x_train.shape[0] // args.batch_size + (0 if x_train.shape[0] % args.batch_size == 0 else 1)
    else:
        n_batches_train = 1

    x_test = data.test.images
    if args.batch_size > 0:
        n_batches_test = x_test.shape[0] // args.batch_size + (0 if x_test.shape[0] % args.batch_size == 0 else 1)
    else:
        n_batches_test = 1

    for e in range(args.epochs):
        print('Epoch:', e)

        # shuffle data
        perm = np.arange(x_train.shape[0])
        np.random.shuffle(perm)
        x_train = x_train[perm]

        # train
        for b in range(n_batches_train):
            x_batch = x_train[b * args.batch_size: (b + 1) * args.batch_size]
            model.update_model(x_batch)

        # eval and save
        if (e % args.n_train_eval_epochs == 0) and args.eval_during_train:
            test_rec_error = 0
            pll_test = 0
            for b in range(n_batches_test):
                x_batch = x_test[b * args.batch_size: (b + 1) * args.batch_size]
                if args.test_rec_error:
                    test_rec_error += model.eval_rec_error(x_batch)
                if args.validation_size > 0:
                    raise NotImplementedError
                pll_test += model.eval_pll(x_batch)

            if args.test_rec_error:
                print('PLL test: ', pll_test / n_batches_test, ' rec_error test: ', test_rec_error / n_batches_test)
            else:
                print('PLL test: {:.9f}'.format(pll_test / n_batches_test))

            model.sample_v_marg(epoch=e)
            if args.save:
                model.save(os.path.join(*[args.log_dir, args.run_name, 'ckpts', 'model_ep' + str(e), 'model_ep' + str(e)]))

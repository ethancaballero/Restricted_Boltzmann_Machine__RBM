from rbm import RBM
from train_eval import train, evaluate
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import numpy as np
import argparse
import os


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='RBM')

parser.add_argument('--data_dir', type=str, default="./data", help='dataset directory.')
parser.add_argument('--log_dir', type=str, default="./logdir",
                    help="directory to save all runs' weights, logs, & samples.")
parser.add_argument('--run_name', type=str, default="run_0", help='name of run to load/save.')
parser.add_argument('--batch_size', type=int, default=200, help='batch size.')
parser.add_argument('--epochs', type=int, default=10, help='number of training epochs.')
parser.add_argument('--n_vis', type=int, default=784, help='number of visible units.')
parser.add_argument('--dist_type_vis', type=str, default="bernoulli", help='visible distribution type.')
parser.add_argument('--n_hid', type=int, default=1000, help='number of hidden units.')
parser.add_argument('--dist_type_hid', type=str, default="bernoulli", help='hidden distribution type.')
parser.add_argument('--lr', type=float, default=0.01, help='learning_rate.')
parser.add_argument('--adam', type=str2bool, default=True, help='use Adam Optimizer.')
parser.add_argument('--cd_k', type=int, default=2, help='number of contrastive divergence iterations.')
parser.add_argument('--vb_mean', type=float, default=0.1, help='mean of visible bias init.')
parser.add_argument('--hb_mean', type=float, default=0.1, help='mean of hidden bias init.')
parser.add_argument('--validation_size', type=int, default=0, help='number of datapoints in validation set.')
parser.add_argument('--v_marg_steps', type=int, default=5000,
                    help='number of gibbs chain step for sampling from v marginal.')
parser.add_argument('--eval_during_train', type=str2bool, default=True, help='eval on validation during training.')
parser.add_argument('--n_train_eval_epochs', type=int, default=1, help='eval every n epochs of training.')
parser.add_argument('--test_rec_error', type=str2bool, default=False, help='evaluate reconstruction test error.')
parser.add_argument('--ckpt_epoch', type=int, default=0, help='which epoch of ckpt to load.')
parser.add_argument('--save', type=str2bool, default=True, help='save model.')
parser.add_argument('--load', type=str2bool, default=False, help='save model.')
parser.add_argument('--train', type=str2bool, default=True, help='train model.')
parser.add_argument('--eval', type=str2bool, default=False, help='evaluate model.')
parser.add_argument('--n_eval_samples', type=int, default=1,
                    help='number of (10x10 concatenations of) images sampled during eval.')

args = parser.parse_args()


def main():
    dataset = read_data_sets(args.data_dir, validation_size=args.validation_size)

    # init or load model
    rbm = RBM(args)
    if args.load:
        rbm.load(os.path.join(*[args.log_dir, args.run_name, 'ckpts', 
                'model_ep' + str(args.ckpt_epoch), 'model_ep' + str(args.ckpt_epoch)]))

    # create directories if not already made
    if not os.path.exists(os.path.join(args.log_dir, args.run_name)):
        os.mkdir(os.path.join(args.log_dir, args.run_name))
    else:
        if args.save:
            print("Warning: might overwrite previous samples & ckpts")
    if not os.path.exists(os.path.join(*[args.log_dir, args.run_name, 'ckpts'])):
        os.mkdir(os.path.join(*[args.log_dir, args.run_name, 'ckpts']))
    if not os.path.exists(os.path.join(*[args.log_dir, args.run_name, 'samples'])):
        os.mkdir(os.path.join(*[args.log_dir, args.run_name, 'samples']))

    # train and/or eval
    if args.train:
        train(args, model=rbm, data=dataset)
    if args.eval:
        evaluate(args, model=rbm, data=dataset)


if __name__ == '__main__':
    main()

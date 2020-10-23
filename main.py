import argparse
import torch

from resnet import ResNet18
from run_root_sgd import run_root
from utils import load_data, save_results


def main(args):
    trainloader, testloader, num_classes = load_data(batch_size=args.batch_size)
    net = ResNet18()
    results = run_root(
        net, args.batch_size, trainloader, testloader, n_epoch=args.n_epoch, 
        lr=args.learning_rate, weight_decay=0, checkpoint=125, noisy_train_stat=True)
    save_results(*results, method='root_sgd')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-bs' ,'--batch-size', default=128, type=int)
    parser.add_argument('-lr', '--learning-rate', default=0.01, type=float)
    parser.add_argument('--n-epoch', default=10, type=int)
    args = parser.parse_args()
    main(args)

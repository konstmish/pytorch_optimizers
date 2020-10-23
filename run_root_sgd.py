import copy
import numpy as np
import torch

from root_sgd import Root
from utils import accuracy_and_loss

def run_root(net, batch_size, trainloader, testloader, n_epoch=2, lr=0.1, weight_decay=0, checkpoint=125, noisy_train_stat=True):
    losses = []
    train_acc = []
    test_losses = []
    test_acc = []
    it_train = []
    it_test = []
    grad_norms = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    prev_net = copy.deepcopy(net)
    prev_net.to(device)
    net.train()
    prev_net.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Root(net.parameters(), lr=lr, weight_decay=weight_decay)
    prev_optimizer = Root(prev_net.parameters(), weight_decay=weight_decay)
    first_it = True
    N_train = 50000
    
    for epoch in range(n_epoch):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            prev_optimizer.zero_grad()

            if first_it:
                first_it = False
            else:
                prev_outputs = prev_net(inputs)
                prev_loss = criterion(prev_outputs, labels)
                prev_loss.backward()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            optimizer.update_buf(prev_optimizer)
            prev_net.load_state_dict(net.state_dict())
            optimizer.step()

            running_loss += loss.item()
            if (i % 10) == 0:
                if noisy_train_stat:
                    losses.append(loss.cpu().item())
                    it_train.append(epoch + i * batch_size / N_train)

            if i % checkpoint == checkpoint - 1:
                if running_loss / checkpoint < 0.01:
                    print('[%d, %5d] loss: %.4f' %
                          (epoch + 1, i + 1, running_loss / checkpoint))
                else:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / checkpoint))
                running_loss = 0.0
                test_a, test_l = accuracy_and_loss(net, testloader, device, criterion)
                test_acc.append(test_a)
                test_losses.append(test_l)
                grad_norms.append(sum([p.grad.data.norm().item() for p in net.parameters()]))
                net.train()
                it_test.append(epoch + i * batch_size / N_train)

        if not noisy_train_stat:
            it_train.append(epoch)
            train_a, train_l = accuracy_and_loss(net, trainloader, device, criterion)
            train_acc.append(train_a)
            losses.append(train_l)
            net.train()

    print('Finished Training')
    del prev_net
    return (np.array(losses), np.array(test_losses), np.array(train_acc), np.array(test_acc),
            np.array(it_train), np.array(it_test), np.array(grad_norms))

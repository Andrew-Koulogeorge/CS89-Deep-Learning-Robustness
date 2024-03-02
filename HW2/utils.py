import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np



def make_dataloader(data_path, batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    val_dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def eval_test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += F.cross_entropy(outputs, targets).item()
            pred = outputs.max(1, keepdim=True)[1]
            correct += pred.eq(targets.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def eval_robust(model, test_loader, pgd_attack):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # ajk
    model.eval()
    robust_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            adv = pgd_attack(inputs, targets)
            outputs = model(adv)
            robust_loss += F.cross_entropy(outputs, targets).item()
            pred = outputs.max(1, keepdim=True)[1]
            correct += pred.eq(targets.view_as(pred)).sum().item()
    robust_loss /= len(test_loader.dataset)

    print('LinfPGD Attack: Average loss: {:.4f}, Robust Accuracy: {}/{} ({:.0f}%)'.format(
        robust_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    robust_accuracy = 100. * correct / len(test_loader.dataset)
    return robust_loss, robust_accuracy


def mixup_data(x, y, mixup_alpha=1.0):
    '''
    Source https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py
    '''
    lam = np.random.beta(mixup_alpha, mixup_alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    '''
    Source https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py
    '''
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=8/255,
                perturb_steps=10,
                beta=1.0):
    '''
    Source https://github.com/yaodongyu/TRADES/blob/master/trades.py
    '''
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                   F.softmax(model(x_natural), dim=1))
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    
    # zero gradient
    optimizer.zero_grad()
    
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import random
import numpy as np
from model import ResNet18
from utils import trades_loss, mixup_data, mixup_criterion, make_dataloader, eval_test
import argparse

class LinfPGDAttack(nn.Module):
    def __init__(self, model, epsilon, steps=10, step_size=0.003):
        super().__init__()
        self.model = model
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = step_size

    def perturb(self, x_natural:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        """
        Computes the gradient of the cross-entropy loss with respect to the input
        image `x_adv` and updates the image based on the gradient direction. The 
        perturbation is clipped to ensure it stays within a specified epsilon range
        and is finally clamped to ensure pixel values are valid. 
        
        The resulting perturbed image is returned.
        """
        # *********** Your code starts here *********** 
        
        cross_entropy = torch.nn.CrossEntropyLoss() # wrapper class whose forward pass uses F.cross_entropy
        x_adv = x_natural.detach().clone() # need to keep a clean version of x_natural for later comparison
        x_adv.requires_grad_() # ensure that the backwards pass will keep track of the gradient wrt the input
        
        for _ in range(self.steps):
            
            with torch.enable_grad(): # ensure gradients are being computed 
                loss = cross_entropy(input=self.model(x_adv), target=y) # compute loss between models prediction on x_adv and label y
                loss.backward() # backward pass to compute the gradient of the loss 

            x_adv.data += self.step_size * torch.sign(x_adv.grad) # update pixels of x_adv in direction of greatest asscent on the loss landscape        
            eta = torch.clamp(x_adv.data - x_natural.data, min=-self.epsilon, max=self.epsilon) # ensure diff between x_natural and x_adv within the epsilon box (since Linf norm) for all dims
            x_adv.data = torch.clamp(x_natural.data + eta, min=0, max=1) # ensure that the image is well defined after the perterbation
        
        # *********** Your code ends here *************
        return x_adv.detach()

    def forward(self, x_natural, y):

        x_adv = self.perturb(x_natural, y)
        return x_adv
        # *********** Your code ends here *************
    

def train_ep(model, train_loader, mode, pgd_attack, optimizer, criterion, epoch, batch_size, device):
    model.train()
    model.to(device)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        if mode == 'normal':
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        elif mode == 'adv_train': # [Ref] https://arxiv.org/abs/1706.06083
            model.eval() # Q --> dont we need to compute the gradients for the attack? 
            adv_x = pgd_attack(inputs, targets)
            model.train()

            optimizer.zero_grad()
            outputs = model(adv_x)
            loss = criterion(outputs, targets)

        elif mode == 'adv_train_trades': # [Ref] https://arxiv.org/abs/1901.08573
            optimizer.zero_grad()
            loss = trades_loss(model=model, x_natural=inputs, y=targets, optimizer=optimizer)
            
        elif mode == 'adv_train_mixup': # [Ref] https://arxiv.org/abs/1710.09412
            model.eval()
            benign_inputs, benign_targets_a, benign_targets_b, benign_lam = mixup_data(inputs, targets)            
            adv_x = pgd_attack(inputs, targets)
            adv_inputs, adv_targets_a, adv_targets_b, adv_lam = mixup_data(adv_x, targets)
            
            model.train()
            optimizer.zero_grad()
            
            benign_outputs = model(benign_inputs)
            adv_outputs = model(adv_inputs)
            loss_1 = mixup_criterion(criterion, benign_outputs, benign_targets_a, benign_targets_b, benign_lam)
            loss_2 = mixup_criterion(criterion, adv_outputs, adv_targets_a, adv_targets_b, adv_lam)
            
            loss = (loss_1 + loss_2) / 2

        else:
            print("No training mode specified.")
            raise ValueError()

        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{:05d}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(inputs), len(train_loader) * batch_size,
                       100. * (batch_idx + 1) / len(train_loader), loss.item()))

def train(model, train_loader, val_loader, pgd_attack,
          mode='natural', epochs=25, batch_size=256, learning_rate=0.1, momentum=0.9, weight_decay=2e-4,
          checkpoint_path='model1.pt', device='cpu'):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    best_acc = 0
    for epoch in range(epochs):
        # training
        train_ep(model, train_loader, mode, pgd_attack, optimizer, criterion, epoch, batch_size, device)

        # evaluate clean accuracy
        test_loss, test_acc = eval_test(model, val_loader, device)

        # remember best acc@1 and save checkpoint
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        # save checkpoint if is a new best
        if is_best:
            torch.save(model.state_dict(), checkpoint_path)
        print('================================================================')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Get hyper parameters')

        # Positional argument
    parser.add_argument('--device', type=str, help='Which GPU to be on')
    parser.add_argument('--lr', default=0.1, type=float)

    # Optional argument
    parser.add_argument('--training_mode', type=str, help='Which training mode to be on')
    # "normal"
    # "adv_train"
    # "adv_train_trades"
    # "adv_train_mixup"

    # Optional argument with a value
    parser.add_argument('--checkpoint_path', type=str, help='Output file')
    # "model1.normal.pt*"
    # "model2.adv_train.pt*"
    # "model3.adv_train_trades.pt*"
    # "model4.adv_train_mixup.pt*"


    args = parser.parse_args()

    device = args.device
    training_mode = args.training_mode
    checkpoint_path = args.checkpoint_path
    lr = args.lr

    # define parameters
    batch_size = 256
    data_path = "../data" # directory of the data
    epsilon = 8/255
    steps = 10
    epochs = 25

    # create data loader
    train_loader, val_loader = make_dataloader(data_path, batch_size)
    model = ResNet18(num_classes=10)
    pgd_attack = LinfPGDAttack(model=model, epsilon=epsilon)

    train(model=model, train_loader=train_loader, val_loader=val_loader, pgd_attack=pgd_attack, mode=training_mode, learning_rate=lr, epochs=epochs, batch_size=256, checkpoint_path=checkpoint_path, device=device)
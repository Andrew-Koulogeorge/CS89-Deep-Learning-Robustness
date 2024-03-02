import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import random
from main import make_model, load_cifar10_data, validate, calculate_bound
import argparse


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Training a fully connected NN with one hidden layer')
    parser.add_argument('--dataset', default='./dataset', help='dataset folder')
    parser.add_argument('--nunits', default=1024, type=int,
                        help='number of hidden units (default: 1024)')
    parser.add_argument('--checkpoint', default='model1.pt', type=str, help='checkpoint file')

    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    init_model = make_model(3, args.nunits,10)
    model = make_model(3, args.nunits,10, checkpoint=args.checkpoint)
    model.eval()

    # Load the data
    train_dataset = load_cifar10_data('train', args.dataset)
    val_dataset = load_cifar10_data('val', args.dataset)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=1, pin_memory=True)

    criterion = nn.CrossEntropyLoss().to(device)

    train_acc, train_loss, train_margin = validate(model, device, train_loader, criterion)
    val_acc, val_loss, val_margin = validate(model, device, val_loader, criterion)

    print(f'=================== Summary ===================\n',
          f'Training loss: {train_loss:.3f}   Training margin {train_margin:.3f}  Validation margin {val_margin:.3f} ',
          f'Training accuracy: {train_acc:.3f}   Validation accuracy: {val_acc:.3f}\n')

    # Print the measures and bounds
    measure = calculate_bound(model, init_model, device, train_loader, train_margin)
    for key, value in measure.items():
        print(f'{key:s}:\t {float(value):3.3}')

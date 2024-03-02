import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import random
import copy
import math


# train the model for one epoch on the given dataset
def train(model, device, train_loader, criterion, optimizer):
    sum_loss, sum_correct = 0, 0

    # switch to train mode
    model.train()
    for data, target in train_loader:
        data, target = data.to(device).view(data.size(0), -1), target.to(device)

        # compute the output
        output = model(data)

        # compute the classification error and loss
        loss = criterion(output, target)
        pred = output.max(1)[1]
        sum_correct += pred.eq(target).sum().item()
        sum_loss += len(data) * loss.item()

        # compute the gradient and do an SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_accuracy = sum_correct / len(train_loader.dataset)
    train_loss = sum_loss / len(train_loader.dataset)
    return train_accuracy, train_loss


# evaluate the model on the given set
def validate(model, device, val_loader, criterion):
    sum_loss, sum_correct = 0, 0
    margin = torch.tensor([]).to(device)  # initialize the margin

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device).view(data.size(0), -1), target.to(device)

            # forward pass for logits
            output = model(data)
            # softmax = nn.Softmax(dim=1)
            # output = softmax(output)

            # compute the classification error and loss (model output is the argmax over the probability distribution over classes)
            pred = output.max(1)[1]
            sum_correct += pred.eq(target).sum().item()
            sum_loss += len(data) * criterion(output, target).item()

            # compute the margin
            # *********** You code starts here ***********

            # output is the B x 10 matrix of logits for each element in the databatch

            # first, we need to compute the logit for the true class of the example; compute index for each row of the matrix
            row_indexs = torch.arange(0,len(output))

            # get the scores for the correct labels for each of the examples 
            fx_y = output[row_indexs,target]
            
            # now, compute all B of the probability distributions over the next classes without the correct class label (taken from target)
            fxnot_y = torch.stack(tuple((torch.cat((row[:idx], row[idx+1:])) for row, idx in zip(output, target))))
            
            # compute the largest output value not including the correct class
            fxnot_y, _ = torch.max(fxnot_y, dim=1)

            # take the difference 
            margin_values = fx_y - fxnot_y

            # doing this for all batches; append to current list
            margin = torch.cat((margin, margin_values))

    # calculate the 5th percentile margin

    percentile_margin = np.percentile(margin, 5)
 

    val_accuracy = sum_correct / len(val_loader.dataset)
    val_loss = sum_loss / len(val_loader.dataset)
    return val_accuracy, val_loss, percentile_margin


# load and preprocess CIFAR-10 data.
def load_cifar10_data(split, datadir):
    # Data Normalization and Augmentation (random cropping and horizontal flipping)
    # The mean and standard deviation of the CIFAR-10 dataset: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    tensor_transform = transforms.ToTensor()
    normalize_transform = transforms.Normalize(mean=MEAN, std=STD)
    randomcrop_transform = transforms.RandomResizedCrop(size=(32,32), antialias=True)
    randomflip_transform = transforms.RandomHorizontalFlip(p=.5)

    train_transform = transforms.Compose([randomcrop_transform,
                                          randomflip_transform, 
                                          tensor_transform,
                                          normalize_transform])
    
    val_transform = transforms.Compose([tensor_transform, normalize_transform])

    if split == 'train':
        dataset = datasets.CIFAR10(root=datadir, train=True, download=True, transform=train_transform)
    else:
        dataset = datasets.CIFAR10(root=datadir, train=False, download=True, transform=val_transform)

    return dataset

# define a fully connected neural network with a single hidden layer
def make_model(nchannels, nunits, nclasses, checkpoint=None):
    # define the model

    model = nn.Sequential(
        nn.Linear(in_features=32*32*nchannels, out_features=nunits),
        nn.ReLU(),
        nn.Linear(in_features=nunits, out_features=nclasses),
        nn.Softmax(dim=1) # output activation fucntion 
    )

    # *********** You code ends here ***********

    # load the model from the checkpoint if provided
    # *********** You code starts here ***********

    if checkpoint: model.load_state_dict(torch.load(checkpoint))

    # *********** You code ends here ***********
    return model


def calculate_bound(model, init_model, device, data_loader, margin):
    # switch to evaluate mode
    model.eval()
    init_model.eval()

    # code is getting each of the layers for the trained and the init model
    modules = list(model.children())
    init_modules = list(init_model.children())

    D = modules[0].weight.size(1) # data dimension
    H = modules[0].weight.size(0) # number of hidden units
    C = modules[2].weight.size(0) # number of classes (output dimension)
    num_param = sum(p.numel() for p in model.parameters()) # number of parameters of the model

    with torch.no_grad():

        # Eigenvalues of the weight matrix in the first and second layer
        _,S1,_ = modules[0].weight.svd()
        _,S2,_ = modules[2].weight.svd()

        # Eigenvalues of the initial weight matrix in the first and second layer
        _,S01,_ = init_modules[0].weight.svd()
        _,S02,_ = init_modules[2].weight.svd()

        # Frobenius norm of the weight matrix in the first and second layer
        Fr1 = S1.square().sum().sqrt()
        Fr2 = S2.square().sum().sqrt()

        # Difference of final weights to the initial weights in the first and second layer
        diff1 = modules[0].weight - init_modules[0].weight
        diff2 = modules[2].weight - init_modules[2].weight

        # Euclidean distance of the weight matrix in the first and second layer to the initial weight matrix
        Dist1 = torch.linalg.matrix_norm(diff1, ord=2)
        Dist2 = torch.linalg.matrix_norm(diff2, ord=2)


        # L_{1,infty} norm of the weight matrix in the first and second layer
        L1max1 = modules[0].weight.norm(p=1, dim=1).max()
        L1max2 = modules[2].weight.norm(p=1, dim=1).max()
        
        # L_{2,1} distance of the weight matrix in the first and second layer to the initial weight matrix
        L1Dist1 = diff1.norm(p=2, dim=1 ).sum()
        L1Dist2 = diff2.norm(p=2, dim=1 ).sum()

        measure = {}
        measure['Frobenius1'] = Fr1
        measure['Frobenius2'] = Fr2
        measure['Distance1'] = Dist1
        measure['Distance2'] = Dist2
        measure['Spectral1'] = S1[0]
        measure['Spectral2'] = S2[0]
        measure['Fro_Fro'] = Fr1 * Fr2
        measure['L1max_L1max'] = L1max1 * L1max2
        measure['Spec_Dist'] = S1[0] * Dist2 * math.sqrt(C)
        measure['Dist_Spec'] = S2[0] * Dist1 * math.sqrt(H)
        measure['Spec_Dist_sum'] = measure['Spec_Dist'] + measure['Dist_Spec']
        measure['Spec_L1max'] = S1[0] * L1Dist2
        measure['L1max_Spec'] = S2[0] * L1Dist1
        measure['Spec_L1max_sum'] = measure['Spec_L1max'] + measure['L1max_Spec']

        # Compute the Frobenius Distance
        # *********** You code starts here ***********
        measure['Dist_Fro'] = Dist1 * Fr2 

        # *********** You code ends here ***********

        # delta is the probability that the generalization bound does not hold
        delta = 0.01

        # m is the number of training samples
        m = len(data_loader.dataset)
        layer_norm, data_L2, data_Linf, domain_L2 = 0, 0, 0, 0
        for data, target in data_loader:
            data = data.to(device).view(target.size(0),-1)
            layer_out = torch.zeros(target.size(0), H).to(device)

            # calculate the norm of the output of the first layer in the initial model
            def fun(m, i, o): layer_out.copy_(o.data)
            h = init_modules[1].register_forward_hook(fun)
            output = init_model(data)
            layer_norm += layer_out.norm(p=2, dim=0) ** 2
            h.remove()

            # L2 norm squared of the data
            data_L2 += data.norm() ** 2
            # maximum L2 norm squared on the training set. We use this as an approximation of this quantity over the domain
            domain_L2 = max(domain_L2, data.norm(p=2, dim = 1).max() ** 2)
            # L_infty norm squared of the data
            data_Linf += data.max(dim = 1)[0].max() ** 2

        # computing the average
        data_L2 /= m
        data_Linf /= m
        layer_norm /= m

        # number of parameters
        measure['#parameter'] = num_param

        # Generalization bound based on the VC dimension by Harvey et al. 2017
        VC = (2 + num_param * math.log(8 * math.e * ( H + 2 * C ) * math.log( 4 * math.e * ( H + 2 * C ) ,2), 2)
                * (2 * (D + 1) * H + (H + 1) * C) / ((D + 1) * H + (H + 1) * C))
        measure['VC bound'] = 8 * (C * VC * math.log(math.e * max(m / VC, 1))) + 8 * math.log(2 / delta)

        # Generalization bound by Bartlett and Mendelson 2002
        R = 8 * C * L1max1 * L1max2 * 2 * math.sqrt(math.log(D)) * math.sqrt(data_Linf) / margin
        measure['L1max bound'] = (R + 3 * math.sqrt(math.log(m / delta))) ** 2

        # Compute the Generalization bound as provided in the instruction
        measure['Your bound'] = (8 * math.sqrt(C) * Fr1 * Fr2 * math.sqrt(data_L2)/margin + 3*math.sqrt(m/delta)) ** 2


    return measure


def main():
    # define the parameters to train your model
    datadir = 'datasets'  # the directory of the dataset
    nchannels = 3
    nclasses = 10

    nunits = 1024
    # nunits = 256
    
    lr = .001
    mt = .9
    batchsize = 64
    epochs = 25
    stopcond = .01


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if device else {}

    # create an initial model
    model = make_model(nchannels, nunits, nclasses)
    model = model.to(device)

    # make a copy of the initial model for later use
    init_model = copy.deepcopy(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=mt)

    # loading data
    train_dataset = load_cifar10_data('train', datadir)
    val_dataset = load_cifar10_data('val', datadir)

    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, **kwargs)

    # training the model

    # these lists used to plot how the validation loss and the validation margin are changing as the training of the network proceeds 
    val_loss_tracker = []
    val_margin_tracker = []

    for epoch in range(0, epochs):

        # Training
        train_acc, train_loss =  train(model=model, 
                                       device=device,
                                       train_loader=train_loader,
                                       criterion=criterion,
                                       optimizer=optimizer)

        # Validation
        val_acc, val_loss, val_margin = validate(model=model,
                                                 device=device,
                                                 val_loader=val_loader,
                                                 criterion=criterion) 
        
         
        val_loss_tracker.append(val_loss)
        val_margin_tracker.append(val_margin)


        print(f'Epoch: {epoch + 1}/{epochs}\t Training loss: {train_loss:.3f}   Training accuracy: {train_acc:.3f}   ',
              f'Validation margin {val_margin:.3f}   Validation accuracy: {val_acc:.3f}')

        # stop training if the cross-entropy loss is less than the stopping condition
        if train_loss < stopcond:
            break

    # save the trained model
    
    torch.save(model.state_dict(), "model1.pt")
    # torch.save(model.state_dict(), "model2.pt")


    # calculate the training error and margin (on Training set) of the learned model
        
    train_acc, train_loss, train_margin = validate(model=model,
                                                 device=device,
                                                 val_loader=train_loader,
                                                 criterion=criterion) 
    
    val_acc, val_loss, val_margin = validate(model=model,
                                                 device=device,
                                                 val_loader=val_loader,
                                                 criterion=criterion)

    print(f'=================== Summary ===================\n',
          f'Training loss: {train_loss:.3f}   Training margin {train_margin:.3f}  Validation margin {val_margin:.3f}     ',
          f'Training accuracy: {train_acc:.3f}   Validation accuracy: {val_acc:.3f}\n')

    # Print the measures and bounds
    measure = calculate_bound(model, init_model, device, train_loader, train_margin)

    for key, value in measure.items():
        print(f'{key:s}:\t {float(value):3.3}')
    
    torch.save(val_loss_tracker, "Val loss per epoch 1")
    torch.save(val_margin_tracker, "Val margin per epoch 1")

    show_graph(val_loss_tracker, "Val loss per Epoch ")
    
    # plot the validation loss and validation margin as a function of epochs
    show_graph(val_margin_tracker, label="Margin per Epoch")
   
   

def show_graph(data,label):
    plt.plot(data, label=label)
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

        
if __name__ == '__main__':
    
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    main()

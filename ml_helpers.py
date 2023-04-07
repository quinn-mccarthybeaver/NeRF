from tqdm import tqdm
from rendering import rendering
import torch

def training(model, optimizer, scheduler, tn, tf, nb_bins, nb_epochs, data_loader, device='cpu'):
    
    training_loss = []
    for epoch in (range(nb_epochs)):
        for batch in tqdm(data_loader):
            o = batch[:, :3].to(device)  #first three ray origin
            d = batch[:, 3:6].to(device)   # ray direction
            
            target = batch[:, 6:].to(device)   #target values
            
            prediction = rendering(model, o, d, tn, tf, nb_bins=nb_bins, device=device)   #forward pass
            
            loss = ((prediction - target)**2).mean()  # compute the loss  the model tries to minimize the value of the training loss by adjusting its parameters
            
            optimizer.zero_grad()   # it helps to clear the gradients of the optimized parameter
                #it ensures the optimizer uses only the gradients computed from the current batch and not previous
            loss.backward()   #when called the autograd computes the gradients of the loss wrt all parameters that have requires_grad=True. These gradients are stored in the .grad attribut of each parameter
            optimizer.step()  #update the parameters of the model based on the computed gradients
            training_loss.append(loss.item())   #  In this case, the new element is the value of the training loss at the current iteration, which is obtained by calling the "item" method on the loss tensor. The "item" method returns a Python scalar that represents the value of the tensor. By appending the training loss to the list at each iteration, we can create a record of the training loss over the course of the training process.
            
        scheduler.step()  # update the learning rate of the optimizer during training. In PyTorch, the learning rate is typically set when the optimizer is created, but it can be changed during training to improve performance.
            #There are several types of learning rate schedulers in PyTorch, such as StepLR, ReduceLROnPlateau, and CosineAnnealingLR, each with a different way of adjusting the learning rate.
            # the StepLR scheduler reduces the learning rate by a factor of gamma every step_size epochs.  By calling the "step" method at the end of each training epoch, we can update the learning rate for the next epoch based on the performance of the model so far. This can help the model converge faster and avoid getting stuck in local minima
        
        torch.save(model.cpu(), 'model_nerf')
        model.to(device)
        
    return training_loss
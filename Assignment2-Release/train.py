from tqdm import tqdm

def train_one_epoch(epoch_index, training_loader, optimizer, model, loss_fn):
    running_loss = 0.
    last_loss = 0.
    len_samples = 0
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(tqdm((training_loader))):
        # Every data instance is an input + label pair
        inputs, labels, _ = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        inputs = inputs.reshape(inputs.shape[0], -1)
        len_samples += inputs.shape[0]
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        
        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()
        
        # Adjust learning weights
        optimizer.step()
        
        # Gather data and report
        running_loss += loss.item()
 
    last_loss = running_loss / len_samples
    print('  batch {} loss: {}'.format(i + 1, last_loss))

    return last_loss
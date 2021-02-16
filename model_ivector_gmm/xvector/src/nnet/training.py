import torch
import logging
import os
import torch.nn.functional as F


# Get the same logger from main"
logger = logging.getLogger("MLP")


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (_, vec, target) in enumerate(train_loader):
        vec, target = vec.to(device), target.to(device)
        target = target.view(-1, 1).float()
        optimizer.zero_grad()
        # output = model(vec)[:, :, 0]
        output = model(vec).reshape(-1, 1)
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(vec), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def snapshot(dir_path, run_name, is_best, state):
    snapshot_file = os.path.join(dir_path, run_name + '-model_best.pth')
    if is_best:
        torch.save(state, snapshot_file)
        logger.info("Snapshot saved to {}\n".format(snapshot_file))
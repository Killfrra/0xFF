import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from model import Net
import torch.nn.functional as F

writer = SummaryWriter(
    #log_dir='runs/May28_23-11-41_k'
)

metrics = {
    'avg_train_loss': 0,
    'avg_val_loss': 0,
    'accuracy': 0
}

def train(args, model, device, train_loader, optimizer, epoch):
    global metrics
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)

        train_loss += loss.item()

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            writer.add_scalar('train_loss', loss.item())
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    train_loss /= len(train_loader)
    metrics['avg_train_loss'] = train_loss
        


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    metrics['avg_val_loss'] = test_loss
    metrics['accuracy'] = accuracy    

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * accuracy))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N', help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR', help='learning rate (default: 1.0)')
    #parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='learning rate (default: 0.9)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M', help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    #torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(46),
        transforms.ToTensor()
    ])

    train_loader = DataLoader(
        datasets.ImageFolder('ram/mini_ru_train_preprocessed', transform),
        batch_size=args.batch_size, shuffle=True, **kwargs
    )
    test_loader  =  DataLoader(
        datasets.ImageFolder('ram/mini_ru_test_preprocessed', transform),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = Net(False).to(device)
    optimizer = optim.Adadelta(model.parameters(), args.lr) #, args.momentum)
    """
    checkpoint = torch.load('saves/mnist_cnn_epoch_14.pt')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    """
    epoch = 0
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma, last_epoch=(epoch - 1))
    """
    scheduler.load_state_dict(checkpoint['scheduler'])
    """
    while epoch < args.epochs:

        epoch += 1

        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

        if args.save_model:
            to_save = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            torch.save(to_save, 'saves/mnist_cnn_epoch_%d.pt' % epoch)
        
        scheduler.step()

        writer.add_scalars('main', metrics, epoch)

    writer.close()


if __name__ == '__main__':
    main()

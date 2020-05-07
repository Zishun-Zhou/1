from __future__ import print_function

import torch
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from AlexNet import AlexNet


def train_test_alexnet(model, device, train_loader, test_loader, optimizer, epoch):
    #10 classes of pic
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    with open("train_acc.txt", "w") as train_f, open("test_acc.txt", "w") as test_f:
        for epoch in range(1, epoch + 1):
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                output = model(data)
                criterion = torch.nn.CrossEntropyLoss()
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                # print loss rate and accuracy
                train_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                # iter_train = batch_idx
                acc_train = 100. * correct / total
                print('Training: [epoch:%d, iter:%d] Loss: %.03f | Accurracy: %.3f%% '
                      % (epoch, (batch_idx + 1 + (epoch - 1) * len(train_loader)), train_loss / (batch_idx + 1),
                         100. * correct / total))

                # iter, acc, loss
                train_f.write('%05d %.3f %.3f' % (
                (batch_idx + 1 + (epoch - 1) * len(train_loader)), acc_train, train_loss / (batch_idx + 1)))
                train_f.write('\n')
                train_f.flush()

            print("Waiting Test!")

            with torch.no_grad():
                correct = 0
                total = 0
                for batch_idx, (data, target) in enumerate(test_loader):
                    model.eval()
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    # get the class which has the highest accuracy
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum()
                    # confusion matrix
                    pred = torch.max(output, 1)[1]
                    conf_matrix = confusion_matrix(target, pred)
                    acc_test = 100. * correct / total

                    test_f.write('%05d %.3f' % (epoch, acc_test))
                    test_f.write('\n')
                    test_f.flush()

                 #plot the confusion matrix
                if epoch == epoch:

                    ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                                           display_labels=classes).plot()

                    plt.show()
                print('Testing Accuracy：%.3f%%' % (100 * correct / total))


def main():

    epoches = 30
    torch.manual_seed(1)
    save_model = True

    # Check whether you can use Cuda
    use_cuda = torch.cuda.is_available()
    # Use Cuda if you can
    device = torch.device("cuda" if use_cuda else "cpu")

    ######################3   Torchvision    ###########################3
    # Use data predefined loader
    # Pre-processing by using the transform.Compose
    # divide into batches
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                             # the most fit normalization numbers
                         ])),
        batch_size=128, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])),
        batch_size=100, shuffle=True, **kwargs)

    # get some random training images
    #dataiter = iter(train_loader)
    #images, labels = dataiter.next()

    # #####################    Build your network and run   ############################

    model = AlexNet().to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.65)

    train_test_alexnet(model, device, train_loader, test_loader, optimizer, epoches)

    if save_model:
        torch.save(model.state_dict(), "./cifar_net.pth")


if __name__ == '__main__':
    main()

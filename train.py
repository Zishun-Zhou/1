import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import itertools
from resnet import ResNet18
import matplotlib.pyplot as plt


train_accuracy = []
test_accuracy = []

def main():
# check if GPU is available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parameters setting
    EPOCH = 50
    BATCH_SIZE = 128

# Data Transform
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  #32*32
        transforms.RandomHorizontalFlip(), #to flip some input images to prevent overfitting
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B Normalized value: mean and variance
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
# Cifar-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# model definition-ResNet
    net = ResNet18().to(device)

# loss function and optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
# Data loading
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


    training_testing(net, device, trainloader, testloader,optimizer, EPOCH, classes)


    Train_acc_and_loss_plot()

# training and testing funcion
def training_testing(net,device,trainloader,testloader,optimizer,EPOCH,classes):
                ########     Training section     ########
    print("Start Training")
    with open("Accuracy_Loss.txt", "w") as f:
        for epoch in range(0, EPOCH):
            print('\nEpoch: %d' % (epoch + 1))
            net.train()
            sum_loss = 0.0
            correct = 0.0
            total = 0.0
            for i, data in enumerate(trainloader, 0):
                # data
                length = len(trainloader)
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()

                # forward + backward
                outputs = net(inputs)
                criterion = nn.CrossEntropyLoss()
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                # print loss rate and accuracy
                sum_loss += loss.item()
                _, prediction = torch.max(outputs.data, 1)
                total += targets.size(0)

                # correct += prediction.eq(targets.data).cpu().sum()
                correct += prediction.eq(targets.data).cpu().sum()

                print('Training: [epoch:%d, iter:%d] Loss: %.03f | Accurracy: %.3f%% '
                      % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                # write the results into a Accuracy_Loss.txt file
                # Accuracy_Loss.txt file will be used to plot the learning curve
                f.write('%05d %.3f %.03f' % ((i + 1 + epoch * length), 100. * correct / total, sum_loss / (i + 1)))
                f.write('\n')
                f.flush()


                ########     Testing section     ########
            print("Waiting Test!")
            with torch.no_grad():
                answer = 0
                total = 0
                conf_matrix = torch.zeros([10, 10],device='cuda')
                for data in testloader:
                    net.eval()
                    images, targets = data
                    images, targets = images.to(device), targets.to(device)
                    outputs = net(images)
                    # get the class which has the highest accuracy
                    _, prediction = torch.max(outputs.data, 1)


                    total += targets.size(0)
                    answer += (prediction == targets).sum()

                    pred = torch.max(outputs, 1)[1]
                    conf_matrix = confusion_matrix(pred, targets, conf_matrix=conf_matrix)

                print('Testing Accuracy：%.3f%%' % (100 * answer / total))
                print('Saving model......')
                torch.save(net.state_dict(),"./result/cifar_ResNet.pth")

                # drawing confusion matrix
                if(epoch +1  ==EPOCH):
                    plot_confusion_matrix(conf_matrix.cpu().numpy(), classes=classes, normalize=True,
                                        title='Normalized confusion matrix')



        print("Training Finished, Total EPOCH=%d" % EPOCH)
#function to plot the learning curve
def Train_acc_and_loss_plot():

    filename = 'Accuracy_Loss.txt'
    X, Y, Z = [], [], []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in lines:
            value = [float(s) for s in i.split()]
            X.append(value[0])
            Y.append(value[1])
            Z.append(value[2])

    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.set_title("ResNet18 Training Accuraccy and Loss")
    ax.plot(X, Y, label='Accuracy', color="tab:red")

    ax2 = ax.twinx()
    ax2.plot(X, Z, label='Loss', color="tab:blue")
    ax.legend(loc=0)
    ax2.legend(loc=0)
    ax.grid()
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Accuracy(%)")
    ax2.set_ylabel("Loss(%)")

    plt.show()

# function to update confusion matrix
def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix

# function to update confusion matrix
def plot_confusion_matrix(cm, classes, normalize, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.

    sum_col = cm.sum(1)


    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            percentage = cm[i][j]/sum_col[i]
            plt.text(j, i, '{:.2f}'.format(percentage), horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, '{:.0f}'.format(cm[i, j]), horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')




if __name__ == '__main__':
    main()


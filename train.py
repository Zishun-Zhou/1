import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from resnet import ResNet18
import matplotlib.pyplot as plt


train_accuracy = []
test_accuracy = []

def main():
# check if GPU is available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# parameters setting
    EPOCH = 1   #epoch
    pre_epoch = 0
    BATCH_SIZE = 128



# Data preparation
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

    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)


    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


    training_testing(net, device, trainloader, testloader,optimizer, EPOCH)




    Train_acc_and_loss_plot()

# training
def training_testing(net,device,trainloader,testloader,optimizer,EPOCH):
    print("Start Training")
    with open("acc.txt", "w") as f:
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
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

                print('Training: [epoch:%d, iter:%d] Loss: %.03f | Accurracy: %.3f%% '
                      % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                f.write('%05d %.3f %.03f' % ((i + 1 + epoch * length), 100. * correct / total, sum_loss / (i + 1)))
                f.write('\n')
                f.flush()




            ########     Testing section     ########
            print("Waiting Test!")
            with torch.no_grad():
                answer = 0
                total = 0
                for data in testloader:
                    net.eval()
                    images, targets = data
                    images, targets = images.to(device), targets.to(device)
                    outputs = net(images)
                    # get the class which has the highest accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    answer += (predicted == targets).sum()
                print('Testing Accuracy：%.3f%%' % (100 * answer / total))

                # write the results into a acc.txt file
                print('Saving model......')



        print("Training Finished, Total EPOCH=%d" % EPOCH)

def Train_acc_and_loss_plot():

    filename = 'acc.txt'
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



if __name__ == '__main__':
    main()

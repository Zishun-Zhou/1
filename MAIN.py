import torch
import torch.optim as optim
import torchvision
from MODEL.resnet import ResNet18
from MODEL.AlexNet import AlexNet
from MODEL.LeNet5 import LeNet_5
from MODEL.VGG11 import VGG11
from functions import training, testing_savedModel, LearningCurve_plot, empty_correspondingTXTfile, transform_vgg, \
    transform
from functions import save_correspondingMODEL


def main():
    # check if GPU is available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # parameters setting
    EPOCH = 50
    BATCH_SIZE = 128
    ifnormalized = False # if you want a normalized matrix, set this to be true

    # Cifar-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # choose the algorithm to run
    algorithm = input("Select your algorithm: ResNet,AlexNet,LeNet5,VGG11(R/A/L/V) : ").upper()
    if algorithm =='R':
        transform_train, transform_test = transform()
        net = ResNet18().to(device)
    elif algorithm =='A':
        transform_train, transform_test = transform()
        net = AlexNet().to(device)
    elif algorithm =='L':
        transform_train, transform_test = transform()
        net = LeNet_5().to(device)
    elif algorithm =='V':
        transform_train, transform_test = transform_vgg()
        net = VGG11().to(device)
    else:
        print("Please enter a correct algorithm")
        exit()

    # loss function and optimizer
    if algorithm == 'V':
        optimizer = optim.Adam(net.parameters(), lr=0.0001)
    else:
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    # Data loading
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory = True)

    # Training will train a new model, testing will use the model that saved from the last train
    choice = input("Start a new training or Testing saved model? (A/B): ").upper()
    if choice == 'A':

        # empty the data recorded form last training if it's starting a new train
        empty_correspondingTXTfile(algorithm)

        #In the training()function, we test the network each epoch but we don't caluclate the F1-score, Recall
        #and precision. Thoses three parameters will be claculated once the train is finished
        training(net, device, trainloader, testloader, optimizer, EPOCH, algorithm,classes)
        # save the selected model
        save_correspondingMODEL(net,algorithm)
    else:
        #testing_existingModel() function will test the saved model,
        # then generate confusion matrix for that model and
        # the values of Precision, Recall and F1-score
        if algorithm == 'R':
            testing_savedModel(net, device, testloader, algorithm, classes, ifnormalized, PATH="./result/cifar_ResNet.pth")
        elif algorithm == 'A':
            testing_savedModel(net, device, testloader, algorithm, classes, ifnormalized, PATH="./result/cifar_AlexNet.pth")
        elif algorithm == 'L':
            testing_savedModel(net, device, testloader, algorithm, classes, ifnormalized, PATH="./result/cifar_LeNet5.pth")
        elif algorithm == 'V':
            testing_savedModel(net, device, testloader, algorithm, classes, ifnormalized, PATH="./result/cifar_VGG11.pth")
        # We saved the training accuracy and loss data in the .txt file.
        # In this case to view the curve, users don't need to start a new training to plot the curve.
        # Select the testing mode and the learning curve from the last train will pop up
        LearningCurve_plot(algorithm)

if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
import numpy as np
import itertools
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

########     Testing section     ########
def testing_savedModel(net,device, testloader, algorithm, classes, ifnormalized, PATH):
    print("Waiting Test for saved model!")
    with torch.no_grad():
        answer = 0
        total = 0
        conf_matrix = torch.zeros([10, 10], device=device)
        net.load_state_dict(torch.load(PATH))#load the model
        net.cuda()
        for data in testloader:
            net.eval()
            images, targets = data
            images, targets = images.to(device), targets.to(device)
            outputs = net(images)

            _, prediction = torch.max(outputs.data, 1)

            # summing up predicted labels and true labels
            total += targets.size(0)
            answer += (prediction == targets).sum()

            pred = torch.max(outputs, 1)[1]
            #passing the value to cofusion_matrix
            conf_matrix = confusion_matrix(pred, targets, conf_matrix=conf_matrix)

        print('Testing Accuracy：%.3f%%' % (100 * answer / total))

        # drawing confusion matrix
        plot_confusion_matrix(conf_matrix.cpu().numpy(), algorithm, classes=classes, normalize=ifnormalized)
        Precision_Recall_F1(conf_matrix.cpu().numpy(),classes)

########     Training section     ########
def training(net, device, trainloader, testloader, optimizer, EPOCH, algorithm, classes):
    print("Start Training")
    with open("./DataToPlot/ResNet18_LearningCurve.txt", "a") as f:
        with open("./DataToPlot/AlexNet_LearningCurve.txt", "a") as f1:
            with open("./DataToPlot/LeNet5_LearningCurve.txt", "a") as f2:
                with open("./DataToPlot/VGG11_LearningCurve.txt", "a") as f3:
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

                            # summing up loss and number of corrections
                            sum_loss += loss.item()
                            _, prediction = torch.max(outputs.data, 1)
                            total += targets.size(0)

                            correct += prediction.eq(targets.data).cpu().sum()

                            #printing out epochs and iterations
                            print('Training: [epoch:%d, iter:%d] Loss: %.03f | Accurracy: %.3f%% '
                                  % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))

                        # writing the results into .txt file corresponding to the algorithm
                        # .txt file will be used to plot the learning curve
                        if algorithm == 'R':
                            f.write('%03d %.3f %.03f' % ((epoch+1), 100. * correct / total, sum_loss / (i + 1)))
                            f.write('\n')
                            f.flush()
                        elif algorithm == 'A':
                            f1.write('%03d %.3f %.03f' % ((epoch+1), 100. * correct / total, sum_loss / (i + 1)))
                            f1.write('\n')
                            f1.flush()
                        elif algorithm == 'L':
                            f2.write('%03d %.3f %.03f' % ((epoch+1), 100. * correct / total, sum_loss / (i + 1)))
                            f2.write('\n')
                            f2.flush()
                        elif algorithm == 'V':
                            f3.write('%03d %.3f %.03f' % ((epoch+1), 100. * correct / total, sum_loss / (i + 1)))
                            f3.write('\n')
                            f3.flush()

                    ##### Testing after one epoch is finished training #####
                        print("Waiting Test!")
                        class_correct = list(0. for i in range(10))
                        class_total = list(0. for i in range(10))
                        with torch.no_grad():
                            answer = 0
                            total = 0
                            conf_matrix = torch.zeros([10, 10], device='cuda')
                            for data in testloader:
                                net.eval()
                                images, targets = data
                                images, targets = images.to(device), targets.to(device)
                                outputs = net(images)
                                # get the class which has the highest accuracy
                                _, prediction = torch.max(outputs.data, 1)

                                total += targets.size(0)
                                answer += (prediction == targets).sum()
                                c = (prediction == targets).squeeze()
                                for i in range(4):
                                    label = targets[i]
                                    class_correct[label] += c[i].item()
                                    class_total[label] += 1
                        print('[EPOCH:%d]: ' % ((epoch + 1)))
                        for i in range(10):
                            print('Accuracy of %5s : %2d %%' % (
                                classes[i], 100 * class_correct[i] / class_total[i]))

                    print("Training Finished, Total EPOCH=%d" % EPOCH)
                    f, f1, f2, f3.close()

######### Functions to plot learning curve, confusion matrix #########
# function to plot the learning curve
def LearningCurve_plot(algorithm):
    #open different txt file corresponding to the selected algorothm
    if algorithm == 'R':
        filename = './DataToPlot/ResNet18_LearningCurve.txt'
    elif algorithm == 'A':
        filename = './DataToPlot/AlexNet_LearningCurve.txt'
    elif algorithm == 'L':
        filename = './DataToPlot/LeNet5_LearningCurve.txt'
    elif algorithm == 'V':
        filename = './DataToPlot/VGG11_LearningCurve.txt'


    X, Y, Z = [], [], []
    # reading values from txt file to plot the curve
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in lines:
            value = [float(s) for s in i.split()]
            X.append(value[0])
            Y.append(value[1])
            Z.append(value[2])

    fig = plt.figure()  # create figure object

    ax = fig.add_subplot(111)  # 1x1 grid, first subplot

    # titles corresponding to the selected algorothm
    if algorithm =='R':
        ax.set_title("ResNet18 Training Accuraccy and Loss")
    elif algorithm =='A':
        ax.set_title("AlexNet Training Accuraccy and Loss")
    elif algorithm =='L':
        ax.set_title("LeNet5 Training Accuraccy and Loss")
    elif algorithm =='V':
        ax.set_title("VGG11 Training Accuraccy and Loss")


    ax.plot(X, Y, label='Accuracy', color="tab:red")# label accuracy and loss curves

    ax2 = ax.twinx()# accuracy and loss are sharing the same x axis
    ax2.plot(X, Z, label='Loss', color="tab:blue")# label accuracy and loss curves

    ax.legend(loc=6)#setting position of legend
    ax2.legend(loc=7)

    ax.grid()
    #labelling axis
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy(%)")
    ax2.set_ylabel("Loss(%)")

    plt.show()


# function to update confusion matrix
def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


# function to plot confusion matrix
def plot_confusion_matrix(cm, algorithm, classes, normalize):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # titles corresponding to the selected algorothm, and also if the confusion matrix is normalized
    if algorithm == 'R':
        if normalize:
            plt.title("ResNet18 Normalized Confusion Matrix")
        else:
            plt.title("ResNet18 Confusion Matrix")
    elif algorithm == 'A':
        if normalize:
            plt.title("AlexNet Normalized Confusion Matrix")
        else:
            plt.title("AlexNet Confusion Matrix")
    elif algorithm == 'L':
        if normalize:
            plt.title("LeNet5 Normalized Confusion Matrix")
        else:
            plt.title("LeNet5 Confusion Matrix")
    elif algorithm == 'V':
        if normalize:
            plt.title("VGG11 Normalized Confusion Matrix")
        else:
            plt.title("VGG11 Confusion Matrix")


    plt.colorbar()
    # label the tick with the name of the classes in dataset
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.#finding middle value of the matrix

    sum_col = cm.sum(1)

    # loop over the matrix and plot normalized value to see the precision
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            percentage = cm[i][j] / sum_col[i]
            plt.text(j, i, '{:.2f}'.format(percentage), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            # if the value is greater than half of the maximum value in matrix then set
            # the color of text to white, else it's black
            plt.text(j, i, '{:.0f}'.format(cm[i, j]), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()#automatically adjust plots
    #labelling axis
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

def Precision_Recall_F1(cm,classes):
    n = len(cm)
    for i in range(len(cm[0])):
        rowsum, colsum = sum(cm[i]), sum(cm[r][i] for r in range(n))
        Precision = cm[i][i] / float(colsum)
        Recall = cm[i][i] / float(rowsum)
        F1 = 2 * ((Recall * Precision) / (Precision + Recall))
        try:
            print('%s :' % classes[i],'Precision: %s' % Precision, 'Recall: %s' % (round(Recall,3)), 'F1-Score: %s' % (round(F1,3)))
        except ZeroDivisionError:
            print('precision: %s' % 0, 'recall: %s' % 0)

######### Functions to clear txt files and save models #########
def empty_correspondingTXTfile(algorithm):
    # empty the  corresponding txt file when a new train start
    if algorithm == 'R':
        f = open('./DataToPlot/ResNet18_LearningCurve.txt', 'r+')
        f.truncate()
        f.close()
    elif algorithm == 'A':
        f = open('./DataToPlot/AlexNet_LearningCurve.txt', 'r+')
        f.truncate()
        f.close()
    elif algorithm == 'L':
        f = open('./DataToPlot/LeNet5_LearningCurve.txt', 'r+')
        f.truncate()
        f.close()
    elif algorithm == 'V':
        f = open('./DataToPlot/VGG11_LearningCurve.txt', 'r+')
        f.truncate()
        f.close()

def save_correspondingMODEL(net,algorithm):
    # save corresponding model
    if algorithm == 'R':
        torch.save(net.state_dict(), "./result/cifar_ResNet.pth")
    elif algorithm == 'A':
        torch.save(net.state_dict(), "./result/cifar_AlexNet.pth")
    elif algorithm == 'L':
        torch.save(net.state_dict(), "./result/cifar_LeNet5.pth")
    elif algorithm == 'V':
        torch.save(net.state_dict(), "./result/cifar_VGG11.pth")


#data transform for vgg11 resize to 64*64
def transform_vgg():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # to flip some input images to reduce the impact of overfitting
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # R,G,B Normalized value: mean and variance
    ])
    transform_test = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    return transform_train, transform_test


def transform():
    # Data Transform
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 32*32
        transforms.RandomHorizontalFlip(),  # to flip some input images to reduce the impact of overfitting
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # R,G,B Normalized value: mean and variance
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    return transform_train, transform_test

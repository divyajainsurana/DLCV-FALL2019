import torch.nn as nn
import torchvision.models as models

class Net(nn.Module):
    def __init__(self, feature_size):
        super(Net, self).__init__()

        self.linear1 = nn.Linear(feature_size,4096)
        self.linear2 = nn.Linear(4096,1024)
        self.linear3 = nn.Linear(1024, 11)
        self.softmax = nn.Softmax(1)
        self.relu = nn.ReLU()
        self.bn_1 = nn.BatchNorm1d(4096)
        self.bn_2 = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = self.relu(self.bn_1(self.linear1(x))) # same as relu output
        x = self.relu(self.bn_2(self.linear2(x)))
        y_pred = self.softmax(self.linear3(x))
        return y_pred,x


class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        model = models.resnet50(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        output = self.feature(x)
        output = output.view(-1,2048)
        return output

import torch.nn as nn
import torchvision.models as models

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, predict, target):
        loss_fn = nn.CrossEntropyLoss()
        loss = 0
        batch_size = len(predict)

        for i in range(batch_size):
            #print(predict[i].size(), target[i].size())
            partial_loss = loss_fn(predict[i], target[i])
            loss += partial_loss
        loss = loss / batch_size
        return loss

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=512, n_layers=1, dropout=0):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size, num_layers=n_layers,dropout=dropout, bidirectional=False)

        self.linear1 = nn.Linear(self.hidden_size,1024)
        self.linear2 = nn.Linear(1024,256)
        self.linear3 = nn.Linear(256, 11)
        self.softmax = nn.Softmax(1)
        self.bn_1 = nn.BatchNorm1d(1024)
        self.bn_2 = nn.BatchNorm1d(256)
        self.softmax = nn.Softmax(1)
        self.relu = nn.ReLU()

    def forward(self, padded_sequence, input_lengths):
        #batch_size x longest sequence x 2048d
        packed = nn.utils.rnn.pack_padded_sequence(padded_sequence, input_lengths,batch_first=True)
        outputs, (hn,cn) = self.lstm(packed) # output: (seq_len, batch, hidden size)
        hidden_output = hn[-1]

        outputs,_ = nn.utils.rnn.pad_packed_sequence(outputs,batch_first=True)
        result = []
        for output in outputs:
            output = self.relu(self.bn_1(self.linear1(output))) # same as relu output
            output = self.relu(self.bn_2(self.linear2(output)))
            result_part = self.softmax(self.linear3(output))
            result.append(result_part)
        return result


class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        model = models.resnet50(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        """
            input shape: 224 x 224
            output shape: batch size x 2048 x 1 x 1
        """
        output = self.feature(x)
        output = output.view(-1,2048)
        return output

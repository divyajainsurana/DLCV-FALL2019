import sys
import os
import numpy as np
import torch
from torchvision import transforms
from reader import getVideoList

test_features_path = sys.argv[1]
test_label_path = sys.argv[2] 
model_path = sys.argv[3]
pre_label_path = sys.argv[4]

# loading features extracted by pretrain model
print("\nloading videos feature...")
test_features = torch.load(test_features_path).view(-1,2048)
print("test_features",test_features.shape)

# load model
my_net = torch.load(model_path)
my_net = my_net.eval()
my_net = my_net.cuda()
predict_labels,_ = my_net(test_features.cuda())
predict_vals = torch.argmax(predict_labels,1).cpu().data
print("prediction:",predict_vals)

# output as csv file
with open(os.path.join(pre_label_path,"p1_valid.txt"),'w') as f:
    for i,predict_val in enumerate(predict_vals):
        f.write(str(int(predict_val)))
        if (i==len(predict_vals)-1): break
        f.write('\n')
print("save predicted file at",os.path.join(pre_label_path,"p1_valid.txt"))

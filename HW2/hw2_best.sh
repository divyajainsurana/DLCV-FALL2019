wget https://www.dropbox.com/s/jxen8qsh2lr7c9q/model_best_improved.pth.tar?dl=0 model_best_improved.pth.tar
mv 'model_best_improved.pth.tar?dl=0' model_best_improved.pth.tar

python3 test.py $1 $2 model_best_improved.pth.tar Net_improved

#python3 mean_iou_evaluate.py -g $2 -p $3
#Make sure that that $3 represents  the path to ground truth directory


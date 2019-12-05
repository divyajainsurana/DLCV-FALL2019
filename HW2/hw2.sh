wget https://www.dropbox.com/s/lr2u80vv5fnaemg/model_best.pth.tar?dl=0 model_best.pth.tar 

mv 'model_best.pth.tar?dl=0' model_best.pth.tar
python3 test.py $1 $2 model_best.pth.tar Net

#python3 mean_iou_evaluate.py -g $2 -p $3
#Make sure that that $3 represents  the path to ground truth directory

# TODO: create shell script for Problem 1i
wget https://www.dropbox.com/s/hnvdmohm2ho2bo2/best_cnnbased.pth?dl=0 best_cnnbased.pth
mv 'best_cnnbased.pth?dl=0' best_cnnbased.pth
python3 problem1/featureExtractor.py $1 $2 ./problem1 valid 
python3 problem1/evaluate_p1.py ./problem1/valid_features.pt $2 best_cnnbased.pth $3

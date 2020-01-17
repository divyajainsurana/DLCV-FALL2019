# TODO: create shell script for Problem 3
wget https://www.dropbox.com/s/bc50aljubt7sgqc/best_temp.pth?dl=0 best_temp.pth
mv 'best_temp.pth?dl=0' best_temp.pth
python3 problem3/featureExtractor.py $1 ./ ./problem3 test
python3 problem3/evaluate_p3.py problem3/test_features.pt problem3/test_infos.pt best_temp.pth $2

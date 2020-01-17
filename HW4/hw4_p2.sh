# TODO: create shell script for Prolem
wget https://www.dropbox.com/s/vci25hphnypmni0/best_rnnbased.pth?dl=0 best_rnnbased.pth
mv 'best_rnnbased.pth?dl=0' best_rnnbased.pth
python3 problem2/featureExtractor.py $1 $2 ./problem2 test
python3 problem2/evaluate_p2.py ./problem2/test_rnnfeatures.pt $2 best_rnnbased.pth $3

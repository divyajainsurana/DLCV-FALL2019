wget https://www.dropbox.com/s/upk4iw1nyjxbbxq/best_model.pth.tar?dl=0 best_model.pth.tar
mv 'best_model.pth.tar?dl=0' best_model.pth.tar
python3 test.py $1 $2

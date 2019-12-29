wget https://www.dropbox.com/s/q8wshanm551diia/epoch95_checkpoint.pth.tar?dl=0 epoch95_checkpoint.pth.tar
mv 'epoch95_checkpoint.pth.tar?dl=0' epoch95_checkpoint.pth.tar

wget https://www.dropbox.com/s/d4hdlpofovzytf7/epoch100_checkpoint.pth.tar?dl=0 epoch100_checkpoint.pth.tar
mv 'epoch100_checkpoint.pth.tar?dl=0' epoch100_checkpoint.pth.tar
python3 dcgan_inference.py -ckpt epoch100_checkpoint.pth.tar $1
python3 acgan_inference.py -ckpt epoch95_checkpoint.pth.tar $1

 

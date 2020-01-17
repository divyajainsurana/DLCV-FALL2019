import os
import glob
import numpy as np
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from model import Resnet50
tqdm.pandas()

# data source
video_path = sys.argv[1] 
labelfile_path = sys.argv[2] 
save_path = sys.argv[3]
mode = sys.argv[4]


# getting absolute filepath
videodir_filenames = sorted(glob.glob(os.path.join(video_path, '*')))
videodir_names = [videodir_filename.split('/')[-1] for videodir_filename in videodir_filenames]
label_filenames = []
if mode != "test": label_filenames = sorted(glob.glob(os.path.join(labelfile_path, '*.txt')))


# loading videos
videos = []
videos_len = []
print("\nloading videos...")
with tqdm(total=len(videodir_filenames)) as pbar:
    for videodir_filename in videodir_filenames:
        video = []
        frame_filenames = sorted(glob.glob(os.path.join(videodir_filename, '*')))
        videos_len.append(len(frame_filenames))
        for frame_filename in frame_filenames:
            frame = Image.open(frame_filename).convert('RGB')
            video.append(frame)
        videos.append(video)
        pbar.update(1)


# extracting features
cnn_feature_extractor = Resnet50().cuda() # to 2048 dims
transform=transforms.Compose([
                        transforms.Pad((0,40),fill=0,padding_mode='constant'),
                        transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                        ])
cnn_feature_extractor.eval()
videos_features = []
print("\nextracting videos feature...")
with torch.no_grad():
    with tqdm(total=len(videodir_filenames)) as pbar:
        for video in videos:
            # make a video as a large torch batch
            local_batch = []
            for frame in video:
                frame = transform(frame)
                local_batch.append(frame)
            local_batch = torch.stack(local_batch)

            # extract feature by parts
            video_features = []
            datalen = len(local_batch)
            BATCH_SIZE = 200
            for batch_idx, batch_val in enumerate(range(0,datalen ,BATCH_SIZE)):
                if batch_val+BATCH_SIZE > datalen: input_part = local_batch[batch_val:]
                input_part = local_batch[batch_val:batch_val+BATCH_SIZE]
                video_features_part = cnn_feature_extractor(input_part.cuda())
                videos_features.append(video_features_part)
            pbar.update(1)


# save feature torch as file
torch.save(videos_features, os.path.join(save_path,mode+'_features.pt'))
# load feature file as torch for test
videos_features = torch.load(os.path.join(save_path,mode+'_features.pt'))

# loading labels for each videos
if mode != "test":
    print("\nloading labels...")
    videos_label_vals = []
    for label_filename in label_filenames:
        f = open(label_filename,'r')
        video_label_vals = f.read().splitlines()
        video_label_vals = np.array(video_label_vals).astype(int)
        datalen = len(video_label_vals)
        BATCH_SIZE = 200
        for batch_idx, batch_val in enumerate(range(0,datalen ,BATCH_SIZE)):
            if batch_val+BATCH_SIZE > datalen: input_part = video_label_vals[batch_val:]
            video_label_vals_part = video_label_vals[batch_val:batch_val+BATCH_SIZE]
            video_label_vals_part = torch.LongTensor(video_label_vals_part)
            videos_label_vals.append(video_label_vals_part)
            print(len(video_label_vals_part))
    # save feature torch as file
    torch.save(videos_label_vals, os.path.join(save_path,mode+'_vals.pt'))
    # load feature file as torch for test
    test_labels = torch.load(os.path.join(save_path,mode+'_vals.pt'))
    print(mode+"_labels:",len(test_labels))

# save video info
if mode == "valid" or "test":
    output = []
    output.append(videodir_names)
    output.append(videos_len)
    torch.save(output, os.path.join(save_path,mode+'_infos.pt'))

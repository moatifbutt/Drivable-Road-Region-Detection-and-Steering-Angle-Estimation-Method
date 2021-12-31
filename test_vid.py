import cv2
import torch
import argparse
import time
import albumentations
import config
import numpy as np

from PIL import Image
from model import model
from utils.helpers import draw_test_segmentation_map, image_overlay

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, 
                    help='path to input image')
parser.add_argument('-m', '--model-path', dest='model_path', required=True, 
                    help='path to the trained weight file')
args = vars(parser.parse_args())

# define the image transforms
transform = albumentations.Compose([
    albumentations.Normalize(
            mean=[0.45734706, 0.43338275, 0.40058118],
            std=[0.23965294, 0.23532275, 0.2398498],
            always_apply=True)
])

# load the model
model = model.to(config.DEVICE)
checkpoint = torch.load(args['model_path'])
# load the trained weights
model.load_state_dict(checkpoint['model_state_dict'])
# set the model to eval model
model.eval()

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(args['input'])
if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')

# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

save_name = f"{args['input'].split('/')[-1].split('.')[0]}"
# define codec and create VideoWriter object 
out = cv2.VideoWriter(f"outputs/{save_name}.mp4", 
                      cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                      (frame_width, frame_height))

frame_count = 0 # to count total frames
total_fps = 0 # to get the final frames per second

# read until end of video
from google.colab.patches import cv2_imshow
while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret == True:
        # get the start time
        start_time = time.time()
        with torch.no_grad():
            orig_frame = frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = transform(image=frame)['image']
            frame = np.transpose(frame, (2, 0, 1))
            frame = torch.tensor(frame, dtype=torch.float32)
            frame = frame.unsqueeze(0).to(config.DEVICE)
            # get predictions for the current frame
            outputs = model(frame)
        
        # draw boxes and show current frame on screen
        segmented_image = draw_test_segmentation_map(outputs['out'])

        final_image = image_overlay(orig_frame, segmented_image)

        # get the end time
        end_time = time.time()
        # get the fps
        fps = 1 / (end_time - start_time)
        # add fps to total fps
        total_fps += fps
        # increment frame count
        frame_count += 1

        # press `q` to exit
        wait_time = max(1, int(fps/4))
        #cv2.imshow('image', final_image)
        cv2_imshow(final_image)
        out.write(final_image)
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
    else:
        break

# release VideoCapture()
cap.release()
# close all frames and video windows
cv2.destroyAllWindows()

# calculate and print the average FPS # while during live no need to comment it 
#avg_fps = total_fps / frame_count
#print(f"Average FPS: {avg_fps:.3f}")

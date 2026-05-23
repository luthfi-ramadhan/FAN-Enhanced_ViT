import numpy as np
import os
import h5py
import math
import matplotlib.pyplot as plt
import cv2
from skimage.metrics import structural_similarity as ssim


def sampling(x, t):

    _, w, h, c = x.shape
    sampled = np.zeros((w, h, c))
    sampled_data = []
    for i in range(len(x)):
        if i == 0:
            sampled = x[i]
            sampled_data.append(sampled)
        else:

            w, h, _ = sampled.shape
            sampled_img = np.reshape(sampled, (w, h))
            new = x[i]
            new = np.reshape(new, (w, h))
            similarity  = ssim(sampled_img, new, data_range=new.max() - new.min())
            

            if similarity <= t:
                sampled = x[i]
                sampled_data.append(sampled)
            else:
                pass
    return np.array(sampled_data)

def interpolate_frames_bidirectional(f1, f2, n):
    gray1 = f1
    gray2 = f2

  
    flow12 = cv2.calcOpticalFlowFarneback(gray1, gray2, None,
                                          pyr_scale=0.5, levels=3,
                                          winsize=15, iterations=3,
                                          poly_n=5, poly_sigma=1.2,
                                          flags=0)

    flow21 = cv2.calcOpticalFlowFarneback(gray2, gray1, None,
                                          pyr_scale=0.5, levels=3,
                                          winsize=15, iterations=3,
                                          poly_n=5, poly_sigma=1.2,
                                          flags=0)

    h, w, c = gray1.shape
    interpolated = []

    for i in range(1, n + 1):
        alpha = i / (n + 1)
   
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        grid_x = grid_x.astype(np.float32)
        grid_y = grid_y.astype(np.float32)


        flow_fwd = flow12 * alpha
        map_x_f1 = grid_x + flow_fwd[..., 0]
        map_y_f1 = grid_y + flow_fwd[..., 1]
        warped_f1 = cv2.remap(f1, map_x_f1, map_y_f1, cv2.INTER_LINEAR)


        flow_bwd = flow21 * (1 - alpha)
        map_x_f2 = grid_x + flow_bwd[..., 0]
        map_y_f2 = grid_y + flow_bwd[..., 1]
        warped_f2 = cv2.remap(f2, map_x_f2, map_y_f2, cv2.INTER_LINEAR)

        blended = cv2.addWeighted(warped_f1, 1 - alpha, warped_f2, alpha, 0)
        interpolated.append(blended)

    return np.array(interpolated)

def distributed_interpolation(frames, target_len):
    N = len(frames)
    required = target_len - N
    if required <= 0:
        return frames[:target_len]

    intervals = N - 1
    interp_plan = [0] * intervals
   


    for i in range(required):
        interp_plan[i % intervals] += 1

    result = []
    for i in range(intervals):

        result.append(frames[i])
        n_interp = interp_plan[i]
        
        if n_interp > 0:
            
            new_frames = interpolate_frames_bidirectional(frames[i], frames[i+1], n_interp)
            n, w, h = new_frames.shape
            for j in range(n):
                result.append(np.reshape(new_frames[j], (w, h, 1)))
            
    last_frame = frames[-1]
    result.append(last_frame)
    result = np.array(result)
    return result

def frame_count(video_path, manual=True):
    def manual_count(handler):
        frames = 0
        while True:
            status, frame = handler.read()
            if not status:
                break
            frames += 1
        return frames 

    cap = cv2.VideoCapture(video_path)

    if manual:
        frames = manual_count(cap)

    else:
        try:
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except:
            frames = manual_count(cap)
    cap.release()
    return frames

def interval(n_frame, frame2use):
    incr = n_frame / frame2use
    incr = math.floor(incr)
    idx = []
    for i in range(0, n_frame, int(incr)):
        idx.append(i)
    
    return idx[0:frame2use]

def load_video(video_path):
    # Load video
    cap = cv2.VideoCapture(video_path)
    vid = []
    while(cap.isOpened()):
        ret, frame = cap.read()

        
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (96, 96))
        frame = np.reshape(frame, (96, 96, 1))
        vid.append(frame)

    vid = np.array(vid) / 255
    return vid

def preprocessing(vid, t, max_frames):
    preprocessed_vid = []

   
    vid = sampling(vid, t)
    n_frame = len(vid)

    if n_frame <= max_frames:
        preprocessed_vid = distributed_interpolation(vid, max_frames)

    else:
        idx = interval(n_frame, max_frames)
    
        for i in range(n_frame):
            if i in idx:
                preprocessed_vid.append(vid[i])
            else:
                pass
    return np.array(preprocessed_vid)

t = 0.85
frame2use = 50

label = []
vid_tensor = []
for i in range(len(path2file_plax)):
    if("normal" in path2file_plax[i]):
        result = load_video(path2file_plax[i])
        
        result = preprocessing(result, t, frame2use)
        print(i, result.shape)
        label.append(0)
        vid_tensor.append(result)
    elif("preserved" in path2file_plax[i]):
        result = load_video(path2file_plax[i])
        result = preprocessing(result, t, frame2use)
        print(i, result.shape)
        label.append(1)
        vid_tensor.append(result)
    elif("reduced" in path2file_plax[i]):
        result = load_video(path2file_plax[i])
        result = preprocessing(result, t, frame2use)
        print(i, result.shape)
        label.append(2)
        vid_tensor.append(result)
    else:
        raise Exception("Error")
vid_tensor = np.array(vid_tensor)
label = np.array(label)

h5f = h5py.File('data/plax_85t_50x96x96x1.h5', 'w')
desc = 'This data is processed by applying distance-based frame selection using SSIM with a t-score of 0.85. After that, if n_frame > max_frame then it is squeezed using the old method'
desc = desc.encode("ascii", "ignore")
h5f.create_dataset('help', data=desc)
h5f.create_dataset('x', data=vid_tensor)
h5f.create_dataset('y', data=label)
path4file_4ch= [n.encode("ascii", "ignore") for n in path2file_plax]
h5f.create_dataset('path2file', data=path2file_plax)
h5f.close()
import numpy as np
def frames_to_sequence(feature_list, L, K, frames_per_video=50):
    sequences = []

    assert (frames_per_video - L - K + 1) > 0, "Invalid L, K for video length"
    assert len(feature_list) % frames_per_video == 0, "Feature list length mismatch"

    video_num = len(feature_list) // frames_per_video

    for i in range(video_num):
        start = i * frames_per_video
        end = start + frames_per_video

        for j in range(start, end - L - K + 1):
            sequences.append(feature_list[j : j + L])

    return sequences



def framelabel_to_sequencelabel(label_list,L,K,frames_per_video=50):
    y=[]
    video_num = len(label_list) // frames_per_video
    for i in range(video_num):
        start=i*frames_per_video+L
        end=(i*frames_per_video)+frames_per_video
        for j in range(start,end-K+1):
            label=0
            for k in range(j,j+K):  
                if label_list[k]==1:
                    label=1
                    break
            y.append(label)
        
    return np.array(y)
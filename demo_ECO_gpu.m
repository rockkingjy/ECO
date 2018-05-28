
% This demo script runs the ECO tracker with deep features on the
% included "Crossing" video.

% Add paths
setup_paths();

% Load video information
%video_path = '/media/elab/sdd/data/VOT/vot2017/drone_flip';%drone_across';%'sequences/Crossing';
[seq, ground_truth] = load_video_info_vot();

% Run ECO
results = testing_ECO_HC(seq);
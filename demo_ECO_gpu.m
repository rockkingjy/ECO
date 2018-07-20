
% This demo script runs the ECO tracker with deep features on the
% included "Crossing" video.

% Add paths
setup_paths();

% Load video information
[seq, ground_truth] = load_video_info_vot();

% Run ECO
results = testing_ECO_gpu(seq);
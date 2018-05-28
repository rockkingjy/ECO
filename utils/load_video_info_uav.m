function [seq, ground_truth] = load_video_info_uav()

video_path = '/media/elab/sdd/data/UAV123';
file = 'wakeboard3';
ground_truth = dlmread([video_path '/anno/UAV123/' file '.txt']);

seq.format = 'otb';
seq.len = size(ground_truth, 1);
seq.init_rect = ground_truth(1,:);

img_path = [video_path '/data_seq/UAV123/' file '/'];

if exist([img_path num2str(1, '%06i.png')], 'file'),
    img_files = num2str((1:seq.len)', [img_path '%06i.png']);
elseif exist([img_path num2str(1, '%06i.jpg')], 'file'),
    img_files = num2str((1:seq.len)', [img_path '%06i.jpg']);
elseif exist([img_path num2str(1, '%06i.bmp')], 'file'),
    img_files = num2str((1:seq.len)', [img_path '%06i.bmp']);
else
    error('No image files to load.')
end

seq.s_frames = cellstr(img_files);

end


function [seq, ground_truth] = load_video_info_tlp()

video_path = '/media/elab/sdd/data/TLP/Sam';%Alladin';%dinosaur';%drone_across';
ground_truth = dlmread([video_path '/groundtruth_rect.txt']);

seq.format = 'otb';
seq.len = size(ground_truth, 1);
seq.init_rect = ground_truth(1,2:5);

img_path = [video_path '/img/'];

if exist([img_path num2str(1, '%05i.png')], 'file'),
    img_files = num2str((1:seq.len)', [img_path '%05i.png']);
elseif exist([img_path num2str(1, '%05i.jpg')], 'file'),
    img_files = num2str((1:seq.len)', [img_path '%05i.jpg']);
elseif exist([img_path num2str(1, '%05i.bmp')], 'file'),
    img_files = num2str((1:seq.len)', [img_path '%05i.bmp']);
else
    error('No image files to load.')
end

seq.s_frames = cellstr(img_files);

end


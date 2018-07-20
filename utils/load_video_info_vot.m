function [seq, ground_truth] = load_video_info_vot()

video_path = '/media/elab/sdd/data/VOT/vot2017/gymnastics3';%iceskater1';%road';%%girl';%flamingo1';%drone_across';
ground_truth = dlmread([video_path '/groundtruth.txt']);

seq.format = 'otb';
seq.len = size(ground_truth, 1);

ground_truth(1,:)
x = min(ground_truth(1,1), ground_truth(1,7));
y = min(ground_truth(1,2), ground_truth(1,4));
w = max(ground_truth(1,3), ground_truth(1,5)) - x;
h = max(ground_truth(1,6), ground_truth(1,8)) - y;

seq.init_rect = [x,y,w,h];

img_path = [video_path '/'];

if exist([img_path num2str(1, '%08i.png')], 'file'),
    img_files = num2str((1:seq.len)', [img_path '%08i.png']);
elseif exist([img_path num2str(1, '%08i.jpg')], 'file'),
    img_files = num2str((1:seq.len)', [img_path '%08i.jpg']);
elseif exist([img_path num2str(1, '%08i.bmp')], 'file'),
    img_files = num2str((1:seq.len)', [img_path '%08i.bmp']);
else
    error('No image files to load.')
end

seq.s_frames = cellstr(img_files);

end


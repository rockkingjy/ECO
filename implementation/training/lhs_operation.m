function hf_out = lhs_operation(hf, samplesf, reg_filter, sample_weights)

% This is the left-hand-side operation in Conjugate Gradient

t1 = toc();

% 1: Get sizes
num_features = length(hf);
filter_sz = zeros(num_features,2);
for k = 1:num_features
    filter_sz(k,:) = [size(hf{k},1), size(hf{k},2)];
end
[~, k1] = max(filter_sz(:,1));  % Index for the feature block with the largest spatial size
block_inds = 1:num_features;
block_inds(k1) = [];
output_sz = [size(hf{k1},1), 2 * size(hf{k1},2) - 1];

t2 = toc();
disp(['update train time3_1_1 ' num2str(t2-t1)]);
t1 = toc();

% Compute the operation corresponding to the data term in the optimization
% (blockwise matrix multiplications)
%implements: A' diag(sample_weights) A f

% 2: sum over all features and feature blocks-------------------------------
sh = mtimesx(samplesf{k1}, permute(hf{k1}, [3 4 1 2]), 'speed');
pad_sz = cell(1,1,num_features);
for k = block_inds % add other features
    pad_sz{k} = (output_sz - [size(hf{k}, 1), 2 * size(hf{k}, 2) - 1]) / 2;
    sh(:, 1, 1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end) = ...
        sh(:, 1, 1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end) ...
        + mtimesx(samplesf{k}, permute(hf{k}, [3 4 1 2]), 'speed');
end
% weight all the samples
sh = bsxfun(@times,sample_weights,sh);

t2 = toc();
disp(['update train time3_1_2 ' num2str(t2-t1)]);
t1 = toc();

% 3: multiply with the transpose----------------------------------------------
hf_out = cell(1,1,num_features);
hf_out{k1} = permute(conj(mtimesx(sh, 'C', samplesf{k1}, 'speed')), [3 4 2 1]);
for k = block_inds % do on other features
    hf_out{k} = permute(conj(mtimesx(...
        sh(:, 1, 1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end), ...
        'C', samplesf{k}, 'speed')), [3 4 2 1]);
end

t2 = toc();
disp(['update train time3_1_3 ' num2str(t2-t1)]);
t1 = toc();

% 4: compute the operation corresponding to the regularization term (convolve
% each feature dimension with the DFT of w, and the tramsposed operation)
% add the regularization part-------------------------------------------------
% hf_conv = cell(1,1,num_features);
for k = 1:num_features
    reg_pad = min(size(reg_filter{k},2)-1, size(hf{k},2)-1);
    
    % add part needed for convolution
    hf_conv = cat(2, hf{k}, conj(rot90(hf{k}(:, end-reg_pad:end-1, :), 2)));
    
    % do first convolution
    hf_conv = convn(hf_conv, reg_filter{k});
    
    % do final convolution and put toghether result
    hf_out{k} = hf_out{k} + convn(hf_conv(:, 1:end-reg_pad, :), reg_filter{k}, 'valid');
end

t2 = toc();
disp(['update train time3_1_4 ' num2str(t2-t1)]);

end

% 1 feature
% update train time3_1_1 0.000198
% update train time3_1_2 0.000327
% update train time3_1_3 0.000406
% update train time3_1_4 0.001009
% 3 features
% update train time3_1_1 7.6e-05
% update train time3_1_2 0.000675
% update train time3_1_3 0.000575
% update train time3_1_4 0.001659
% 1 feature -----------------------------------
% samplesf: 1x1 cell - 30 x 10 x 25 x 13
% hf:       1x1 cell - 25 x 13 x 10
% sh:       30 x 1 x 25 x 13
% hf_out:   1x1 cell - 25 x 13 x 10
% sample_weights: 30 x 1
% hf_conv:  27 x 25 x 10 

% 3 features -----------------------------------
% samplesf: 1x1x3 cell
%           30 x 16 x 33 x 17
%           30 x 64 x 9  x 5
%           30 x 10 x 21 x 11
% hf:       1x1x3 cell
%           33 x 17 x 16
%           9  x 5  x 64
%           21 x 11 x 10
% sh:       30 x 1 x 33 x 17
% hf_out:   1x1x3 cell
%           33 x 17 x 16
%           9  x 5  x 64
%           21 x 11 x 10
% sample_weights: 30 x 1
% hf_conv:  23 x 23 x 10 

% ---------------------------------------------
% permute(hf{k}, [3 4 1 2]) % 25 x 13 x 10 -> 10 x 1 x 25 x 13
% C = mtimesx(A,B) % performs the calculation C = A * B
% C = mtimesx(A,'T',B) % performs the calculation C = A.' * B
% C = mtimesx(A,B,'g') % performs the calculation C = A * conj(B)
% C = mtimesx(A,'c',B,'C') % performs the calculation C = A' * B'


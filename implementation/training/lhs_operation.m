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
output_sz = [size(hf{k1},1), 2*size(hf{k1},2)-1];

t2 = toc();
disp(['update train time3_1_1 ' num2str(t2-t1)]);
t1 = toc();

% Compute the operation corresponding to the data term in the optimization
% (blockwise matrix multiplications)
%implements: A' diag(sample_weights) A f

% 2: sum over all features and feature blocks-------------------------------
% samplesf(k1}: 30 x 10 x 25 x 13
% permute: 25 x 13 x 10 -> 10 x 1 x 25 x 13
% sh: 30 x 1 x 25 x 13
sh = mtimesx(samplesf{k1}, permute(hf{k1}, [3 4 1 2]), 'speed');    % assumes the feature with the highest resolution is first
pad_sz = cell(1,1,num_features);
for k = block_inds % num_features
    pad_sz{k} = (output_sz - [size(hf{k}, 1), 2 * size(hf{k}, 2) - 1]) / 2;
    sh(:, 1, 1 + pad_sz{k}(1): end - pad_sz{k}(1), 1 + pad_sz{k}(2): end) = ...
        sh(:, 1, 1 + pad_sz{k}(1): end - pad_sz{k}(1), 1 + pad_sz{k}(2): end) ...
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
for k = block_inds % num_features
    hf_out{k} = permute(conj(mtimesx(...
        sh(:, 1, 1 + pad_sz{k}(1): end - pad_sz{k}(1),1 + pad_sz{k}(2):end), ...
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
    hf_out{k} = hf_out{k} + convn(hf_conv(:,1:end-reg_pad,:), reg_filter{k}, 'valid');
end
% hf_out: 25 x 13 x 10

t2 = toc();
disp(['update train time3_1_4 ' num2str(t2-t1)]);

end
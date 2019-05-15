D = im2double(imread('mountain.png'));
h = size(D,1);
w = size(D,2);
D = reshape(D, h*w, 3);
global N K;
N = size(D,1);
K = 20;
max_iter = 20;

% Initialize lambdas uniformly
lambdas = ones(K,1) / K;
% initialize means by drawing from uniform distriution
mus = rand(K,3);
% initialize covariance matrices as diagonal matrices, with elements
% uniformly drawn between 0 and 0.2
% sigs = rand(K,3) / 5;
sigmas = zeros(3,3,K);
for k = 1:K
    sigmas(:,:,k) = diag(0.1*ones(3,1));
end


% cluster by doing 100 iterations
% TODO: Calcualalte likelihood to measure progress in learning and use as
% termination criterion.
tic
[resp,old_log_l] = e_step(D,lambdas, mus, sigmas);
[lambdas, mus, sigmas] = m_step(D, resp);
display_compression(resp,mus,h,w)
log_diff = 100;
iter = 1;
while log_diff >= 100 && iter < 100
    [resp,log_l] = e_step(D,lambdas, mus, sigmas);
    [lambdas, mus, sigmas] = m_step(D, resp);
    log_diff = log_l - old_log_l;
    old_log_l = log_l;
    iter = iter + 1;
    fprintf('At Iteration %2d: Log Probability is %f, an improvement of %f\n',iter, old_log_l, log_diff)
    if mod(iter,10) == 0
        display_compression(resp,mus,h,w)
    end
    
end
toc

color_palette = reshape(mus, K,1,3);
figure; imagesc(color_palette)

function display_compression(resp,mus,h,w)
    [~,idx] = max(resp,[], 2);
    compressed_img = reshape(mus(idx,:),h,w,3);
    figure; imagesc(compressed_img)
end

function [new_lambdas, new_mus, new_sigmas] = m_step(D, resp)
    global N K;
    new_lambdas = zeros(K,1);
    new_mus = zeros(K,3);
    new_sigmas = zeros(3,3,K);
    for k = 1:K
        normalizer = sum(resp(:,k));
        new_lambdas(k) = normalizer / N;
        new_mus(k,:) = sum(D .* resp(:,k)) / normalizer;
        new_sigmas(:,:,k) =  (D - new_mus(k,:))' * (resp(:,k) .* (D-new_mus(k,:))) / normalizer;
    end
end

function [resp,log_l] = e_step(D, lambdas, mus, sigmas)
    global N K;
    resp = zeros(N,K);
    numerator = zeros(N,K);
    denominator = zeros(N,1);
    for j = 1:K
        numerator(:,j) = lambdas(j) * mvnpdf(D,mus(j,:),sigmas(:,:,j));
        denominator(:) = denominator(:) + numerator(:,j);
    end
    resp(:,:) = numerator(:,:)./denominator(:);
    log_l = sum(reallog(sum(numerator, 2)));
end


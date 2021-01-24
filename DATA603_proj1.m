data = load('data.mat');
face = data.face;
face_neutral = face(:,:,1:3:end);
face_exp = face(:,:,2:3:end);
face_illum = face(:,:,3:3:end);

% dimension = 200;
[d1, d2, dimension] = size(face_neutral);

neutral = zeros(d1*d2,dimension);
express = zeros(d1*d2,dimension);
illuminated = zeros(d1*d2,dimension);


for i = 1:dimension
    aux = face_neutral(:,:,i);
    neutral(:,i) = aux(:);
    aux = face_exp(:,:,i);
    express(:,i) = aux(:);
    aux = face_illum(:,:,i);
    illuminated(:,i) = aux(:);
end


%combine neutral and expression faces by concatenating columns horizontally
X= [neutral express];

[Y, U] = perform_pca(X', 20);

% recalculate this in order to create a plot
[dim1,dim2] = size(X');
X_mean = mean(X',1);
X_centered = X' - ones(dim1,1)*X_mean;

figure;
hold on; grid;
plot(X_centered(1:200,1), X_centered(1:200,2), '.', 'Markersize', 20, 'color', 'k');
plot(X_centered(201:400,1), X_centered(201:400,2), '.', 'Markersize', 20, 'color', 'r');
view(3);
title('PCA for data.mat, p=20');
legend('Neutral', 'Expression');

%% Split into Test and Train data 

% Split the data into Train
Y_neutral_train = data_splitter(Y, 1, dimension/2);
Y_express_train = data_splitter(Y, dimension+1, dimension + (dimension/2));

% Split the data into Test
Y_neutral_test = data_splitter(Y, dimension/2 + 1, dimension);
Y_express_test = data_splitter(Y, dimension + (dimension/2) + 1, 2*dimension);

Y_train = [Y_neutral_train; Y_express_train];

%% Now apply Bayesian Decision Theory

% Calculate the mean of each set
Y_mu_neutral_train = mean(Y_neutral_train, 1);
Y_mu_express_train = mean(Y_express_train, 1);
fprintf('norm = %d\n', norm(Y_mu_neutral_train  - Y_mu_express_train));

% Center the data sets
Y_train_neutral_center = Y_neutral_train - ones(dimension/2,1)*Y_mu_neutral_train;
Y_train_express_center = Y_express_train - ones(dimension/2,1)*Y_mu_express_train;

% Calculate the Covariance matrices
Y_cov_neutral_train = (1/dimension)*(Y_train_neutral_center'*Y_train_neutral_center);
Y_cov_express_train = (1/dimension)*(Y_train_express_center'*Y_train_express_center);
 
figure;
imagesc(Y_cov_neutral_train);
colorbar;

figure;
imagesc(Y_cov_express_train);
colorbar;

% discriminant function
inv_1 = inv(Y_cov_neutral_train);
inv_2 = inv(Y_cov_express_train);
mu1 = Y_mu_neutral_train';
mu2 = Y_mu_express_train';
w0 = 0.5*(log(det(Y_cov_neutral_train)/det(Y_cov_express_train))) - 0.5*(mu1'*inv_1*mu1 - mu2'*inv_2*mu2);
gaussian = @(x) - 0.5*x'*(inv_1-inv_2)*x + x'*(inv_1*mu1 - inv_2*mu2) + w0;

%% Classify test data using Gaussian classifier

nPCA = 20;

% test with illuminated data
illuminated = zeros(dimension, nPCA);
label = zeros(dimension,1);

for i = 1:dimension
    aux = face_illum(:,:,i);
    y = (aux(:)'*U(:,1:nPCA))';
    illuminated(i,:) = y';
    label(i) = sign(gaussian(y));
end

neutral_size = length(find(label > 0));
express_size = length(find(label < 0));

fprintf('Test with Illuminated data: \nNeutral = %d\nExpress = %d\n', neutral_size, express_size);
 
% test with neutral data 
[c1, c2] = size(Y_neutral_test);
label_neutral = zeros(c1,1);

for j = 1:c1
    label_neutral(j) = sign(gaussian((Y_neutral_test(j,:))'));
end

neutral_size1 = length(find(label_neutral > 0));
express_size1 = length(find(label_neutral < 0));

fprintf('Test with Neutral data: \nNeutral = %d\nExpress = %d\n', neutral_size1, express_size1);

% test with expression data 
[a1, a2] = size(Y_express_test);
label_express = zeros(a1,1);

for k = 1:a1
    label_express(k) = sign(gaussian((Y_express_test(k,:))'));
end

neutral_size2 = length(find(label_express > 0));
express_size2 = length(find(label_express < 0));

fprintf('Test with Expression data: \nNeutral = %d\nExpress = %d\n', neutral_size2, express_size2);

%% Classfiy test data using KNN

% test neutral set
limit = 0;
best_k = 0;
for k=1:2:30
    knn_answer = zeros(1,100);
    for i = 1:100
        current_dist = vecnorm(Y_train' - Y_neutral_test(i,:)');
        [C, I] = mink(current_dist, k);
        knn_neutral = length(find(I <100));
        knn_express = k - knn_neutral;
        knn_answer(1,i) = sign(knn_neutral - knn_express);
    end
    
    knn_neutral_size = length(find(knn_answer > 0));
    knn_express_size = length(find(knn_answer < 0));
    fprintf('k = %d\n', k);
    fprintf('KNN for Neutral test set: \nNeutral = %d\nExpress = %d\n', knn_neutral_size, knn_express_size);
    
    if knn_neutral_size > limit
        limit = knn_neutral_size;
        best_k = k;
    end
end

fprintf('Best k value = %d\n', best_k);

% test express set
limit = 0;
best_k = 0;
for k=1:2:30
    knn_answer = zeros(1,100);
    for z = 1:100
        current_dist = vecnorm(Y_train' - Y_express_test(z,:)');
        [C, I] = mink(current_dist, k);
        knn_express = length(find(I <100));
        knn_neutral = k - knn_express;
        knn_answer(1,z) = sign(knn_express - knn_neutral);
    end
    
    knn_express_size = length(find(knn_answer > 0));
    knn_neutral_size = length(find(knn_answer < 0));
    fprintf('k = %d\n', k);
    fprintf('KNN for Expression test set: \nNeutral = %d\nExpress = %d\n', knn_neutral_size, knn_express_size);
    
    if knn_express_size > limit
        limit = knn_express_size;
        best_k = k;
    end
end

fprintf('Best k value = %d\n', best_k);

%% Pose data

% 68 subjects in total
data_p = load('pose.mat');
pose = data_p.pose;

% only gathers poses for first 5 subjects
pca_poses = pose(:,:,:,1:5);

[d1, d2,pose_num,subject_num] = size(pca_poses);
P = [];

for i = 1:subject_num
    for j = 1:pose_num
        P(:,(i-1)*pose_num+j) = reshape(pose(:,:,j,i), [d1*d2 1]);
    end
end

%% Perform PCA on Pose data
% U represents the pixels of each image
nPCA = 20;
[pca_Y, U] = perform_pca(P',nPCA);

[dim1,dim2] = size(P');
P_mean = mean(P',1);
P_centered = P' - ones(dim1,1)*P_mean;

figure;
hold on; grid;

plot(P_centered(1:13,1), P_centered(1:13,2), '.', 'Markersize', 20, 'color', 'k');
plot(P_centered(14:26,1), P_centered(14:26,2), '.', 'Markersize', 20, 'color', 'r');
plot(P_centered(27:39,1), P_centered(27:39,2), '.', 'Markersize', 20, 'color', 'blue');
plot(P_centered(40:52,1), P_centered(40:52,2), '.', 'Markersize', 20, 'color', 'green');
plot(P_centered(53:65,1), P_centered(53:65,2), '.', 'Markersize', 20, 'color', 'magenta');
view(3);
title('PCA for pose.mat, p=20');

%% Multiple Discriminant Analysis

class_mu = zeros(subject_num, nPCA);
class_scatter = zeros(nPCA, nPCA, subject_num);

% row vectors
for i = 1:subject_num
    class_mu(i,:) = mean(pca_Y((i-1)*pose_num+1:i*pose_num,:),1);
    C_center_class = pca_Y((i-1)*pose_num+1:i*pose_num,:)-ones(pose_num,1)*class_mu(i,:);
    class_scatter(:,:,i) = C_center_class'*C_center_class;
end

mu_total = mean(pca_Y,1); % total mean vector
Sw = sum(class_scatter,3); % within class scatter
Pc = pca_Y-ones(size(pca_Y,1),1)*mu_total;
Stotal = Pc'*Pc; % total scatter matrix
Sb = Stotal - Sw; % between class scatter
[evec,eval] = eig(Sb, Sw);
[esort, isort] = sort(diag(eval), 'descend');
evec = evec(:,isort);
w = evec(:,1:subject_num-1);

mda_Y = pca_Y*w;

%% Split into Test and Train data 

% every 13 numbers is a new subject
subject_1 = mda_Y(1:13,:);
subject_2 = mda_Y(14:26,:);
subject_3 = mda_Y(27:39,:);
subject_4 = mda_Y(40:52,:);
subject_5 = mda_Y(53:65,:);

train_index = 1;
train_finish = 10;

% Split the data into Training
subject_1_train = data_splitter(subject_1, train_index, train_finish);
subject_2_train = data_splitter(subject_2, train_index, train_finish);
subject_3_train = data_splitter(subject_3, train_index, train_finish);
subject_4_train = data_splitter(subject_4, train_index, train_finish);
subject_5_train = data_splitter(subject_5, train_index, train_finish);

knn_train = [subject_1_train; subject_2_train; subject_3_train; subject_4_train; subject_5_train];

test_index = 11;
test_finish = 13;
% Split the data into Test
subject_1_test = data_splitter(subject_1, test_index, test_finish);
subject_2_test = data_splitter(subject_2, test_index, test_finish);
subject_3_test = data_splitter(subject_3, test_index, test_finish);
subject_4_test = data_splitter(subject_4, test_index, test_finish);
subject_5_test = data_splitter(subject_5, test_index, test_finish);

%% Apply Bayesian Decision Theory 

% Calculate the mean of each set 
subject_1_mu_train = mean(subject_1_train, 1);
subject_2_mu_train = mean(subject_2_train, 1);
subject_3_mu_train = mean(subject_3_train, 1);
subject_4_mu_train = mean(subject_4_train, 1);
subject_5_mu_train = mean(subject_5_train, 1);

% Center the data sets 10x4
subject_1_center = subject_1_train - ones(10,1)*subject_1_mu_train;
subject_2_center = subject_2_train - ones(10,1)*subject_2_mu_train;
subject_3_center = subject_3_train - ones(10,1)*subject_3_mu_train;
subject_4_center = subject_4_train - ones(10,1)*subject_4_mu_train;
subject_5_center = subject_5_train - ones(10,1)*subject_5_mu_train;

% Calculate the Covariance matrices
subject_1_cov_train = (1/10)*(subject_1_center'*subject_1_center);
subject_2_cov_train = (1/10)*(subject_2_center'*subject_2_center);
subject_3_cov_train = (1/10)*(subject_3_center'*subject_3_center);
subject_4_cov_train = (1/10)*(subject_4_center'*subject_4_center);
subject_5_cov_train = (1/10)*(subject_5_center'*subject_5_center);

% Create Gaussian Functions
subject_1_dist = @(x)-0.5*log(det(subject_1_cov_train))-0.5*((x-subject_1_mu_train)*inv(subject_1_cov_train)*(x-subject_1_mu_train)');
subject_2_dist = @(x)-0.5*log(det(subject_2_cov_train))-0.5*((x-subject_2_mu_train)*inv(subject_2_cov_train)*(x-subject_2_mu_train)');
subject_3_dist = @(x)-0.5*log(det(subject_3_cov_train))-0.5*((x-subject_3_mu_train)*inv(subject_3_cov_train)*(x-subject_3_mu_train)');
subject_4_dist = @(x)-0.5*log(det(subject_4_cov_train))-0.5*((x-subject_4_mu_train)*inv(subject_4_cov_train)*(x-subject_4_mu_train)');
subject_5_dist = @(x)-0.5*log(det(subject_5_cov_train))-0.5*((x-subject_5_mu_train)*inv(subject_5_cov_train)*(x-subject_5_mu_train)');

%% Classify test data with Gaussian classifier

subject_1_gaus = gaussian_for_MDA(subject_1_test, subject_1_dist, subject_2_dist, subject_3_dist, subject_4_dist, subject_5_dist);
subject_2_gaus = gaussian_for_MDA(subject_2_test, subject_1_dist, subject_2_dist, subject_3_dist, subject_4_dist, subject_5_dist);
subject_3_gaus = gaussian_for_MDA(subject_3_test, subject_1_dist, subject_2_dist, subject_3_dist, subject_4_dist, subject_5_dist);
subject_4_gaus = gaussian_for_MDA(subject_4_test, subject_1_dist, subject_2_dist, subject_3_dist, subject_4_dist, subject_5_dist);
subject_5_gaus = gaussian_for_MDA(subject_5_test, subject_1_dist, subject_2_dist, subject_3_dist, subject_4_dist, subject_5_dist);

%% PCA/MDA for KNN

% 68 subjects in total
nPCA = 100;
data_p = load('pose.mat');
pose = data_p.pose;

[d1, d2,pose_num,subject_num] = size(pose);
P_knn = [];

for i = 1:subject_num
    for j = 1:pose_num
        P_knn(:,(i-1)*pose_num+j) = reshape(pose(:,:,j,i), [d1*d2 1]);
    end
end


[pca_Y, U] = perform_pca(P_knn',nPCA);


% Now calculate MDA
[d1,d2]= size(pca_Y);

class_mu = zeros(subject_num, nPCA);
class_scatter = zeros(nPCA, nPCA, subject_num);

% row vectors
for i = 1:subject_num
    class_mu(i,:) = mean(pca_Y((i-1)*pose_num+1:i*pose_num,:),1);
    C_center_class = pca_Y((i-1)*pose_num+1:i*pose_num,:)-ones(pose_num,1)*class_mu(i,:);
    class_scatter(:,:,i) = C_center_class'*C_center_class;
end

mu_total = mean(pca_Y,1); % total mean vector
Sw = sum(class_scatter,3); % within class scatter
Pc = pca_Y-ones(size(pca_Y,1),1)*mu_total;
Stotal = Pc'*Pc; % total scatter matrix
Sb = Stotal - Sw; % between class scatter
[evec,eval] = eig(Sb, Sw);
[esort, isort] = sort(diag(eval), 'descend');
evec = evec(:,isort);
w = evec(:,1:subject_num-1);

mda_Y = pca_Y*w;

% Split up PCA/MDA data by class for KNN
subject = [];
for i = 1:subject_num
    counter = (i-1)*pose_num;
    for j =1:pose_num
        subject(:,j,i) = mda_Y(counter+j,:);
    end
end

%% Split data into training and test sets
train_num = 8;
test_num = pose_num - train_num;
subject_train = [];
subject_test = [];
for i = 1:subject_num
    [subject_train(:,:,i), subject_test(:,:,i)] = train_test_splitter(subject(:,:,i), train_num);
end

% Combine all training data into one variable
knn_train = [];
for i = 1:subject_num
    for j = 1:train_num
        knn_train(:,(i-1)*train_num+j) = subject_train(:,j,i);
    end
end
%% Perform KNN

classes = zeros(1, subject_num*train_num);
for i = 1:size(classes,2)
    classes(1,i) = floor((i-1) / train_num) + 1;
end


k=5;
correct_total = 0;
wrong_total = 0;
for i = 1:subject_num
    correct_class = 0;
    wrong_class = 0;
    for j = 1:test_num
        [classification] = knn(k, knn_train, subject_test(:,j,i), classes);
        if classification == i
            correct_class = correct_class + 1;
            correct_total = correct_total + 1;
        else
            wrong_class = wrong_class + 1;
            wrong_total = wrong_total + 1;
        end
    end
    fprintf('Class %d Accuracies\n\n', i);
    fprintf('%d correctly classified\n', correct_class);
    fprintf('%d incorrectly classified\n', wrong_class);
end
fprintf('k-value is: %d\n', k);
fprintf('Total Results\n\n');
fprintf('%d correctly classified\n', correct_total);
fprintf('%d incorrectly classified\n', wrong_total);

%% Functions

function [Y,U] = perform_pca(x, num_pca)
    % calculate dimension
    [d1,d2] = size(x);
    % center the data
    X_mean = mean(x,1);
    X_centered = x - ones(d1,1)*X_mean;
    
    % solve for eigenvalues 
    % we do transpose because we have more columns than rows
    [U, Sigma, ~] = svd(X_centered', 'econ');
    esort = diag(Sigma);
    
    % project to nPCA-dimensional space
    Y = x*U(:,1:num_pca);
    
    figure;
    plot(esort, '.', 'Markersize', 20);
    grid;
   
end

function [x_split] = data_splitter(x_centered, start, finish)
    x_split = x_centered(start:finish, :);
end

function [decision, I] = gaussian_for_MDA(subject_test, subject_1_dist, subject_2_dist, subject_3_dist, subject_4_dist, subject_5_dist)
    [c1, c2] = size(subject_test);
    label = zeros(1,5);
    
    % test each gaussian distribution function on the given test set
    for j = 1:c1
        label(1) = subject_1_dist(subject_test(j,:));
        label(2) = subject_2_dist(subject_test(j,:));
        label(3) = subject_3_dist(subject_test(j,:));
        label(4) = subject_4_dist(subject_test(j,:));
        label(5) = subject_5_dist(subject_test(j,:));
        [decision, I] = max(label);
        fprintf('Best Class Gaussian Function: %d\n', I);
    end
end

% split data into the given size for training and test
function [train, test] = train_test_splitter(A, number)
    train = A(:,1:number);
    test = A(:,number+1:end);
end

% calculate k-NN 
function [classification] = knn(k,train,test, classes)
    dist = vecnorm(train-test);
    [~,I] = mink(dist, k);
    classification = mode(classes(:,I),2);
end


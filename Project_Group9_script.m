% QUANTITATIVE MODELS FOR DATA SCIENCE - GROUP 9 PROJECT
% GROUP MEMBERS:    Matteo Spadaccia (277141), Valerio Romano Cadura (264851),
%                   Dan Mark Tolod (276181), Filippo Castello (not available)


% Data importing
abalone_data = readtable('Abalone_data.xlsx')
% Data saving 
save('Abalone_data.mat','abalone_data')

% Storing of each regressors' and response variable's arrays
L = table2array(abalone_data(:,1));
D = table2array(abalone_data(:,2));
H = table2array(abalone_data(:,3));
W = table2array(abalone_data(:,4));
R = table2array(abalone_data(:,5));

% Random seed definition to obtain always the same permutations and results
rng('default')
s = rng;
% (comment if needed)

% Obtaining datasets sizes
[m,n] = size(abalone_data);

% Defining partitions' sizes
partition = 0.90;

% Dividing the dataset into a train-part and a test-part
positions = randperm(m); % returns a row vector containing a random permutation of the integers from 1 to m
Data_train = abalone_data(positions(1:round(partition*m)),:);
Data_test = abalone_data(positions(round(partition*m)+1:end),:);

% Preparing the variables and separating data into individual regressors and response variable arrays
X_train = Data_train(:,1:4);
X_test = Data_test(:,1:4);
y_train = Data_train(:,5);
y_test = Data_test(:,5);
LDHW_train = table2array(X_train);
LDHW_test = table2array(X_test);
L_train = table2array(X_train(:,1));
L_test = table2array(X_test(:,1));
D_train = table2array(X_train(:,2));
D_test = table2array(X_test(:,2));
H_train = table2array(X_train(:,3));
H_test = table2array(X_test(:,3));
W_train = table2array(X_train(:,4));
W_test = table2array(X_test(:,4));
R_train = table2array(y_train);
R_test = table2array(y_test);
m_train = length(R_train);
m_test = length(R_test);
regressors = ['Length';'Depth ';'Height';'Weight'];

% Plotting the response variable vs each of the regressors
hold off
scatter(L_train,R_train,10,'filled','green')
ylabel('R')
xlabel('L')
title('Rings vs. Length')
scatter(D_train,R_train,10,'filled','blue')
ylabel('R')
xlabel('D')
title('Rings vs. Depth')
scatter(H_train, R_train,10,'filled','cyan')
ylabel('R')
xlabel('H')
title('Rings vs. Height')
scatter(W_train, R_train,10,'filled','black')
ylabel('R')
xlabel('W')
title('Rings vs. Weight')

% Plotting the response variable vs each of the regressors (single graph)
scatter(L_train, R_train,10,'filled','green')
hold on
scatter(D_train, R_train,10,'filled','blue')
scatter(H_train, R_train,10,'filled','cyan')
scatter(W_train, R_train,10,'filled','black')
ylabel('R')
xlabel('Regressors')
title('Rings vs. Regressors')
legend('= L','= D', '= H', '= W','location','best')
hold off

% Plotting the response variable vs each of the regressors (single figure)
figure()
for i = 1:4
    subplot(2,2,i)
    scatter(LDHW_train(:,i),R_train,'b.')
    title(['Rings vs. ' regressors(i,1:6)])
    xlabel(regressors(i,1))
    ylabel('R')
end
figure()

% Applying the LSM to the pair R-L
X_L = [ones(m_train,1), L_train];
XtX_L = X_L' * X_L;
XtX_inv_L = inv(XtX_L);
Xty_L = X_L' * R_train;
beta_hat_L = XtX_inv_L * Xty_L;
beta_0_L = beta_hat_L(1);
beta_1_L = beta_hat_L(2);
l_reg_L = @(x) beta_0_L + beta_1_L * x;

% Applying the LSM to the pair R-D
X_D = [ones(m_train,1), D_train];
XtX_D = X_D' * X_D;
XtX_inv_D = inv(XtX_D);
Xty_D = X_D' * R_train;
beta_hat_D = XtX_inv_D * Xty_D;
beta_0_D = beta_hat_D(1);
beta_1_D = beta_hat_D(2);
l_reg_D = @(x) beta_0_D + beta_1_D * x;

% Applying the LSM to the pair R-H
X_H = [ones(m_train,1), H_train];
XtX_H = X_H' * X_H;
XtX_inv_H = inv(XtX_H);
Xty_H = X_H' * R_train;
beta_hat_H = XtX_inv_H * Xty_H;
beta_0_H = beta_hat_H(1);
beta_1_H = beta_hat_H(2);
l_reg_H = @(x) beta_0_H + beta_1_H * x;

% Applying the LSM to the pair R-W
X_W = [ones(m_train,1), W_train];
XtX_W = X_W' * X_W;
XtX_inv_W = inv(XtX_W);
Xty_W = X_W' * R_train;
beta_hat_W = XtX_inv_W * Xty_W;
beta_0_W = beta_hat_W(1);
beta_1_W = beta_hat_W(2);
l_reg_W = @(x) beta_0_W + beta_1_W * x;

% Plotting the regression line of the pair R-L
scatter(L_train,R_train,'green')
hold on
fplot(l_reg_L, [0 0.8],'black')
ylabel('R')
xlabel('L')
title('R-L regression line')
legend('Data','Fitted line','Location','best')
hold off
beta_0_L
beta_1_L

% Plotting the regression line of the pair R-D
scatter(D_train,R_train,'blue')
hold on
fplot(l_reg_D, [0 0.8],'black')
ylabel('R')
xlabel('D')
title('R-D regression line')
legend('Data','Fitted line','Location','best')
hold off
beta_0_D
beta_1_D

% Plotting the regression line of the pair R-H
scatter(H_train,R_train,'cyan')
hold on
fplot(l_reg_H, [0 0.3],'black')
ylabel('R')
xlabel('H')
title('R-H regression line')
legend('Data','Fitted line','Location','best')
hold off
beta_0_H
beta_1_H

% Plotting the regression line of the pair R-W
scatter(W_train,R_train,'black')
hold on
fplot(l_reg_W, [0 2.5],'black')
ylabel('R')
xlabel('W')
title('R-W regression line')
legend('Data','Fitted line','Location','best')
hold off
beta_0_W
beta_1_W

% Computing the RMSE of the pair R-L
y_hat_L = X_L * beta_hat_L;
epsilon_hat_L = R_train - y_hat_L;
epsilon_hat_squared_L = epsilon_hat_L .^2;
RMSE_L_train = sqrt(mean(epsilon_hat_squared_L))

% Computing the RMSE of the pair R-D
y_hat_D = X_D * beta_hat_D;
epsilon_hat_D = R_train - y_hat_D;
epsilon_hat_squared_D = epsilon_hat_D .^2;
RMSE_D_train = sqrt(mean(epsilon_hat_squared_D))

% Computing the RMSE of the pair R-H
y_hat_H = X_H * beta_hat_H;
epsilon_hat_H = R_train - y_hat_H;
epsilon_hat_squared_H = epsilon_hat_H .^2;
RMSE_H_train = sqrt(mean(epsilon_hat_squared_H))

% Computing the RMSE of the pair R-W
y_hat_W = X_W * beta_hat_W;
epsilon_hat_W = R_train - y_hat_W;
epsilon_hat_squared_W = epsilon_hat_W .^2;
RMSE_W_train = sqrt(mean(epsilon_hat_squared_W))

% Computing the test RMSE of the pair R-L
y_pred_L = l_reg_L(L_test);
SSR_L = sum((y_pred_L - R_test) .^2);
MSE_L = SSR_L / m_test;
RMSE_L_test = sqrt(MSE_L)

% Computing the test RMSE of the pair R-D
y_pred_D = l_reg_D(D_test);
SSR_D = sum((y_pred_D - R_test) .^2);
MSE_D = SSR_D / m_test;
RMSE_D_test = sqrt(MSE_D)

% Computing the test RMSE of the pair R-H
y_pred_H = l_reg_H(H_test);
SSR_H = sum((y_pred_H - R_test) .^2);
MSE_H = SSR_H / m_test;
RMSE_H_test = sqrt(MSE_H)

% Computing the test RMSE of the pair R-W
y_pred_W = l_reg_W(W_test);
SSR_W = sum((y_pred_W - R_test) .^2);
MSE_W = SSR_W / m_test;
RMSE_W_test = sqrt(MSE_W)

% Computing the multivariate linear regression with all the regressors
X_all = [ones(m_train,1) LDHW_train];
beta_hat_all = X_all \ R_train; % \ solves the linear system if possible if not possible it solves it in l-s-sense
beta_0_all = beta_hat_all(1)
beta_1L_all = beta_hat_all(2)
beta_2D_all = beta_hat_all(3)
beta_3H_all = beta_hat_all(4)
beta_4W_all = beta_hat_all(5)

% Computing the RMSE of the all-regressors multivariate regression
y_hat_all = X_all * beta_hat_all;
epsilon_hat_all = R_train - y_hat_all;
epsilon_hat_squared_all = epsilon_hat_all .^2;
RMSE_all = sqrt(mean(epsilon_hat_squared_all))

% Computing the test RMSE of the all-regressors multivariate regression
y_pred_all = beta_0_all + beta_1L_all * L_test + beta_2D_all * D_test + beta_3H_all * H_test + beta_4W_all * W_test;
SSR_all = sum((y_pred_all - R_test) .^2);
MSE_all = SSR_all / m_test;
RMSE_all_test = sqrt(MSE_all)

% Computing all the regressors couples' linear regressions
two_reg_data = Dictionary;
for i = 1:4
    for j = i+1:4
        % Computing the multivariate linear regression with the two selected regressors
        X_temp = [ones(m_train,1) LDHW_train(:,[i j])];
        beta_hat_temp = X_temp \ R_train; % \ solves the linear system if possible if not possible it solves it in l-s-sense
        beta_0_temp = beta_hat_temp(1);
        beta_1_temp = beta_hat_temp(2);
        beta_2_temp = beta_hat_temp(3);

        % Computing the RMSE of the multivariate regression with the two selected regressors
        y_hat_temp = X_temp * beta_hat_temp;
        epsilon_hat_temp = R_train - y_hat_temp;
        epsilon_hat_squared_temp = epsilon_hat_temp .^2;
        RMSE_temp = sqrt(mean(epsilon_hat_squared_temp));

        % Computing the test RMSE of the multivariate regression with the two selected regressors
        y_pred_temp = beta_0_temp + beta_1_temp * LDHW_test(:,i) + beta_2_temp * LDHW_test(:,j);
        SSR_temp = sum((y_pred_temp - R_test) .^2);
        MSE_temp = SSR_temp / m_test;
        RMSE_temp_test = sqrt(MSE_temp);

        % Storing data in a dictionary
        two_reg_data(strcat(regressors(i,1:6),'-',regressors(j,1:6))) = [RMSE_temp;RMSE_temp_test;beta_hat_temp];
    end
end

% Computing all the regressors triplets' linear regressions
three_reg_data = Dictionary;
for i = 1:4
    % Computing the multivariate linear regression with the three selected regressors
    X_temp = [ones(m_train,1) LDHW_train(:,cat(2,1:(i-1),(i+1):4))];
    beta_hat_temp = X_temp \ R_train; % \ solves the linear system if possible if not possible it solves it in l-s-sense
    beta_0_temp = beta_hat_temp(1);
    beta_1_temp = beta_hat_temp(2);
    beta_2_temp = beta_hat_temp(3);
    beta_3_temp = beta_hat_temp(4);

    % Computing the RMSE of the multivariate regression with the three selected regressors
    y_hat_temp = X_temp * beta_hat_temp;
    epsilon_hat_temp = R_train - y_hat_temp;
    epsilon_hat_squared_temp = epsilon_hat_temp .^2;
    RMSE_temp = sqrt(mean(epsilon_hat_squared_temp));

    % Computing the test RMSE of the multivariate regression with the three selected regressors
    y_pred_temp = (beta_0_temp + beta_hat_temp(2:4)' * LDHW_test(:,cat(2,1:(i-1),(i+1):4))')';
    SSR_temp = sum((y_pred_temp - R_test) .^2);
    MSE_temp = SSR_temp / m_test;
    RMSE_temp_test = sqrt(MSE_temp);

    % Storing data in a dictionary
    three_reg_data(strcat('allBut',regressors(i,1:6))) = [RMSE_temp;RMSE_temp_test;beta_hat_temp];
end

% Displaying results of all the regressors couples' linear regressions
reg_couples_temp = two_reg_data.Keys;
for i = 1:length(two_reg_data.Keys)
    two_reg_data_sel_temp = two_reg_data(char(reg_couples_temp(i)));
    disp(strcat('RMSE_',char(reg_couples_temp(i)),'_train: ',num2str(two_reg_data_sel_temp(1))))
end

% Displaying results of all the regressors triplets' linear regressions
reg_couples_temp = three_reg_data.Keys;
for i = 1:length(three_reg_data.Keys)
    three_reg_data_sel_temp = three_reg_data(char(reg_couples_temp(i)));
    disp(strcat('RMSE_',char(reg_couples_temp(i)),'_train: ',num2str(three_reg_data_sel_temp(1))))
end

% displaying the test RMSE of the model considering all the regressors but Depth
allButDepth_reg_data = three_reg_data('allButDepth');
disp(strcat('RMSE_allButDepth_test: ',num2str(allButDepth_reg_data(2))))

% displaying the coefficients of the model considering all the regressors but Depth
beta_0_allButDepth = allButDepth_reg_data(3)
beta_1L_allButDepth = allButDepth_reg_data(4)
beta_2H_allButDepth = allButDepth_reg_data(5)
beta_3W_allButDepth = allButDepth_reg_data(6)

% Computing all the polinomial models for the various regressors and choosing the best one for each by RMSE
DegRMSEs_poly = zeros(1,8);
for i=1:4
    RMSEs_temp = zeros(1,5);
    Xiy = [X_train(:,i) y_train];
    sorted_Xiy = sortrows(Xiy,1);
    X_train_i = table2array(sorted_Xiy(:,1));
    y_train_i = table2array(sorted_Xiy(:,2));
    for j=1:5
        if j == 1 
            poly_temp = [ones(m_train,1) X_train_i];
        elseif j == 2
            poly_temp = [ones(m_train,1) X_train_i X_train_i.^2];
        elseif j == 3
            poly_temp = [ones(m_train,1) X_train_i X_train_i.^2 X_train_i.^3];
        elseif j == 4
            poly_temp = [ones(m_train,1) X_train_i X_train_i.^2 X_train_i.^3 X_train_i.^4];
        elseif j == 5
            poly_temp = [ones(m_train,1) X_train_i X_train_i.^2 X_train_i.^3 X_train_i.^4 X_train_i.^5];
        end
        beta_hat_temp = poly_temp \ y_train_i;
        y_hat_temp = poly_temp * beta_hat_temp;
        epsilon_hat_temp = y_train_i - y_hat_temp;
        SSR_temp = sum(epsilon_hat_temp .^2);
        RMSE_temp = sqrt(SSR_temp / m_train);
        RMSEs_temp(j) = RMSE_temp;    
    end

    % Considering the best polinomial model for each regressor by RMSEs
    [bestRMSE_temp, bestRMSE_idx_temp] = min(RMSEs_temp);
    DegRMSEs_poly(i*2-1) = bestRMSE_idx_temp;
    DegRMSEs_poly(i*2) = bestRMSE_temp;

    % Displaying the best polinomial model for each regressor
    if bestRMSE_idx_temp == 1 
        poly_temp = [ones(m_train,1) X_train_i];
    elseif bestRMSE_idx_temp == 2
        poly_temp = [ones(m_train,1) X_train_i X_train_i.^2];
    elseif bestRMSE_idx_temp == 3
        poly_temp = [ones(m_train,1) X_train_i X_train_i.^2 X_train_i.^3];
    elseif bestRMSE_idx_temp == 4
        poly_temp = [ones(m_train,1) X_train_i X_train_i.^2 X_train_i.^3 X_train_i.^4];
    elseif bestRMSE_idx_temp == 5
        poly_temp = [ones(m_train,1) X_train_i X_train_i.^2 X_train_i.^3 X_train_i.^4 X_train_i.^5];
    end
    beta_hat_temp = poly_temp \ y_train_i;
    y_hat_temp = poly_temp * beta_hat_temp;
    figure()
    scatter(X_train_i,y_train_i,'b')
    title(strcat('R-',regressors(i),' regression polinomial'))
    xlabel(regressors(i,1))
    ylabel('R')
    hold on
    plot(X_train_i,y_hat_temp,'black')
    legend('Data','Fitted curve','Location','best')
    hold off
    
    % Computing the best polinomial model's test RMSE
    if bestRMSE_idx_temp == 1 
        poly_test_temp = [ones(m_test,1) LDHW_test(:,i)];
    elseif bestRMSE_idx_temp == 2
        poly_test_temp = [ones(m_test,1) LDHW_test(:,i) LDHW_test(:,i).^2];
    elseif bestRMSE_idx_temp == 3
        poly_test_temp = [ones(m_test,1) LDHW_test(:,i) LDHW_test(:,i).^2 LDHW_test(:,i).^3];
    elseif bestRMSE_idx_temp == 4
        poly_test_temp = [ones(m_test,1) LDHW_test(:,i) LDHW_test(:,i).^2 LDHW_test(:,i).^3 LDHW_test(:,i).^4];
    elseif bestRMSE_idx_temp == 5
        poly_test_temp = [ones(m_test,1) LDHW_test(:,i) LDHW_test(:,i).^2 LDHW_test(:,i).^3 LDHW_test(:,i).^4 LDHW_test(:,i).^5];
    end
    y_hat_test_temp = poly_test_temp * beta_hat_temp;
    epsilon_hat_test_temp = R_test - y_hat_test_temp;
    SSR_test_temp = sum(epsilon_hat_test_temp .^2);
    RMSE_test_temp = sqrt(SSR_test_temp / m_test);

    % Displaying information
    disp(strcat('Degree: ',num2str(DegRMSEs_poly(i*2-1)),'  RMSE: ',num2str(DegRMSEs_poly(i*2)),'  RMSE_test: ',num2str(RMSE_test_temp)))
    disp(strcat('Beta_hat: ',num2str(beta_hat_temp')))
end

% Computing the multivariate polynomial model and its RMSE
X_multipoly = [ones(m_train,1) LDHW_train(:,1) LDHW_train(:,1).^2 LDHW_train(:,2) LDHW_train(:,2).^2 LDHW_train(:,3) LDHW_train(:,3).^2 LDHW_train(:,4) LDHW_train(:,4).^2];
beta_hat_multipoly = X_multipoly \ R_train;
y_hat_multipoly = X_multipoly * beta_hat_multipoly;
epsilon_hat_multipoly = R_train - y_hat_multipoly;
epsilon_hat_multipoly_squared = epsilon_hat_multipoly .^2;
RMSE_multipoly_train = sqrt(mean(epsilon_hat_multipoly_squared))

% Computing the multivariate polynomial model's test RMSE
X_multipoly_test = [ones(m_test,1) LDHW_test(:,1) LDHW_test(:,1).^2 LDHW_test(:,2) LDHW_test(:,2).^2 LDHW_test(:,3) LDHW_test(:,3).^2 LDHW_test(:,4) LDHW_test(:,4).^2];
y_pred_multipoly = X_multipoly_test * beta_hat_multipoly;
epsilon_pred_multipoly = R_test - y_pred_multipoly;
epsilon_pred_multipoly_squared = epsilon_pred_multipoly.^2;
RMSE_multipoly_test = sqrt(mean(epsilon_pred_multipoly_squared))

% computing the eigenvalues and eigenvectors of the correlation matrix
LDHW = table2array(abalone_data(:,1:4));
corr_mat = corrcoef(LDHW)
[corr_eigenvectors,corr_eigenvalues] = eig(corr_mat,'vector');

% displaying the eigenvalues and eigenvectors of the correlation matrix
for i = 1:length(corr_eigenvalues)
    display(strcat('Eigenvalue: ',num2str(corr_eigenvalues(i)),', Eigenvector:'))
    for j = 1:length(corr_eigenvectors)
        display(num2str(corr_eigenvectors(j,i)))
    end
end

% Computing the covariance matrix and the variance proportion of each component
covar_mat = cov(LDHW)
[covar_eigenvectors,covar_eigenvalues] = eig(covar_mat,'vector');
total_variance = sum(covar_eigenvalues);
sorted_covar_eigenvalues = sort(covar_eigenvalues,'descend');
var_proportion = sorted_covar_eigenvalues ./ total_variance  

% Reverting the eigenvectors order as we sorted the eigenvalues in descending order
coeff = zeros(4,4);
for i = 1:4
    coeff(:,i) = covar_eigenvectors(:,end-i+1);
end

% Plotting the scree plot
plot(1:4,var_proportion,'.','MarkerSize',16)
xlim([0 5]);
xlabel('i-th principal component')
ylabel('Proportion of variance')
title('Scree plot')

% Plotting the cumulative proportion
cumulative_var_proportion = cumsum(var_proportion)                                
figure()
plot(0:4,[0; cumulative_var_proportion], 'k-')
hold on
plot(1:4,cumulative_var_proportion,'b.','MarkerSize',15)
xlim([0 4+1])
ylim([0 1.05])
plot(linspace(0,4+1,100),cumulative_var_proportion(1)*ones(100),'k--')
xlabel('i-th principal component')
ylabel('Aggr. prop. of total variance')
title('Cumulative proportion')

% Displaying the first two principal components' loadings
disp("1st principal component's loading:")
disp(num2str(coeff(:,1)'))
disp("2nd principal component's loading:")
disp(num2str(coeff(:,2)'))

% Computing the scores by the pca function
[coeff_pca,scores_pca,latent_pca] = pca(LDHW);

% Plotting the scores in the plane generated by the first two eigenvectors
figure()
plot(scores_pca(:,1), scores_pca(:,2),'kx')
xmax = max(abs(scores_pca(:,1))) + 0.01;
ymax = max(abs(scores_pca(:,2))) + 0.01;
xlim([-xmax xmax])
ylim([-ymax ymax])
hold on
x = linspace(-xmax, xmax, m); 
y = linspace(-ymax, ymax, m);
plot(x,zeros(m),'b--')
plot(zeros(m),y,'b--')
xlabel('y_1')
ylabel('y_2')
title("First two principal components' scores")
%% Housekeeping
clear;
clc;
close all;
%% 
% 1) Generate a normal series of errors for 100 observations. Using beta1=10 and beta2=2.5,
% generate Y (hint: X could be a uniform distributed series, but pay attention about the
% stochastic generation of X ...). Show the plot of errors. Show separately the plots of X and Y.


format long;
% Setting size and parameters
N = 100;
beta1 = 10; beta2 = 2.5;
B = [beta1; beta2;];

% Generate errors
e = randn(N,1);

% Generate X and Y

%fix seed
rng(0)
x = rand(N,1);
X = [ones(N,1) x];
Y = X*B + e;

% Plot of errors
figure(1);
filename1 = "errors_plot.fig";
scatter(e(:,1),Y,'filled');
savefig(filename1);

% Plots of X and Y
figure(2);
filename2 = "scatter_plot.fig";
scatter(X(:,2),Y,'filled');
savefig(filename2);

%% 
% 2) Estimate the OLS regression, showing the estimates beta1 and beta2, the standard errors,
% confidence intervals, t-statistics at 5%, p-value, R-Squared, and F-statistic. Show the plot of
% residuals. Are they different from errors? Comment on results.

% make OLS regressions
% making regression to find the ESTIMATED B

Bhat = inv(X'*X)*(X'*Y);
Beta1 = Bhat(1,:)
Beta2 = Bhat(2,:)

% the ESTIMATED Y

Yhat = X * Bhat;

% residuals

r = Y - Yhat;
figure(3);
filename3 = "Plot_of_Residuals";
scatter(r,Y,'filled');
savefig(filename3);

%Standard Errors
% the number of regressors including the constant
k = size(X,2);

% Variance of residuals
S = (r'*r)/(N-k);

% Covariance of the estimated B hat
V = S.*inv(X'*X);

% Standard errors
SE = diag(V).^0.5;

% Confidence intervals
alpha = 0.05; % significance level
fprintf("The confidence intervals under the significance level of %f",alpha)
[CI_lower,CI_upper] = CI_calculator(Bhat,SE,alpha,N-k);
CI = [CI_lower,CI_upper]

% T-statistic, p-value at 5% level

% SS calculation
[TSS ESS RSS] = SS_calculator(Y,Yhat,r);
t_statistic = t_calculator(Bhat,SE);
p_value = p_calculator(t_statistic,N-k)

%R-squared and adjusted R-squared
R2 = ESS/TSS
R2adj = 1 - (RSS/(N-k))/(TSS/(N-1))

%% 
% 3) Create an m.file for estimating a linear regression where you should include: estimation of beta coeffecients, 
% standard errors, calculation of t statistics (at 5%), pvalue, confidence intervals (at 5%), R-squared, Adjusted 
% R-squared, and F statistic.

% Open function OLS.m and add to path

%% 
% 4) Split the sample in two parts: in the first part you generate normal errors with a standard deviation of 1 and in
% the second part you generate normal errors with a standard deviation of 2. Re-estimate OLS regression as in 2).

rng(1);
mu=0;
sigma=1;
R = normrnd(mu,sigma)
error_1sd = normrnd(0,1,50,1);
error_2sd = normrnd(0,2,50,1);
new_error = [error_1sd; error_2sd];
y_new = X*B + new_error;
beta_hats = OLS(y_new, X);
yhats = X*beta_hats;
resid = yhats - y_new;

%% 
% 5) Create a Monte Carlo experiment with 1000 replications, where you generate 100
% observations of X and Y using normal residuals with 0 mean and 1 standard deviation, and
% with beta1=5 and beta2=0.7. Show how beta1 and beta2 evolve in replications (using a
% histogram). Do they satisfy the unbiasedness property?

% Define the true population parameters
beta11=5;
beta22=0.7;

% Define the sample size T
T=1000;

% Define the number of samples S
S=100;

% Store the estimated beta in a vector
beta_hat_mtx = zeros(S,2);

% Loop the samples

for i=1:S

% Generate x using the normal-distribution random number generator
x = 4*randn(T,1);

% Generate the errors in the population
e = randn(T,1);

% Generate y
y = beta11 + beta22*x + e;

% Generate beta1 hat and beta2 hat
X = [ones(T,1) x];
beta_hat = (inv((X'*X)))*X'*y;

% Save the betas to a matrix
% beta_hat_mtx(i,1) = beta_hat(1);
% beta_hat_mtx(i,2) = beta_hat(2);
% Try this
beta_hat_mtx(i,:) = beta_hat;
end

% Draw histograms of the betas
figure(4);
hist(beta_hat_mtx(:,2),20)
xlabel('Beta2 hat')
ylabel('frequency');
figure(5);
hist(beta_hat_mtx(:,1),20)
xlabel('Beta1 hat')
ylabel('frequency')

% Don't satisfy the unbiasedness property because as you can see both histograms aren't normally distributed.

%% 
% 6) Repeat estimation of 6) ignoring the constant, beta1. Does beta2 satisfy the unbiasedness
% property? Why?
% Define the true population parameters
%beta11=5;
beta22=0.7;
% Define the sample size T
T=1000;
% Define the number of samples S
S=100;
% Store the estimated beta in a vector
beta_hat_mtx = zeros(S,2);
% Changed y to and x to a and X to A
% Loop the samples
for i=1:S
% Generate x using the normal-distribution random number generator
a = 4*randn(T,1);
% Generate the errors in the population
e = randn(T,1);
% Generate y
D = beta22*a + e;
% Generate beta1 hat and beta2 hat
A = [ones(T,1) a];
beta_hat = (inv((A'*A)))*A'*D;
% Save the betas to a matrix
% beta_hat_mtx(i,1) = beta_hat(1);
% beta_hat_mtx(i,2) = beta_hat(2);
% Try this
beta_hat_mtx(i,:) = beta_hat;
end
% Draw histograms of the betas
%changed hist to histogram
figure(7);
histogram(beta_hat_mtx(:,2),20)
xlabel('Beta2 hat')
ylabel('frequency');

% Beta 2 is closer to satisfying unbiasedness criteria but it still doesn’t appear to be normally distributed.

%% 
% 7) Keep features of point 5), generate a new variable Z which is correlated with X. Show how
% the estimation of Betas change. Comment. (In this case, generate two Z, one with a correla-
% tion of 10-20% with X and the other one with a correlation of 90% with X). 

%Monte Carlo experiment with correlation of 10%

%Parameters
Z10 = 5;
B1Z10 = 0.7;
B2Z10 = 2;
BZ10 = [Z10;B1Z10;B2Z10];
rZ10 =[1 0.10; 0.10 1];

%Sample Size
NZ10 = 100;

%Itterations
IZ10= 1000;

%Store for betas we estimate
BolsZ10 = zeros(IZ10,3);

%Loop the samples
for i=1:IZ10
L = chol(rZ10);
xZ0= 69*rand(NZ10,2);
xZ10 = [ones(NZ10,1) xZ0*L];
Z10 = xZ10(:,2);
eZ10 = randn(NZ10,1);
A10 = corr(xZ10,Z10);
yZ10 = xZ10 *BZ10 + eZ10;
BhatZ10 = inv(xZ10'*xZ10)*(xZ10'*yZ10);
%Save betas to a matrix
BolsZ10(i,:) = BhatZ10;
end

%Histogram of Bols Monte Carlo
figure(7)
hist(BolsZ10(:,1),20)
title('Bhat1 Correlation 10%')
xlabel('Bhat1')
ylabel('Frequency')
savefig("B1 Histogram Correlation 10%")
figure(8)
hist(BolsZ10(:,2),20)
title('Bhat2 Correlation 10%')
xlabel('Bhat2')
ylabel('Frequency')
savefig("B2 Histogram Correlation 10%")
figure(9)
hist(BolsZ10(:,3),20)
title('Bhat3 Correlation 10%')
xlabel('Bhat3')
ylabel('Frequency')
savefig("B3 Histogram Correlation 10%")

%Monte Carlo experiment with correlation of 90%

%Parameters
Z90 = 5;
B1Z90 = 0.7;
B2Z90 = 2;
BZ90 = [Z90;B1Z90;B2Z90];
rZ90 =[1 0.90; 0.90 1];

%Sample Size
NZ90 = 100;

%Itterations
IZ90= 1000;

%Store for betas we estimate
BolsZ90 = zeros(IZ90,3);

%Loop the samples
for i=1:IZ90
L = chol(rZ90);
xZ090= 69*rand(NZ90,2);
xZ90 = [ones(NZ90,1) xZ090*L];
Z90 = xZ90(:,2);
eZ90 = randn(NZ90,1);
A90 = corr(xZ10,Z90);
yZ90 = xZ90 *BZ90 + eZ90;
BhatZ90 = inv(xZ90'*xZ90)*(xZ90'*yZ90);

%Save betas to a matrix
BolsZ90(i,:) = BhatZ90;
end

%Histogram of B_ols_MC
figure(10)
hist(BolsZ90(:,1),20)
title('Bhat1 Correlation 90%')
xlabel('Bhat1')
ylabel('Frequency')
savefig("B1 Histogram Correlation 90%")
figure(11)
hist(BolsZ90(:,2),20)
title('Bhat2 Correlation 90%')
xlabel('Bhat2')
ylabel('Frequency')
savefig("B2 Histogram Correlation 90%")
figure(12)
hist(BolsZ90(:,3),20)
title('Bhat3 Correlation 90%')
xlabel('Bhat3')
ylabel('Frequency')
savefig("B3 Histogram Correlation 90%")
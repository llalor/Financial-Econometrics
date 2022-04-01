function [betas, SE, t_statistic, p_value, CI, R2, R2adj,F_statistic] = OLS(Y,X)

% OLS Regression Function

%initialize function variables

u = X;
v = Y;
N = size(u,1);
SST = zeros(1,1);
SSR = zeros(1,1);
SSE = zeros(1,1);

%if the input matrix doesnt contain a column of ones, hstack one onto the
%front of the matrix

one = ones(1,1); %scalar equal to 1
if u(:,1) == one % if all rows of col 1 are equal to one, do nothing
u = u;
else
u = [ones(N,1), u]; %if not, stack a col of ones
end
k = size(u,2); %number of cols including ones

%beta estimates
betas = inv(u'*u)*(u'*v);

%Sum of Squares Calculations
vhat = u*betas;
vm = mean(v);
resid = v - vhat;
for ii=1:N %loop for SSE
SSEi = (vhat(ii,1)-vm)^2;
SSE = SSEi+ SSE;
end
SSR = resid'*resid; %defn of SSR is u'u
SST = SSE + SSR; %total = explained + unexplained

%R2 and adjR2
R2 = SSE/SST;
rnum = (N-1)/(N-k);
rdenom = 1 - R2;
adjR2 = 1 - rnum*rdenom;

%standard errors

%Var(resid)
sigma_res = SSR/(N-k);

%Var-cov of bhats
var = sigma_res.*(inv(X'*X));

%standard errors
se = diag(var).^0.5;

%t-stat & pvalue
t = betas./se;
pval = 2*(1-tcdf(abs(t),N-k));

%f-stat for overall significance
fnum = (R2/(k-1));
fdenom = (1-R2)/(N-k);
F = fnum/fdenom;

%conf interval @ 5%
tcrit=tinv(1-0.05/2,N-k);% critical value of t distribution at 5% @ N-k df
upper = betas + se*tcrit;
lower = betas - se*tcrit;
conf = [lower, upper];
end
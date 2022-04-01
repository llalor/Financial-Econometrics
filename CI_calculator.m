function [CI_lower,CI_upper] = CI_calculator(Bhat,SE,alpha,df)
%CI_CALCULATOR Summary of this function goes here

t_critical = tinv(1-alpha/2,df);

CI_lower = Bhat - SE.*t_critical;
CI_upper = Bhat + SE.*t_critical;

end


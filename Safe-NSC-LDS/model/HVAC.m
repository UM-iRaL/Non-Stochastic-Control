function [A_dis, B_dis, Ed_dis, Ec_dis] = HVAC()
% HVAC model
% state: [x]'
% input: [u]'

v = 100;
zeta = 6;
theta0 = 30;
pi = 1.5;
dt = 60;

% xdot = 1/(v*zeta) * (theta0 - x) - 1/v*u + 1/v*pi + 1/v*wt
%      = -1/(v*zeta)*x - 1/v*u + 1/(v*zeta)*theta0 + 1/v*pi + 1/v*wt
A = -1/(v*zeta);
B = - 1/v;
Ed = 1/v;
Ec = 1/(v*zeta)*theta0 + 1/v*pi;

sysd = c2d(ss(A,B,eye(1),zeros(1,1)),dt,'zoh');
A_dis = sysd.A; B_dis = sysd.B;

sysd = c2d(ss(A,Ed,eye(1),zeros(1,1)),dt,'zoh');
Ed_dis = sysd.B;

sysd = c2d(ss(A,Ec,eye(1),zeros(1,1)),dt,'zoh');
Ec_dis = sysd.B;

end


function [A, B] = UAV(g, m)
% LTI UAV model
% state: [x y z xdot ydot zdot]'
% input: [thera phi f_t]'

A = [0 0 0 1 0 0;
     0 0 0 0 1 0;
     0 0 0 0 0 1;
     0 0 0 0 0 0;
     0 0 0 0 0 0;
     0 0 0 0 0 0];
 
B = [0 0 0;
     0 0 0;
     0 0 0;
     -g 0 0;
     0 g 0;
     0 0 1/m];
 
sysd = c2d(ss(A,B,eye(6),zeros(6,3)),0.1,'zoh');
A = sysd.A; B = sysd.B;
end


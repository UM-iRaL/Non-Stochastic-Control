clear all;
close all;



%% Computation time of Safe-OCO
disturbance = ["Gaussian" "Uniform" "Beta" "Gamma" "Exponential" "Weibull"];
noise_num = length(disturbance);
T = 200;
dx = 1; 
du = 1;
time = zeros(noise_num*T, 1);
A = [0.9];
B = [0.6];

for i = 1:noise_num
    filename = disturbance(i) + "-OPT.mat";
    load(filename);
    disturbance(i)
    Lx = double(Lx);
    lx = double(lx);
    Lu = double(Lu);
    lu = double(lu);
    W = double(W);
    kappa = double(kappa);
    gamma = double(gamma);
    for j = 1:T
        tStart = tic;
        xt = xs(j,:)';
        K = sdpvar(du, dx, 'full');
        objective = norm(K - squeeze(K_updates(j,:,:)), 2);
        constraints = [];
        constraints = [constraints, ...
                       K <= [kappa], ...
                       -K <= [kappa]
                       A - B * K <= [1-gamma], ...
                       - A + B * K <= [1-gamma], ...
                       -Lu * K * xt <= lu, ...
                       lx - Lx * A * xt + Lx * B * K * xt - Lx_norm * W >= 0];        
        
        options = sdpsettings('verbose', 0, 'solver', 'mosek');
        sol = optimize(constraints, objective, options);
        if ~(sol.problem == 0)
            error('Something went wrong...');
        end

        tEnd = toc(tStart);
        time(j+(i-1)*T) = tEnd;

    end
end

save("Time", "time");
mean(time)
std(time)
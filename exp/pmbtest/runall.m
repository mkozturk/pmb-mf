%% Add paths
addpath(genpath('../../matlab/minFunc_2012'));
addpath(genpath('../../matlab'));
%% Initialization
n = 100;
funcs = {@(x)rosenbrock(x), @(x)fletchcr(x), @(x)eg2(x), @(x)liarwhd(x), ...
         @(x)edensch(x), @(x)nonscomp(x)};
fun = funcs {1};
%% PMB
% x0 = 5.0 + rand(n, 1)*10.0;
pars.M = 5;
part.etta = 0.25;
pars.display = 0;
pars.maxiniter = 1000;
pars.maxiter = 5000;
pars.tol = 1.0e-5;
% pars.etta = 0.25;
% etta is very critical why????
pars.etta = rand();
pars.maxfcalls = 5000;
pmb_out = pmbsolve(fun, x0, pars);
fprintf('PMB Objective Function Value: %f\n', pmb_out.fval);
fprintf('PMB Final Gradient Norm Value: %f\n', max(abs(pmb_out.g)));
fprintf('PMB Exit Condition: %d\n\n', pmb_out.exit);

%% L-BFGS (default)
options.display = 'none';
options.useMex = 0; % For fair comparison in time
options.maxFunEvals = pars.maxfcalls;
options.MaxIter = pars.maxiter;
options.Method = 'lbfgs';
options.Corr = 5;
tstart = tic;
[~, lbfgs_f, ~, lbfgs_output] = minFunc(fun, x0, options);
lbfgs_time = toc(tstart);
fprintf('LBFGS Objective Function Value: %f\n', lbfgs_f);
fprintf('LBFGS Final Gradient Norm Value: %f\n', lbfgs_output.firstorderopt);
fprintf('LBFGS Exit Message: %s\n\n', lbfgs_output.message);
%% Figures

% plot(pmb_out.fhist, 'LineWidth', 2, 'DisplayName', 'PMB');

%%
addpath(genpath('../../matlab/minFunc_2012'));
addpath(genpath('../../matlab'));

maxFunEvals = 1000;

nInst = 5000;
nVars = 2000;
X = randn(nInst,nVars);
w = randn(nVars,1);
y = sign(X*w + randn(nInst,1));

w_init = zeros(nVars,1);
fun = @(w)LogisticLoss(w,X,y);

options.display = 'none';
options.useMex = 0; % For fair comparison in time
options.maxFunEvals = maxFunEvals;
options.MaxIter = maxFunEvals;

%% Steepest Descent
options.Method = 'sd';
tstart = tic;
[~, sd_f, ~, sd_output] = minFunc(fun, w_init, options);
sd_time = toc(tstart);
fprintf('Steepest Descent Objective Function Value: %f\n', sd_f);

%% Cyclic Steepest Descent
options.Method = 'csd';
tstart = tic;
[~, csd_f, ~, csd_output] = minFunc(fun, w_init, options);
csd_time = toc(tstart);
fprintf('Cyclic Steepest Descent Objective Function Value: %f\n', csd_f);

%% Barzilai & Borwein
options.Method = 'bb';
options.bbType = 1;
tstart = tic;
[~, bb_f, ~, bb_output] = minFunc(fun, w_init, options);
bb_time = toc(tstart);
fprintf('Barzilai-Borwein Objective Function Value: %f\n', bb_f);

%% Hessian-Free Newton
options.Method = 'newton0';
tstart = tic;
[~, hfn_f, ~, hfn_output] = minFunc(fun, w_init, options);
hfn_time = toc(tstart);
fprintf('Hessian-Free Objective Function Value: %f\n', hfn_f);

%% Conjugate Gradient
options.Method = 'cg';
tstart = tic;
[~, cg_f, ~, cg_output] = minFunc(fun, w_init, options);
cg_time = toc(tstart);
fprintf('Conjugate Gradient Objective Function Value: %f\n', cg_f);

%% Scaled Conjugate Gradient
options.Method = 'scg';
tstart = tic;
[~, scg_f, ~, scg_output] = minFunc(fun, w_init, options);
scg_time = toc(tstart);
fprintf('Scaled Conjugate Gradient Objective Function Value: %f\n', scg_f);

%% Preconditioned Conjugate Gradient
options.Method = 'pcg';
tstart = tic;
[~, pcg_f, ~, pcg_output] = minFunc(fun, w_init, options);
pcg_time = toc(tstart);
fprintf('Preconditioned Conjugate Gradient Objective Function Value: %f\n', pcg_f);

%% L-BFGS (default)
options.Method = 'lbfgs';
options.Corr = 5;
tstart = tic;
[~, lbfgs_f, ~, lbfgs_output] = minFunc(fun, w_init, options);
fprintf('LBFGS (default) Objective Function Value: %f\n', lbfgs_f);
lbfgs_time = toc(tstart);

%% PMB
pars.M = 5; pars.display = 0;
pars.maxiniter = 100; pars.etta = 0.25;
pars.maxiter = maxFunEvals; pars.tol = 1.0e-5;
pars.maxfcalls = maxFunEvals;
pmb_out = pmbsolve(fun, w_init, pars);
fprintf('PMB Objective Function Value: %f\n', pmb_out.fval);

%% Line Plots
figure;
maxx = 15;
% plot(1:maxx, sqrt(sd_output.trace.fval(1:maxx)), 'LineWidth', 2, 'DisplayName', 'SD'); hold on;
% plot(1:maxx, sqrt(csd_output.trace.fval(1:maxx)), 'LineWidth', 2, 'DisplayName', 'CSD');
% plot(1:maxx, sqrt(bb_output.trace.fval(1:maxx)), 'LineWidth', 2, 'DisplayName', 'BB');
% plot(1:maxx, sqrt(hfn_output.trace.fval(1:maxx)), 'LineWidth', 2, 'DisplayName', 'HFN');
% plot(1:maxx, sqrt(cg_output.trace.fval(1:maxx)), 'LineWidth', 2, 'DisplayName', 'CG');
% plot(1:maxx, sqrt(scg_output.trace.fval(1:maxx)), 'LineWidth', 2, 'DisplayName', 'SCG');
plot(1:maxx, sqrt(pcg_output.trace.fval(1:maxx)), 'LineWidth', 2, 'DisplayName', 'PCG'); hold on
plot(1:maxx, sqrt(lbfgs_output.trace.fval(1:maxx)), 'LineWidth', 2, 'DisplayName', 'LBFGS'); 
plot(1:maxx, sqrt(pmb_out.fhist(1:maxx)), 'LineWidth', 2, 'DisplayName', 'PMB');
hold off;
legend('show');
xlabel('Iterations');
ylabel('RMSE Values')

%% Barplots
figure;
fvals = [sd_f, csd_f, bb_f, hfn_f, cg_f, scg_f, pcg_f, lbfgs_f, pmb_out.fval];
bplot = bar(1:length(fvals), fvals, 'r');
xlabel('Optimization Methods');
ylabel('Final Objective Function Values')
alpha(bplot, 0.5);
set(gca,'XTickLabel',{'SD','CSD','BB', 'HFN', 'CG', 'SCG', 'PCG', 'LBFGS', 'PMB'});

figure;
ngvals = [sd_output.firstorderopt, csd_output.firstorderopt, ...
    bb_output.firstorderopt, hfn_output.firstorderopt, ...
    cg_output.firstorderopt, scg_output.firstorderopt, ...
    pcg_output.firstorderopt, lbfgs_output.firstorderopt, max(abs(pmb_out.g))];
bplot = bar(1:length(ngvals), ngvals, 'b');
xlabel('Optimization Methods');
ylabel('Final Gradient Norm')
alpha(bplot, 0.5);
set(gca,'XTickLabel',{'SD','CSD','BB', 'HFN', 'CG', 'SCG', 'PCG', 'LBFGS', 'PMB'});

figure;
tvals = [sd_time, csd_time, ...
    bb_time, hfn_time, ...
    cg_time, scg_time, ...
    pcg_time, lbfgs_time, pmb_out.time];
bplot = bar(1:length(tvals), tvals, 'b');
xlabel('Optimization Methods');
ylabel('Time (secs)')
alpha(bplot, 0.5);
set(gca,'XTickLabel',{'SD','CSD','BB', 'HFN', 'CG', 'SCG', 'PCG', 'LBFGS', 'PMB'});
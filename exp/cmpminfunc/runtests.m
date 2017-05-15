%%
addpath(genpath('../../matlab/minFunc_2012'));
addpath(genpath('../../matlab'));
dset = [5, 20, 50];
mset = {'csd', 'bb', 'newton0', 'cg', 'scg', 'pcg', 'lbfgs'};
num_of_rep = 50;
maxFunEvals = 1000;

% 1M Data
load X_3883_6040.dat
Y = spconvert(X_3883_6040);
datasize = full(sum(sum(Y>0)));
[nrows, ncols] = size(Y);

% minFunc options
options = [];
options.display = 'none';
options.useMex = 0; % For fair comparison in time
options.maxFunEvals = maxFunEvals;
options.MaxIter = maxFunEvals;

% PMB options
pars.M = 5; pars.display = 0;
pars.maxiniter = 500; pars.etta = 0.25;
pars.maxiter = maxFunEvals; pars.tol = 1.0e-5;
pars.maxfcalls = maxFunEvals;


for lat=dset
    n = (nrows + ncols)*lat;
    X0 = sqrt(randi(5, n, num_of_rep)/lat);
    fun = @(x)seuc_fun(x, Y, lat, datasize);
    
    % minFunc testst
    for m=1:length(mset)
        fname = sprintf('out_1M_d%02d_%s', lat, mset{m});
        fprintf('Current file: %s\n', fname);
        for rep=1:num_of_rep
            
            x0 = X0(:,rep);
            
            options.Method = mset{m};
            if (strcmp(mset{m}, 'bb'))
                options.bbType = 1;
            end
            if (strcmp(mset{m}, 'lbfgs'))
                options.Corr = 5;
            end
            
            fprintf('\t Replication: %d\n', rep);
            tstart = tic;
            [~, m_f, ~, m_output] = minFunc(fun, x0, options);
            m_time = toc(tstart);
            
            fileID = fopen(fname, 'a');
            fprintf(fileID, '%f\t%f\t%f\n', sqrt(2.0*m_f), m_output.firstorderopt, m_time);
            fclose(fileID);
        end
    end
    
    % pmb tests
    fname = sprintf('out_1M_d%02d_%s', lat, 'pmb');
    fprintf('Current file: %s\n', fname);
    for rep=1:num_of_rep
        
        x0 = X0(:,rep);
        
        fprintf('\t Replication: %d\n', rep);
        tstart = tic;
        pmb_out = pmbsolve(fun, x0, pars);
        pmb_time = toc(tstart);
        
        fileID = fopen(fname, 'a');
        fprintf(fileID, '%f\t%f\t%f\n', sqrt(2.0*pmb_out.fval), max(abs(pmb_out.g)), pmb_time);
        fclose(fileID);
    end
    
end

function [output] = pmbsolve(fun, x_0, pars)

output.exit = -1;
output.fhist=[];
output.nghist=[];
output.fcalls = 0;

x = x_0;
[f,g] = fun(x);
output.fcalls = output.fcalls + 1;

n = length(x);
S = zeros(n, pars.M);
Y = zeros(n, pars.M);
YS = zeros(pars.M, 1);
mem_start = 1;
mem_end = 0;
Hdiag = 1;

iter = 1;
tstart = tic;
while(iter<pars.maxiter && output.fcalls<pars.maxfcalls)
    
    if(norm(g,'inf') < pars.tol)
        output.exit = 1;
        output.time = toc(tstart);
        break;
    end
    
    output.fhist(iter)=f;
    output.nghist(iter)=norm(g,'inf');
    
    if (iter == 1)
        s = -g;
    else
        [s, Hdiag, S, Y, YS, mem_start, mem_end] = ...
            precond(g, g_old, s, Hdiag, S, Y, YS, pars.M, mem_start, mem_end);
    end
    g_old = g;
    
    initer = 1;
    Delta = s'*g;
    while(initer < pars.maxiniter)
        
        xt = x + s; % x^k_t
        
        [ft,gt] = fun(xt); % f^k_t and g^k_t
        output.fcalls = output.fcalls + 1;
        
        if(f - ft > -0.1*Delta)
            x = xt; f = ft; g = gt;
            break;
        end
        
        y = gt-g;  % y^k_t
        ys = y'*s; % v1
        ss = s'*s; % v2
        yy = y'*y; % v3
        yg = y'*g; % v4
        gg = g'*g; % v5
        sg = s'*g; % v6
        
        etta = min(max(ys/yy, Hdiag), pars.etta);
        sigma = 1/2*(sqrt(ss)*(sqrt(yy)+1/etta*sqrt(gg))-ys); % sigma
        
        teta = (ys + 2*sigma)^2-ss*yy; % theta
        cg= -ss/(2*sigma); % cg(sigma)
        cs = cg/teta*(-(ys+2*sigma)*yg+yy*sg); % cs(sigma)
        cy = cg/teta*(-(ys+2*sigma)*sg+ss*yg); % cy(sigma)
        
        Delta = cg*gg+cy*yg+cs*sg;
        s = cg*g+cs*s+cy*y;
        
        initer = initer+1;
        
    end % inner iterations
    
    if(initer >= pars.maxiniter)
        output.exit = 0;
        output.time = toc(tstart);
        if (pars.display)
            fprintf('Maximum number of inner iterations (maxiniter) is reached\n');
        end
        break;
    end
    
    if (pars.display)
        fprintf('PMB - Iter: %d ===> f = %f \t norm(g) = %f\n', iter, f, norm(g, 'inf'));
    end
    
    iter = iter + 1;
    
end % outer iterations

if (output.exit < 0)
    output.time = toc(tstart);
    if(iter >= pars.maxiter)
        if (pars.display)
            fprintf('Maximum number of iterations (maxiter) is reached\n');
        end
        output.exit = -1; % This line is for code clarity - not needed
    else
        if (pars.display)
            fprintf('Maximum number of function calls (maxfcalls) is reached\n');
        end
        output.exit = -2;
    end
end

output.fval = f;
output.g = g;
output.niter = iter;

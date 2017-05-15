function [s, Hdiag, S, Y, YS, mem_start, mem_end] = ...
    precond(g, g_old, s, Hdiag, S, Y, YS, M, mem_start, mem_end)

y = g-g_old;
ys = y'*s;
if (ys > 1e-10)
    if (mem_end < M)
        mem_end = mem_end+1;
        if (mem_start ~= 1)
            if (mem_start == M)
                mem_start = 1;
            else
                mem_start = mem_start+1;
            end
        end
    else
        mem_start = min(2,M);
        mem_end = 1;
    end
    S(:,mem_end) = s;
    Y(:,mem_end) = y;
    YS(mem_end) = ys;
    Hdiag = ys/(y'*y);
end
if (mem_start == 1)
    ind = 1:mem_end;
    nMem = mem_end-mem_start+1;
else
    ind = [mem_start:M 1:mem_end];
    nMem = M;
end
al = zeros(nMem,1);
be = zeros(nMem,1);
s = -g;
for j = 1:length(ind)
    i = ind(end-j+1);
    al(i) = (S(:,i)'*s)/YS(i);
    s = s-al(i)*Y(:,i);
end
s = Hdiag*s;
for i = ind
    be(i) = (Y(:,i)'*s)/YS(i);
    s = s + S(:,i)*(al(i)-be(i));
end

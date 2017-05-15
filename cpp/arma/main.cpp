//
//  main.cpp
//
//  Created by S. Ilker Birbil on 19 November 2016.
//  Copyright (c) 2015 S. Ilker Birbil. All rights reserved.
//
// Compilation:
// g++ main.cpp -o main -O2 -I/opt/local/include -L/opt/local/lib -larmadillo

#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

typedef struct {
        int exit;
        double fval;
        double gnorm;
        double iters;
        double fcalls;
} Output;

typedef struct {
        long int maxfcalls;
        long int maxiter;
        long int maxiniter;
        double tol;
        double etta;
        int M;
        bool disp;
} Options;

void rosenbrock(vec x, double &f, vec &g) {

        f = 100.0*pow((x[1] - pow(x[0], 2.0)), 2.0) + pow((1.0 - x[0]), 2.0);

        g[0] = -400.0*x[0]*(x[1] - pow(x[0], 2.0)) - 2.0*(1.0 - x[0]);
        g[1] = 200.0*(x[1] - pow(x[0], 2.0));
}

void extrosenbrock(vec x, double &f, vec &g) {

        int n = x.size();

        f = 0.0;
        for (int i=0; i<n/2; i++) {
                f += 100.0*pow(x[2*i+1] - pow(x[2*i], 2.0), 2.0) + pow((1.0 - x[2*i]), 2.0);
        }

        g[0] = -400.0*x[0]*(x[1] - pow(x[0], 2.0)) - 2.0*(1.0 - x[0]);
        for (int i=1; i<n-1; i++) {
                g[i] = 200.0*(x[i] - pow(x[i-1], 2.0)) - 400.0*x[i]*(x[i+1] - pow(x[i], 2.0)) - 2.0*(1.0-x[i]);
        }
        g[n-1] = 200.0*(x[n-1] - pow(x[n-2], 2.0));

}

void pmbsolve(vec x, void (*fun)(vec, double &, vec &), Options opts, Output &out) {

        out.exit = -1;
        out.fcalls = 0;

        int n;
        n = x.size();
        vec g(n);
        double f = 0.0;
        (*fun)(x, f, g);
        out.fcalls += 1;

        mat S = zeros<mat>(n, opts.M);
        mat Y = zeros<mat>(n, opts.M);
        vec YS = zeros<vec>(opts.M);
        int mem_start = 1;
        int mem_end = 0;
        double scf = 1.0; // Scaling factor

        vec xt(n), gt(n), s(n), sprec(n), y(n);
        double ft = 0.0;
        double ys, ss, yy, yg, gg, sg;
        double etta, sigma, teta, cg, cs, cy;

        int nMem = 1; // Current memory size
        int iter = 1; int initer;

        while(iter < opts.maxiter && out.fcalls < opts.maxfcalls) {

                if (norm(g, "inf") < opts.tol) {
                        out.exit = 1;
                        break;
                }

                if (iter == 1)
                        s = -g;
                else
                        s = sprec;

                gg = dot(g,g);

                initer = 1;
                while(initer < opts.maxiniter) {

                        xt = x + s;

                        (*fun)(xt, ft, gt);
                        out.fcalls += 1;

                        y = gt-g;
                        ys = dot(y,s); ss = dot(s,s); yy = dot(y,y);
                        yg = dot(y,g); sg = dot(s,g); // "gg" done in outer iteration

                        if (ys > 1.0e-10) {
                                if (mem_end < opts.M) {
                                        mem_end += 1;
                                        if (mem_start != 1) {
                                                if (mem_start == opts.M)
                                                        mem_start = 1;
                                                else
                                                        mem_start += 1;
                                        }
                                }
                                else {
                                        mem_start = min(2, opts.M);
                                        mem_end = 1;
                                }
                                S.col(mem_end - 1) = s;
                                Y.col(mem_end - 1) = y;
                                YS(mem_end - 1) = ys;
                                scf = ys/yy;
                        }

                        // Cyclic shift --->
                        ivec ind(nMem);
                        if (mem_start == 1) {
                                ind = regspace<ivec>(1, mem_end);
                                nMem = mem_end - mem_start + 1;
                        }
                        else {
                                ind.subvec(0, opts.M-mem_start) = regspace<ivec>(mem_start, opts.M);
                                ind.subvec(opts.M-mem_start+1, opts.M-1) = regspace<ivec>(1, mem_end);
                                nMem = opts.M;
                        }
                        // <---

                        sprec = -gt;

                        vec al = zeros<vec>(nMem);
                        vec be = zeros<vec>(nMem);
                        for(int i, j=0; j<nMem; j++) {
                                i = ind[nMem-j-1]-1;
                                al[i] = (dot(S.col(i), sprec))/YS[i];
                                sprec = sprec + ((al[i] - be[i])*S.col(i));
                        }

                        sprec = scf*sprec;

                        for(int i, j=0; j<nMem; j++) {
                                i = ind[j]-1;
                                al[i] = (dot(S.col(i), sprec))/YS[i];
                                sprec = sprec + ((al[i] - be[i])*S.col(i));
                        }

                        if(f - ft > -0.1*sg) {
                                x = xt; f = ft; g = gt;
                                break;
                        }
                        etta = min(max(ys/yy, scf), opts.etta);
                        sigma = 0.5*(sqrt(ss)*(sqrt(yy)+1.0/etta*sqrt(gg))-ys);
                        teta = pow((ys + 2.0*sigma),2.0)-ss*yy;
                        cg = -ss/(2.0*sigma);
                        cs = cg/teta*(-(ys+2.0*sigma)*yg+yy*sg);
                        cy = cg/teta*(-(ys+2.0*sigma)*sg+ss*yg);

                        s = cg*g + cs*s +cy*y;

                        initer += 1;

                } //initer

                if(initer >= opts.maxiniter) {
                        out.exit = 0;
                        cout << "Maximum number of inner iterations (maxiniter) is reached." << endl;
                        break;
                }

                if (opts.disp)
                        cout << "Iteration: " << iter << " -- f: " << f << endl;

                iter += 1;

        } //iter


        if (out.exit < 0) {
                if(iter >= opts.maxiter) {
                        cout << "Maximum number of iterations (maxiter) is reached." << endl;
                        out.exit = -1;
                }
                else {
                        cout << "Maximum number of function calls (maxfcalls) is reached." << endl;
                        out.exit = -2;
                }
        }

        out.fval = f;
        out.gnorm = norm(g, "inf");
        out.iters = iter;

}

int main(int argc, const char * argv[]) {

        arma_rng::set_seed(int(time(NULL)));
        int n = 2;
        vec x(n);
        x = 5.0*x.randu();

        Options options;
        options.tol = 1e-05;
        options.etta = 0.25;
        options.maxiter = 50000;
        options.maxiniter = 100000;
        options.maxfcalls = 50000;
        options.M = 5;
        options.disp = false;

        Output output;
        void (*fun)(vec, double &, vec &) = rosenbrock; // Pointer to the function returns f value and df vector

        pmbsolve(x, fun, options, output);

        cout << endl << "***** OUTPUT *****" << endl;
        cout << "Exit: " << output.exit << endl;
        cout << "Fval: " << output.fval << endl;
        cout << "Norm: " << output.gnorm << endl;
        cout << "Iterations: " << output.iters << endl;
        cout << "Total function calls: " << output.fcalls;
        cout << endl << "******************" << endl;

        return 0;
}

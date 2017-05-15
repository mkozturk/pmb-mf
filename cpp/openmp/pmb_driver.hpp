//  pmtob_driver.cpp
//  Name: pmb_driver
//  Author: kamer

#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include "omp.h"
#include "common.h"
#include "pmb_precond.hpp"

using namespace std;

template<class Data>
void pmb_driver(Data* data, opt_prec_t* x_0, Options opts, Output& output, int n) {
        opt_prec_t* x = new opt_prec_t[n];
        opt_prec_t* g = new opt_prec_t[n];
        opt_prec_t* gt = new opt_prec_t[n];
        opt_prec_t* g_old = new opt_prec_t[n];
        opt_prec_t* s = new opt_prec_t[n];
        opt_prec_t* y = new opt_prec_t[n];
        opt_prec_t* xt = new opt_prec_t[n];
        opt_prec_t* S = new opt_prec_t[n * opts.M];
        opt_prec_t* Y = new opt_prec_t[n * opts.M];
        opt_prec_t* YS = new opt_prec_t[opts.M];
        opt_prec_t* al = new opt_prec_t [opts.M];
        opt_prec_t* be = new opt_prec_t [opts.M];
        int* ind = new int[opts.M];

        opt_prec_t ys = 1.0, yy = 1.0, ss = 1.0, yg = 1.0, sg = 1.0, gg = 1.0, cg, f, ft, Hdiag;
        int evals = 1, iteration = 0, mem_end = 0, mem_start = 1;

        output.exit = 1;

#ifdef TIMER
        double t1, tt;

        double totalTime = 0.0f;
        double funTime = 0.0f;
        double vecOpTime = 0.0f;
        double precondTime = 0.0f;
        double memTime = 0.0f;

        t1 = omp_get_wtime();
        tt = omp_get_wtime();
#endif

#pragma omp parallel for schedule(static)
        for(int k = 0; k < n; k++) {
                x[k] = x_0[k];
                g[k] = gt[k] = s[k] = y[k] = xt[k] = g_old[k] = 0;
        }

        for(int i = 0; i < opts.M; i++) {
                opt_prec_t* dest_S = S + i*n;
                opt_prec_t* dest_Y = Y + i*n;
#pragma omp parallel for schedule(static)
                for(int k = 0; k < n; k++) {
                        dest_S[k] = dest_Y[k] = 0;
                }
        }
        memset(YS, 0, (opts.M *sizeof(opt_prec_t)));

#ifdef TIMER
        memTime += omp_get_wtime() - t1;
        t1 = omp_get_wtime();
#endif

        pmb_function(data, x, f, g);

#ifdef TIMER
        funTime += omp_get_wtime() - t1;
#endif

        while (iteration < opts.maxiter) {
#ifdef TIMER
                t1 = omp_get_wtime();
#endif
                opt_prec_t gmax = fabs(g[0]);

                for (int i = 1; i < n; i++) {
                        gmax = max(gmax, fabs(g[i]));
                }

                if (gmax < opts.tol) {
                        output.exit = 0;
                        break;
                }
                iteration++;

#ifdef DEBUG
                cout << "Function Value: " << f << ", iteration: " << iteration << ", Gmax: " << gmax << endl;
#endif

                opt_prec_t alpha = min(ys / yy, 1.0);
                alpha = min(ys / yy, 1.0);
                cg = -1 * alpha;

#ifdef TIMER
                vecOpTime += omp_get_wtime() - t1;
                t1 = omp_get_wtime();
#endif

                pmb_precond(s, y, g, g_old,
                            Hdiag, mem_start, mem_end, ind,
                            S, Y, YS, al, be, n, opts.M, iteration);

#ifdef TIMER
                precondTime += omp_get_wtime() - t1;
                t1 = omp_get_wtime();
#endif

#pragma omp parallel for schedule(static)
                for(int k = 0; k < n; k++) {
                        g_old[k] = g[k];
                }

#ifdef TIMER
                memTime += omp_get_wtime() - t1;
#endif

                int inner_iteration = 1;
                while (inner_iteration < opts.maxinneriter) {
#ifdef TIMER
                        t1 = omp_get_wtime();
#endif
#pragma omp parallel for schedule(static)
                        for (int k = 0; k < n; k++) {
                                xt[k] = s[k] + x[k];
                        }
#ifdef TIMER
                        vecOpTime += omp_get_wtime() - t1;
                        t1 = omp_get_wtime();
#endif

                        pmb_function(data, xt, ft, gt);

#ifdef TIMER
                        funTime += omp_get_wtime() - t1;
                        t1 = omp_get_wtime();
#endif

                        evals++;
                        ys = yg = sg = yy = ss = gg = 0;
#pragma omp parallel for schedule(static) reduction(+:ys,yg,sg,yy,ss,gg)
                        for (int k = 0; k < n; k++) {
                                opt_prec_t yv =  gt[k] - g[k];
                                opt_prec_t sv = s[k];
                                opt_prec_t gv = g[k];

                                y[k] = yv;

                                ys += yv * sv;  yg += yv * gv;  sg += sv * gv;  yy += yv * yv;  ss += sv * sv;  gg += gv * gv;
                        }

#ifdef TIMER
                        vecOpTime += omp_get_wtime() - t1;
                        t1 = omp_get_wtime();
#endif
                        if (f - ft > -0.1 * sg) {
#ifdef DEBUG
                                cout << "\t\tInner iterations are done." << endl;
#endif

                                f = ft;
#pragma omp parallel for schedule(static)
                                for(int k = 0; k < n; k++) {
                                        x[k] = xt[k];
                                        g[k] = gt[k];
                                }
                                break;
                        }
#ifdef TIMER
                        memTime += omp_get_wtime() - t1;
#endif
                        inner_iteration++;

                        opt_prec_t etta = min(max(ys/yy, Hdiag), 0.25);

                        opt_prec_t sigma = 0.5 * (sqrt(ss) * (sqrt(yy) + 1.0 / etta * sqrt(gg)) - ys);
                        opt_prec_t teta = pow((ys + 2.0 * sigma), 2.0) - ss * yy;

                        cg = -ss / (2 * sigma);
                        opt_prec_t cy = cg / teta * (-(ys + 2.0 * sigma) * sg + ss * yg);
                        opt_prec_t cs = cg / teta * (-(ys + 2.0 * sigma) * yg + yy * sg);

#ifdef TIMER
                        t1 = omp_get_wtime();
#endif
#pragma omp parallel for schedule(static)
                        for (int k = 0; k < n; k++) {
			  s[k] = g[k] * cg + s[k] * cs + y[k] * cy;
			}
#ifdef TIMER
                        memTime += omp_get_wtime() - t1;
#endif
                }

		//cout << f << endl;
#ifdef DEBUG
                cout << "\tInner iteration is done." << endl;
#endif

                if (inner_iteration >= opts.maxinneriter) {
                        output.exit = 2;
                        break;
                }
        }
#ifdef TIMER
        totalTime += omp_get_wtime() - tt;
#endif
        output.fval = f;
        output.iterations = iteration;

        opt_prec_t gmax = g[0];
        for (int i = 1; i < n; i++) {
                gmax = max(gmax, g[i]);
        }
        output.gradient_norm = gmax;
        output.evaluations = evals;

#ifdef TIMER
        cout << "Overall   time:\t" << totalTime << endl;
        cout << "  Func.   time:\t" << funTime << endl;
        cout << "  Vec op. time:\t" << vecOpTime << endl;
        cout << "  Precond time:\t" << precondTime << endl;
        cout << "  Memory  time:\t" << memTime << endl;
        cout << "  Extra   time:\t" << totalTime - funTime - vecOpTime - precondTime - memTime << endl;
#endif

        delete[] x;
        delete[] g;
        delete[] gt;
        delete[] g_old;
        delete[] s;
        delete[] y;
        delete[] xt;
        delete[] S;
        delete[] Y;
        delete[] YS;
        delete[] al;
        delete[] be;
        delete[] ind;
}

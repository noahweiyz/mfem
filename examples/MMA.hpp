#ifndef __MMA__HPP
#define __MMA__HPP

#include "mfem.hpp"
#include <petsc.h>

/* -----------------------------------------------------------------------------
Authors: Niels Aage
 Copyright (C) 2013-2020,
This MMA implementation is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.
This Module is distributed in the hope that it will be useful,implementation
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public
License along with this Module; if not, write to the Free Software
Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
-------------------------------------------------------------------------- */
class MMA {
  public:
    // Construct using defaults subproblem penalization
    MMA(PetscInt n, PetscInt m, Vec x);
    // User defined subproblem penalization
    MMA(MPI_Comm comm_,PetscInt n, PetscInt m, Vec x, PetscScalar* a, PetscScalar* c, PetscScalar* d);
    // Initialize with restart from itr
    MMA(PetscInt n, PetscInt m, PetscInt itr, Vec xo1, Vec xo2, Vec U, Vec L);
    // Initialize with restart and specify subproblem parameters
    MMA(PetscInt n, PetscInt m, PetscInt itr, Vec xo1, Vec xo2, Vec U, Vec L, PetscScalar* a, PetscScalar* c,
        PetscScalar* d);
    // Destructor
    ~MMA();

    // Set and solve a subproblem: return new xval
    PetscErrorCode Update(Vec xval, Vec dfdx, PetscScalar* gx, Vec* dgdx, Vec xmin, Vec xmax);

    // Return necessary data for possible restart
    PetscErrorCode Restart(Vec xo1, Vec xo2, Vec U, Vec L);

    // Set the aggresivity of the moving asymptotes
    PetscErrorCode SetAsymptotes(PetscScalar init, PetscScalar decrease, PetscScalar increase);

    // do/don't add convexity approx to constraints: default=false
    PetscErrorCode ConstraintModification(PetscBool conMod) {
        constraintModification = conMod;
        return 0;
    };

    // val=0: default, val=1: increase robustness, i.e
    // control the spacing between L < alp < x < beta < U,
    PetscErrorCode SetRobustAsymptotesType(PetscInt val);

    // Sets outer movelimits on all primal design variables
    // This is often requires to prevent the solver from oscilating
    PetscErrorCode SetOuterMovelimit(PetscScalar Xmin, PetscScalar Xmax, PetscScalar movelim, Vec x, Vec xmin,
                                     Vec xmax);

    // Return KKT residual norms (norm2 and normInf)
    PetscErrorCode KKTresidual(Vec xval, Vec dfdx, PetscScalar* gx, Vec* dgdx, Vec xmin, Vec xmax, PetscScalar* norm2,
                               PetscScalar* normInf);

    // Inf norm on diff between two vectors: SHOULD NOT BE HERE - USE BASIC
    // PETSc!!!!!
    PetscScalar DesignChange(Vec x, Vec xold);

  private:
    // Set up the MMA subproblem based on old x's and xval
    PetscErrorCode GenSub(Vec xval, Vec dfdx, PetscScalar* gx, Vec* dgdx, Vec xmin, Vec xmax);

    // Interior point solver for the subproblem
    PetscErrorCode SolveDIP(Vec xval);

    // Compute primal vars based on dual solution
    PetscErrorCode XYZofLAMBDA(Vec x);

    // Dual gradient
    PetscErrorCode DualGrad(Vec x);

    // Dual Hessian
    PetscErrorCode DualHess(Vec x);

    // Dual line search
    PetscErrorCode DualLineSearch();

    // Dual residual
    PetscScalar DualResidual(Vec x, PetscScalar epsi);

    // Problem size and iteration counter
    PetscInt n, m, k;

    // "speed-control" for the asymptotes
    PetscScalar asyminit, asymdec, asyminc;

    // do/don't add convexity constraint approximation in subproblem
    PetscBool constraintModification; // default = FALSE

    // Bool specifying if non lin constraints are included or not
    PetscBool NonLinConstraints;

    // 0: (default) span between alp L x U beta,
    // 1: increase the span for further robustness
    PetscInt RobustAsymptotesType;

    // Local vectors: penalty numbers for subproblem
    PetscScalar *a, *c, *d;

    // Local vectors: elastic variables
    PetscScalar* y;
    PetscScalar  z;

    // Local vectors: Lagrange multipliers:
    PetscScalar *lam, *mu, *s;

    // Global: Asymptotes, bounds, objective approx., constraint approx.
    Vec L, U, alpha, beta, p0, q0, *pij, *qij;

    // Local: subproblem constant terms, dual gradient, dual hessian
    PetscScalar *b, *grad, *Hess;

    // Global: Old design variables
    Vec xo1, xo2;

    // Math helpers
    PetscErrorCode Factorize(PetscScalar* K, PetscInt nn);
    PetscErrorCode Solve(PetscScalar* K, PetscScalar* x, PetscInt nn);
    PetscScalar    Min(PetscScalar d1, PetscScalar d2);
    PetscScalar    Max(PetscScalar d1, PetscScalar d2);
    PetscInt       Min(PetscInt d1, PetscInt d2);
    PetscInt       Max(PetscInt d1, PetscInt d2);
    PetscScalar    Abs(PetscScalar d1);

    // Communicator
    MPI_Comm mma_comm;
};


namespace mfem {

class NativeMMA
{
public:
    // User defined subproblem penalization
    NativeMMA(MPI_Comm comm_, int m, mfem::Vector& x, double* a, double* c, double* d){

        comm=comm_;
        num_con=m;
        Vec pv;
        VecCreateMPI(comm,x.Size(),PETSC_DETERMINE,&pv);

        PetscScalar* ap=new PetscScalar[m];
        PetscScalar* cp=new PetscScalar[m];
        PetscScalar* dp=new PetscScalar[m];
        for(int i=0;i<m;i++)
        {
            ap[i]=a[i];
            cp[i]=c[i];
            dp[i]=d[i];
        }

        PetscInt nn;
        VecGetSize(pv,&nn);

        mma=new MMA(comm_,nn, m, pv, ap, cp, dp);

        delete [] ap;
        delete [] cp;
        delete [] dp;

        VecDestroy(&pv);

        //allocate the native PETSc objects necessary for the subproblems

        VecCreateMPI(comm,x.Size(),PETSC_DETERMINE,&xval);
        VecCreateMPI(comm,x.Size(),PETSC_DETERMINE,&dfdx);
        VecCreateMPI(comm,x.Size(),PETSC_DETERMINE,&xmin);
        VecCreateMPI(comm,x.Size(),PETSC_DETERMINE,&xmax);

        VecDuplicateVecs(xval,num_con, &dgdx);
    }

    ~NativeMMA()
    {
        delete mma;

        VecDestroy(&xval);
        VecDestroy(&dfdx);
        VecDestroy(&xmin);
        VecDestroy(&xmax);
        VecDestroyVecs(num_con,&dgdx);
    }

    // Set and solve a subproblem: return new xval
    void Update(Vector& xval_, Vector& dfdx_, double* gx_, Vector* dgdx_, Vector& xmin_, Vector& xmax_){
        //copy data
        double* data;
        VecGetArray(xval,&data);
        for(int i=0;i<xval_.Size();i++){data[i]=xval_[i];}
        VecRestoreArray(xval,&data);
        //dfdx
        VecGetArray(dfdx,&data);
        for(int i=0;i<xval_.Size();i++){data[i]=dfdx_[i];}
        VecRestoreArray(dfdx,&data);
        //dgdx
        for(int j=0;j<num_con;j++){
            VecGetArray(dgdx[j],&data);
            for(int i=0;i<xval_.Size();i++){data[i]=(dgdx_[j])[i];}
            VecRestoreArray(dgdx[j],&data);
        }
        //xmin
        VecGetArray(xmin,&data);
        for(int i=0;i<xval_.Size();i++){data[i]=xmin_[i];}
        VecRestoreArray(xmin,&data);
        //xmax
        VecGetArray(xmax,&data);
        for(int i=0;i<xval_.Size();i++){data[i]=xmax_[i];}
        VecRestoreArray(xmax,&data);

        mma->Update(xval,dfdx,gx_,dgdx,xmin,xmax);

        VecGetArray(xval,&data);
        for(int i=0;i<xval_.Size();i++){xval_[i]=data[i];}
        VecRestoreArray(xval,&data);
    }
private:
    MMA* mma;
    int num_con;
    MPI_Comm comm;

    Vec xval;
    Vec dfdx;
    Vec* dgdx;
    Vec xmin;
    Vec xmax;


};

}


#include "MMA.hpp"
#include <iostream>
#include <math.h>

/* -----------------------------------------------------------------------------
Authors: Niels Aage
 Copyright (C) 2013-2019,
This MMA implementation is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.
This Module is distributed in the hope that it will be useful,implementation
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public
License along with this Module; if not, write to the Free Software
Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
-------------------------------------------------------------------------- */

MMA::MMA(PetscInt nn, PetscInt mm, PetscInt kk, Vec xo1t, Vec xo2t, Vec Ut, Vec Lt, PetscScalar* at, PetscScalar* ct,
         PetscScalar* dt) {

    mma_comm=PETSC_COMM_WORLD;

    n = nn;
    m = mm;
    k = kk;
    if (k < 3) {
        PetscPrintf(mma_comm, "NOT A LEGAL RESTART POINT (k<3): EXPECT BREAKDOWN\n");
    }

    asyminit = 0.5;
    asymdec  = 0.7;
    asyminc  = 1.2;

    NonLinConstraints      = PETSC_TRUE;
    constraintModification = PETSC_FALSE;
    RobustAsymptotesType   = 0;

    a = new PetscScalar[m];
    c = new PetscScalar[m];
    d = new PetscScalar[m];

    memcpy(a, at, mm * sizeof(PetscScalar));
    memcpy(c, ct, mm * sizeof(PetscScalar));
    memcpy(d, dt, mm * sizeof(PetscScalar));

    y   = new PetscScalar[m];
    lam = new PetscScalar[m];

    VecDuplicate(xo1t, &L);
    VecDuplicate(xo1t, &U);

    VecDuplicate(xo1t, &alpha);
    VecDuplicate(xo1t, &beta);

    VecDuplicate(xo1t, &p0);
    VecDuplicate(xo1t, &q0);
    VecDuplicateVecs(xo1t, m, &pij);
    VecDuplicateVecs(xo1t, m, &qij);

    b = new PetscScalar[m];

    VecDuplicate(xo1t, &xo1);
    VecDuplicate(xo1t, &xo2);

    grad = new PetscScalar[m];
    mu   = new PetscScalar[m];
    s    = new PetscScalar[2 * m];

    Hess = new PetscScalar[m * m];

    // Now insert the values into xo1,xo2,U,L
    PetscInt nloc;
    VecGetLocalSize(xo1t, &nloc);

    // input
    PetscScalar *pxo1t, *pxo2t, *pUt, *pLt;
    VecGetArray(xo1t, &pxo1t);
    VecGetArray(xo2t, &pxo2t);
    VecGetArray(Ut, &pUt);
    VecGetArray(Lt, &pLt);

    // internal
    PetscScalar *pxo1, *pxo2, *pU, *pL;
    VecGetArray(xo1, &pxo1);
    VecGetArray(xo2, &pxo2);
    VecGetArray(U, &pU);
    VecGetArray(L, &pL);

    // Copy data
    memcpy(pxo1, pxo1t, nloc * sizeof(PetscScalar));
    memcpy(pxo2, pxo2t, nloc * sizeof(PetscScalar));
    memcpy(pU, pUt, nloc * sizeof(PetscScalar));
    memcpy(pL, pLt, nloc * sizeof(PetscScalar));

    // Restore arrays
    VecRestoreArray(xo1t, &pxo1t);
    VecRestoreArray(xo2t, &pxo2t);
    VecRestoreArray(Ut, &pUt);
    VecRestoreArray(Lt, &pLt);

    VecRestoreArray(xo1, &pxo1);
    VecRestoreArray(xo2, &pxo2);
    VecRestoreArray(U, &pU);
    VecRestoreArray(L, &pL);
}

MMA::MMA(PetscInt nn, PetscInt mm, PetscInt kk, Vec xo1t, Vec xo2t, Vec Ut, Vec Lt) {

    mma_comm=PETSC_COMM_WORLD;

    n = nn;
    m = mm;
    k = kk;
    if (k < 3) {
        PetscPrintf(mma_comm, "NOT A LEGAL RESTART POINT (k<3): EXPECT BREAKDOWN\n");
    }

    asyminit = 0.5;
    asymdec  = 0.7;
    asyminc  = 1.2;

    NonLinConstraints      = PETSC_TRUE;
    constraintModification = PETSC_FALSE;
    RobustAsymptotesType   = 0;

    a = new PetscScalar[m];
    c = new PetscScalar[m];
    d = new PetscScalar[m];

    for (PetscInt i = 0; i < m; i++) {
        a[i] = 0.0;
        c[i] = 1000.0;
        d[i] = 0.0;
    }

    y   = new PetscScalar[m];
    lam = new PetscScalar[m];

    VecDuplicate(xo1t, &L);
    VecDuplicate(xo1t, &U);

    VecDuplicate(xo1t, &alpha);
    VecDuplicate(xo1t, &beta);

    VecDuplicate(xo1t, &p0);
    VecDuplicate(xo1t, &q0);
    VecDuplicateVecs(xo1t, m, &pij);
    VecDuplicateVecs(xo1t, m, &qij);

    b = new PetscScalar[m];

    VecDuplicate(xo1t, &xo1);
    VecDuplicate(xo1t, &xo2);

    grad = new PetscScalar[m];
    mu   = new PetscScalar[m];
    s    = new PetscScalar[2 * m];

    Hess = new PetscScalar[m * m];

    // Now insert the values into xo1,xo2,U,L
    PetscInt nloc;
    VecGetLocalSize(xo1t, &nloc);

    // input
    PetscScalar *pxo1t, *pxo2t, *pUt, *pLt;
    VecGetArray(xo1t, &pxo1t);
    VecGetArray(xo2t, &pxo2t);
    VecGetArray(Ut, &pUt);
    VecGetArray(Lt, &pLt);

    // internal
    PetscScalar *pxo1, *pxo2, *pU, *pL;
    VecGetArray(xo1, &pxo1);
    VecGetArray(xo2, &pxo2);
    VecGetArray(U, &pU);
    VecGetArray(L, &pL);

    // Copy data
    memcpy(pxo1, pxo1t, nloc * sizeof(PetscScalar));
    memcpy(pxo2, pxo2t, nloc * sizeof(PetscScalar));
    memcpy(pU, pUt, nloc * sizeof(PetscScalar));
    memcpy(pL, pLt, nloc * sizeof(PetscScalar));

    // Restore arrays
    VecRestoreArray(xo1t, &pxo1t);
    VecRestoreArray(xo2t, &pxo2t);
    VecRestoreArray(Ut, &pUt);
    VecRestoreArray(Lt, &pLt);

    VecRestoreArray(xo1, &pxo1);
    VecRestoreArray(xo2, &pxo2);
    VecRestoreArray(U, &pU);
    VecRestoreArray(L, &pL);
}

MMA::MMA(MPI_Comm comm_, PetscInt nn, PetscInt mm, Vec x, PetscScalar* at, PetscScalar* ct, PetscScalar* dt) {

    mma_comm=comm_;

    n = nn;
    m = mm;

    asyminit = 0.5;
    asymdec  = 0.7;
    asyminc  = 1.2;

    NonLinConstraints      = PETSC_TRUE;
    constraintModification = PETSC_FALSE;
    RobustAsymptotesType   = 0;

    k = 0;

    a = new PetscScalar[m];
    c = new PetscScalar[m];
    d = new PetscScalar[m];

    memcpy(a, at, mm * sizeof(PetscScalar));
    memcpy(c, ct, mm * sizeof(PetscScalar));
    memcpy(d, dt, mm * sizeof(PetscScalar));

    y   = new PetscScalar[m];
    lam = new PetscScalar[m];

    VecDuplicate(x, &L);
    VecDuplicate(x, &U);

    VecDuplicate(x, &alpha);
    VecDuplicate(x, &beta);

    VecDuplicate(x, &p0);
    VecDuplicate(x, &q0);
    VecDuplicateVecs(x, m, &pij);
    VecDuplicateVecs(x, m, &qij);

    b = new PetscScalar[m];

    VecDuplicate(x, &xo1);
    VecDuplicate(x, &xo2);

    grad = new PetscScalar[m];
    mu   = new PetscScalar[m];
    s    = new PetscScalar[2 * m];

    Hess = new PetscScalar[m * m];
}

MMA::MMA(PetscInt nn, PetscInt mm, Vec x) {
    mma_comm=PETSC_COMM_WORLD;

    n = nn;
    m = mm;

    asyminit = 0.5;
    asymdec  = 0.7;
    asyminc  = 1.2;

    NonLinConstraints      = PETSC_TRUE;
    constraintModification = PETSC_FALSE;

    RobustAsymptotesType = 0;

    k = 0;

    a = new PetscScalar[m];
    c = new PetscScalar[m];
    d = new PetscScalar[m];

    for (PetscInt i = 0; i < m; i++) {
        a[i] = 0.0;
        c[i] = 1000.0;
        d[i] = 0.0;
    }

    y   = new PetscScalar[m];
    lam = new PetscScalar[m];

    VecDuplicate(x, &L);
    VecDuplicate(x, &U);

    VecDuplicate(x, &alpha);
    VecDuplicate(x, &beta);

    VecDuplicate(x, &p0);
    VecDuplicate(x, &q0);
    VecDuplicateVecs(x, m, &pij);
    VecDuplicateVecs(x, m, &qij);
    b = new PetscScalar[m];

    VecDuplicate(x, &xo1);
    VecDuplicate(x, &xo2);

    grad = new PetscScalar[m];
    mu   = new PetscScalar[m];
    s    = new PetscScalar[2 * m];

    Hess = new PetscScalar[m * m];
}

MMA::~MMA() {

    delete[] a;
    delete[] b;
    delete[] c;
    delete[] d;
    delete[] y;
    delete[] lam;
    VecDestroy(&L);
    VecDestroy(&U);
    VecDestroy(&alpha);
    VecDestroy(&beta);
    VecDestroy(&p0);
    VecDestroy(&q0);
    VecDestroyVecs(m, &pij);
    VecDestroyVecs(m, &qij);
    VecDestroy(&xo1);
    VecDestroy(&xo2);
    delete[] grad;
    delete[] mu;
    delete[] s;
    delete[] Hess;
}

// restart method

PetscErrorCode MMA::Restart(Vec xo1t, Vec xo2t, Vec Ut, Vec Lt) {

    PetscErrorCode ierr = 0;

    // Insert values into xo1t,xo2t,Ut,Lt
    PetscInt nloc;
    VecGetLocalSize(xo1t, &nloc);

    // input
    PetscScalar *pxo1t, *pxo2t, *pUt, *pLt;
    VecGetArray(xo1t, &pxo1t);
    VecGetArray(xo2t, &pxo2t);
    VecGetArray(Ut, &pUt);
    VecGetArray(Lt, &pLt);

    // internal
    PetscScalar *pxo1, *pxo2, *pU, *pL;
    VecGetArray(xo1, &pxo1);
    VecGetArray(xo2, &pxo2);
    VecGetArray(U, &pU);
    VecGetArray(L, &pL);

    // Copy data
    memcpy(pxo1t, pxo1, nloc * sizeof(PetscScalar));
    memcpy(pxo2t, pxo2, nloc * sizeof(PetscScalar));
    memcpy(pUt, pU, nloc * sizeof(PetscScalar));
    memcpy(pLt, pL, nloc * sizeof(PetscScalar));

    // Restore arrays
    VecRestoreArray(xo1t, &pxo1t);
    VecRestoreArray(xo2t, &pxo2t);
    VecRestoreArray(Ut, &pUt);
    VecRestoreArray(Lt, &pLt);

    VecRestoreArray(xo1, &pxo1);
    VecRestoreArray(xo2, &pxo2);
    VecRestoreArray(U, &pU);
    VecRestoreArray(L, &pL);

    return (ierr);
}

// Set the aggresivity of the moving asymptotes
PetscErrorCode MMA::SetAsymptotes(PetscScalar init, PetscScalar decrease, PetscScalar increase) {
    PetscErrorCode ierr = 0;

    // asymptotes initialization and increase/decrease
    asyminit = init;
    asymdec  = decrease;
    asyminc  = increase;
    return ierr;
}

PetscErrorCode MMA::SetRobustAsymptotesType(PetscInt val) {

    PetscErrorCode ierr = 0;

    RobustAsymptotesType = val;

    if (RobustAsymptotesType == 0 || RobustAsymptotesType == 1) {
    } else {
        RobustAsymptotesType = 0;
        PetscPrintf(mma_comm, "ERROR in MMA.cc/h: RobustAsymptotesType cannot be set to: %d \n", val);
    }
    return ierr;
}

PetscErrorCode MMA::SetOuterMovelimit(PetscScalar Xmin, PetscScalar Xmax, PetscScalar movlim, Vec x, Vec xmin,
                                      Vec xmax) {

    PetscErrorCode ierr = 0;

    PetscScalar *xv, *xmiv, *xmav;
    PetscInt     nloc;
    VecGetLocalSize(x, &nloc);
    VecGetArray(x, &xv);
    VecGetArray(xmin, &xmiv);
    VecGetArray(xmax, &xmav);
    for (PetscInt i = 0; i < nloc; i++) {
        xmav[i] = Min(Xmax, xv[i] + movlim);
        xmiv[i] = Max(Xmin, xv[i] - movlim);
    }
    VecRestoreArray(x, &xv);
    VecRestoreArray(xmin, &xmiv);
    VecRestoreArray(xmax, &xmav);
    return ierr;
}

PetscScalar MMA::DesignChange(Vec x, Vec xold) {

    PetscScalar *xv, *xo;
    PetscInt     nloc;
    VecGetLocalSize(x, &nloc);
    VecGetArray(x, &xv);
    VecGetArray(xold, &xo);
    PetscScalar ch = 0.0;
    for (PetscInt i = 0; i < nloc; i++) {
        ch    = PetscMax(ch, PetscAbsReal(xv[i] - xo[i]));
        xo[i] = xv[i];
    }
    PetscScalar tmp;
    MPI_Allreduce(&ch, &tmp, 1, MPIU_SCALAR, MPI_MAX, mma_comm);
    ch = tmp;
    VecRestoreArray(x, &xv);
    VecRestoreArray(xold, &xo);

    return (ch);
}

PetscErrorCode MMA::KKTresidual(Vec x, Vec dfdx, PetscScalar* fx, Vec* dgdx, Vec xmin, Vec xmax, PetscScalar* norm2,
                                PetscScalar* normInf) {
    PetscErrorCode ierr = 0;

    if (!NonLinConstraints) {
        PetscErrorPrintf("MMA->KKTresidual called WITH constraints but object was "
                         "allocated WITHOUT !\n");
        return -1;
    }

    PetscScalar *xp, *xminp, *xmaxp, *df0dxp, **dfdxp;
    PetscInt     locsiz;
    VecGetLocalSize(x, &locsiz);

    VecGetArray(x, &xp);
    VecGetArray(xmin, &xminp);
    VecGetArray(xmax, &xmaxp);
    VecGetArray(dfdx, &df0dxp);
    VecGetArrays(dgdx, m, &dfdxp);

    PetscScalar resi, ri, mu_min, mu_max;

    norm2[0]   = 0;
    normInf[0] = 0;
    for (PetscInt i = 0; i < locsiz; i++) {
        ri = df0dxp[i];
        for (PetscInt j = 0; j < m; j++) {
            ri += lam[j] * dfdxp[j][i];
        }
        mu_min = 0.0;
        if (xp[i] < xminp[i] + 1.0e-5 && ri > 0.0) {
            mu_min = ri;
        }
        mu_max = 0.0;
        if (xp[i] > xmaxp[i] - 1.0e-5 && ri < 0.0) {
            mu_max = -ri;
        }
        ri += -mu_min + mu_max;
        norm2[0] += pow(ri, 2.0);
        normInf[0] = Max(Abs(ri), normInf[0]);
        resi       = mu_min * (xp[i] - xminp[i]);
        norm2[0] += pow(resi, 2.0);
        normInf[0] = Max(Abs(resi), normInf[0]);
        resi       = mu_max * (xmaxp[i] - xp[i]);
        norm2[0] += pow(resi, 2.0);
        normInf[0] = Max(Abs(resi), normInf[0]);
    }

    VecRestoreArray(x, &xp);
    VecRestoreArray(xmin, &xminp);
    VecRestoreArray(xmax, &xmaxp);
    VecRestoreArray(dfdx, &df0dxp);
    VecRestoreArrays(dgdx, m, &dfdxp);
    PetscScalar n2tmp = norm2[0];
    PetscScalar nItmp = normInf[0];
    norm2[0]          = 0.0;
    normInf[0]        = 0.0;
    MPI_Allreduce(&n2tmp, norm2, 1, MPIU_SCALAR, MPI_SUM, mma_comm);
    MPI_Allreduce(&nItmp, normInf, 1, MPIU_SCALAR, MPI_MAX, mma_comm);
    ri = 0.0;
    for (PetscInt j = 0; j < m; j++) {
        ri += lam[j] * (a[j] * z + y[j] - fx[j]);
    }
    norm2[0] += pow(ri, 2.0);
    normInf[0] = Max(Abs(ri), normInf[0]);
    norm2[0]   = sqrt(norm2[0]);

    return ierr;
}

// Set and solve a subproblem: return new xval
PetscErrorCode MMA::Update(Vec xval, Vec dfdx, PetscScalar* gx, Vec* dgdx, Vec xmin, Vec xmax) {
    PetscErrorCode ierr = 0;

    if (!NonLinConstraints) {
        PetscErrorPrintf("MMA->Update called WITH constraints but object was "
                         "allocated WITHOUT !\n");
        return -1;
    }

    // Generate the subproblem
    GenSub(xval, dfdx, gx, dgdx, xmin, xmax);

    // Update xolds
    VecCopy(xo1, xo2);
    VecCopy(xval, xo1);

    // Solve the dual with an interior point method
    SolveDIP(xval);
    return ierr;
}

// PRIVATE METHODS

PetscErrorCode MMA::GenSub(Vec xval, Vec dfdx, PetscScalar* gx, Vec* dgdx, Vec xmin, Vec xmax) {
    PetscErrorCode ierr = 0;

    PetscScalar gamma, helpvar;

    k++;
    PetscInt nloc;
    VecGetLocalSize(xval, &nloc);
    PetscScalar *xv, *Lv, *Uv, *x1v, *x2v, *xminv, *xmaxv;
    PetscScalar *alf, *bet, *dfdxv, *p0v, *q0v, **dgdxv, **pijv, **qijv;
    if (k < 3) {
        VecAXPBYPCZ(L, (PetscScalar)1.0, -asyminit, (PetscScalar)0.0, xval, xmax);
        VecAXPY(L, asyminit, xmin);
        VecAXPBYPCZ(U, (PetscScalar)1.0, +asyminit, (PetscScalar)0.0, xval, xmax);
        VecAXPY(U, -asyminit, xmin);
    }
    VecGetArray(xval, &xv);
    VecGetArray(L, &Lv);
    VecGetArray(U, &Uv);
    VecGetArray(xo1, &x1v);
    VecGetArray(xo2, &x2v);
    VecGetArray(xmin, &xminv);
    VecGetArray(xmax, &xmaxv);

    VecGetArray(alpha, &alf);
    VecGetArray(beta, &bet);
    VecGetArray(dfdx, &dfdxv);
    VecGetArray(p0, &p0v);
    VecGetArray(q0, &q0v);

    VecGetArrays(dgdx, m, &dgdxv);
    VecGetArrays(pij, m, &pijv);
    VecGetArrays(qij, m, &qijv);
    if (k > 2) {
        for (PetscInt i = 0; i < nloc; i++) {
            helpvar = (xv[i] - x1v[i]) * (x1v[i] - x2v[i]);
            if (helpvar < 0.0) {
                gamma = asymdec;
            } else if (helpvar > 0.0) {
                gamma = asyminc;
            } else {
                gamma = 1.0;
            }
            Lv[i] = xv[i] - gamma * (x1v[i] - Lv[i]);
            Uv[i] = xv[i] + gamma * (Uv[i] - x1v[i]);
            PetscScalar xmi, xma;
            xmi = Max(1.0e-5, xmaxv[i] - xminv[i]);
            if (RobustAsymptotesType == 0) {
                Lv[i] = Max(Lv[i], xv[i] - 10.0 * xmi);
                Lv[i] = Min(Lv[i], xv[i] - 0.01 * xmi);
                Uv[i] = Max(Uv[i], xv[i] + 0.01 * xmi);
                Uv[i] = Min(Uv[i], xv[i] + 10.0 * xmi);
            } else if (RobustAsymptotesType == 1) {
                Lv[i] = Max(Lv[i], xv[i] - 100.0 * xmi);
                Lv[i] = Min(Lv[i], xv[i] - 1.0e-4 * xmi);
                Uv[i] = Max(Uv[i], xv[i] + 1.0e-4 * xmi);
                Uv[i] = Min(Uv[i], xv[i] + 100.0 * xmi);
                xmi   = xminv[i] - 1.0e-5;
                xma   = xmaxv[i] + 1.0e-5;
                if (xv[i] < xmi) {
                    Lv[i] = xv[i] - (xma - xv[i]) / 0.9;
                    Uv[i] = xv[i] + (xma - xv[i]) / 0.9;
                }
                if (xv[i] > xma) {
                    Lv[i] = xv[i] - (xv[i] - xmi) / 0.9;
                    Uv[i] = xv[i] + (xv[i] - xmi) / 0.9;
                }
            }
        }
    }
    PetscScalar dfdxp, dfdxm;
    PetscScalar feps = 1.0e-6;
    for (PetscInt i = 0; i < nloc; i++) {
        alf[i] = Max(xminv[i], 0.9 * Lv[i] + 0.1 * xv[i]);
        bet[i] = Min(xmaxv[i], 0.9 * Uv[i] + 0.1 * xv[i]);
        dfdxp  = Max(0.0, dfdxv[i]);
        dfdxm  = Max(0.0, -1.0 * dfdxv[i]);
        p0v[i] = pow(Uv[i] - xv[i], 2.0) * (dfdxp + 0.001 * Abs(dfdxv[i]) + 0.5 * feps / (Uv[i] - Lv[i]));
        q0v[i] = pow(xv[i] - Lv[i], 2.0) * (dfdxm + 0.001 * Abs(dfdxv[i]) + 0.5 * feps / (Uv[i] - Lv[i]));
        for (PetscInt j = 0; j < m; j++) {
            dfdxp = Max(0.0, dgdxv[j][i]);
            dfdxm = Max(0.0, -1.0 * dgdxv[j][i]);
            if (constraintModification) {
                pijv[j][i] =
                    pow(Uv[i] - xv[i], 2.0) * (dfdxp + 0.001 * Abs(dgdxv[j][i]) + 0.5 * feps / (Uv[i] - Lv[i]));
                qijv[j][i] =
                    pow(xv[i] - Lv[i], 2.0) * (dfdxm + 0.001 * Abs(dgdxv[j][i]) + 0.5 * feps / (Uv[i] - Lv[i]));
            } else {
                pijv[j][i] = pow(Uv[i] - xv[i], 2.0) * (dfdxp);
                qijv[j][i] = pow(xv[i] - Lv[i], 2.0) * (dfdxm);
            }
        }
    }
    for (PetscInt j = 0; j < m; j++) {
        b[j] = 0.0;
        for (PetscInt i = 0; i < nloc; i++) {
            b[j] += pijv[j][i] / (Uv[i] - xv[i]) + qijv[j][i] / (xv[i] - Lv[i]);
        }
    }
    {
        PetscScalar* tmp = new PetscScalar[m];
        for (PetscInt i = 0; i < m; i++) {
            tmp[i] = 0.0;
        }
        MPI_Allreduce(b, tmp, m, MPIU_SCALAR, MPI_SUM, mma_comm);
        memcpy(b, tmp, sizeof(PetscScalar) * m);
        delete[] tmp;
    }
    for (PetscInt j = 0; j < m; j++) {
        b[j] += -gx[j];
    }
    VecRestoreArray(xval, &xv);
    VecRestoreArray(L, &Lv);
    VecRestoreArray(U, &Uv);
    VecRestoreArray(xo1, &x1v);
    VecRestoreArray(xo2, &x2v);
    VecRestoreArray(xmin, &xminv);
    VecRestoreArray(xmax, &xmaxv);

    VecRestoreArray(alpha, &alf);
    VecRestoreArray(beta, &bet);
    VecRestoreArray(dfdx, &dfdxv);

    VecRestoreArrays(dgdx, m, &dgdxv);
    VecRestoreArrays(pij, m, &pijv);
    VecRestoreArrays(qij, m, &qijv);
    return ierr;
}

PetscErrorCode MMA::SolveDIP(Vec x) {
    PetscErrorCode ierr = 0;

    for (PetscInt j = 0; j < m; j++) {
        lam[j] = c[j] / 2.0;
        mu[j]  = 1.0;
    }
    PetscScalar tol  = 1.0e-9 * sqrt(m + n);
    PetscScalar epsi = 1.0;
    PetscScalar err  = 1.0;
    PetscInt    loop;
    while (epsi > tol) {

        loop = 0;
        while (err > 0.9 * epsi && loop < 100) {
            loop++;
            XYZofLAMBDA(x);
            DualGrad(x);
            for (PetscInt j = 0; j < m; j++) {
                grad[j] = -1.0 * grad[j] - epsi / lam[j];
            }
            DualHess(x);
            Factorize(Hess, m);
            Solve(Hess, grad, m);
            for (PetscInt j = 0; j < m; j++) {
                s[j] = grad[j];
            }
            for (PetscInt i = 0; i < m; i++) {
                s[m + i] = -mu[i] + epsi / lam[i] - s[i] * mu[i] / lam[i];
            }
            DualLineSearch();
            XYZofLAMBDA(x);
            err = DualResidual(x, epsi);
        }
        epsi = epsi * 0.1;
    }
    return ierr;
}

PetscErrorCode MMA::XYZofLAMBDA(Vec x) {
    PetscErrorCode ierr = 0;

    PetscInt nloc;
    VecGetLocalSize(x, &nloc);
    PetscScalar *xv, **pijv, **qijv, *p0v, *q0v, *alf, *bet, *Lv, *Uv;
    VecGetArray(x, &xv);
    VecGetArray(p0, &p0v);
    VecGetArray(q0, &q0v);
    VecGetArray(alpha, &alf);
    VecGetArray(beta, &bet);
    VecGetArrays(pij, m, &pijv);
    VecGetArrays(qij, m, &qijv);
    VecGetArray(L, &Lv);
    VecGetArray(U, &Uv);
    PetscScalar lamai = 0.0;
    for (PetscInt i = 0; i < m; i++) {
        if (lam[i] < 0.0) {
            lam[i] = 0;
        }
        y[i] = Max(0.0, lam[i] - c[i]);
        lamai += lam[i] * a[i];
    }
    z = Max(0.0, 10.0 * (lamai - 1.0)); // SINCE a0 = 1.0
    PetscScalar pjlam, qjlam;
    for (PetscInt i = 0; i < nloc; i++) {
        pjlam = p0v[i];
        qjlam = q0v[i];
        for (PetscInt j = 0; j < m; j++) {
            pjlam += pijv[j][i] * lam[j];
            qjlam += qijv[j][i] * lam[j];
        }
        xv[i] = (sqrt(pjlam) * Lv[i] + sqrt(qjlam) * Uv[i]) / (sqrt(pjlam) + sqrt(qjlam));
        if (xv[i] < alf[i]) {
            xv[i] = alf[i];
        }
        if (xv[i] > bet[i]) {
            xv[i] = bet[i];
        }
    }
    VecRestoreArray(x, &xv);
    VecRestoreArrays(pij, m, &pijv);
    VecRestoreArrays(qij, m, &qijv);
    VecRestoreArray(p0, &p0v);
    VecRestoreArray(q0, &q0v);
    VecRestoreArray(alpha, &alf);
    VecRestoreArray(beta, &bet);
    VecRestoreArray(L, &Lv);
    VecRestoreArray(U, &Uv);
    return ierr;
}

PetscErrorCode MMA::DualGrad(Vec x) {
    PetscErrorCode ierr = 0;

    PetscInt nloc;
    VecGetLocalSize(x, &nloc);
    PetscScalar *xv, *Lv, *Uv, **pijv, **qijv;
    VecGetArray(x, &xv);
    VecGetArrays(pij, m, &pijv);
    VecGetArrays(qij, m, &qijv);
    VecGetArray(L, &Lv);
    VecGetArray(U, &Uv);
    for (PetscInt j = 0; j < m; j++) {
        grad[j] = 0.0;
        for (PetscInt i = 0; i < nloc; i++) {
            grad[j] += pijv[j][i] / (Uv[i] - xv[i]) + qijv[j][i] / (xv[i] - Lv[i]);
        }
    }
    {
        PetscScalar* tmp = new PetscScalar[m];
        for (PetscInt i = 0; i < m; i++) {
            tmp[i] = 0.0;
        }
        MPI_Allreduce(grad, tmp, m, MPIU_SCALAR, MPI_SUM, mma_comm);
        memcpy(grad, tmp, sizeof(PetscScalar) * m);
        delete[] tmp;
    }
    for (PetscInt j = 0; j < m; j++) {
        grad[j] += -b[j] - a[j] * z - y[j];
    }
    VecRestoreArray(x, &xv);
    VecRestoreArrays(pij, m, &pijv);
    VecRestoreArrays(qij, m, &qijv);
    VecRestoreArray(L, &Lv);
    VecRestoreArray(U, &Uv);
    return ierr;
}

PetscErrorCode MMA::DualHess(Vec x) {
    PetscErrorCode ierr = 0;

    PetscInt nloc;
    VecGetLocalSize(x, &nloc);
    PetscScalar *xv, *Lv, *Uv, **pijv, **qijv, *alf, *bet, *p0v, *q0v;
    VecGetArray(x, &xv);
    VecGetArrays(pij, m, &pijv);
    VecGetArrays(qij, m, &qijv);
    VecGetArray(L, &Lv);
    VecGetArray(U, &Uv);
    VecGetArray(alpha, &alf);
    VecGetArray(beta, &bet);
    VecGetArray(p0, &p0v);
    VecGetArray(q0, &q0v);
    PetscScalar* df2 = new PetscScalar[nloc];
    PetscScalar* PQ  = new PetscScalar[nloc * m];
    PetscScalar  pjlam, qjlam;
    for (PetscInt i = 0; i < nloc; i++) {
        pjlam = p0v[i];
        qjlam = q0v[i];
        for (PetscInt j = 0; j < m; j++) {
            pjlam += pijv[j][i] * lam[j];
            qjlam += qijv[j][i] * lam[j];
            PQ[i * m + j] = pijv[j][i] / pow(Uv[i] - xv[i], 2.0) - qijv[j][i] / pow(xv[i] - Lv[i], 2.0);
        }
        df2[i]         = -1.0 / (2.0 * pjlam / pow(Uv[i] - xv[i], 3.0) + 2.0 * qjlam / pow(xv[i] - Lv[i], 3.0));
        PetscScalar xp = (sqrt(pjlam) * Lv[i] + sqrt(qjlam) * Uv[i]) / (sqrt(pjlam) + sqrt(qjlam));
        if (xp < alf[i]) {
            df2[i] = 0.0;
        }
        if (xp > bet[i]) {
            df2[i] = 0.0;
        }
    }
    PetscScalar* tmp = new PetscScalar[n * m];
    for (PetscInt j = 0; j < m; j++) {
        for (PetscInt i = 0; i < nloc; i++) {
            tmp[j * nloc + i] = 0.0;
            tmp[j * nloc + i] += PQ[i * m + j] * df2[i];
        }
    }
    for (PetscInt i = 0; i < m; i++) {
        for (PetscInt j = 0; j < m; j++) {
            Hess[i * m + j] = 0.0;
            for (PetscInt k = 0; k < nloc; k++) {
                Hess[i * m + j] += tmp[i * nloc + k] * PQ[k * m + j];
            }
        }
    }
    {
        PetscScalar* tmpp = new PetscScalar[m * m];
        for (PetscInt i = 0; i < m * m; i++) {
            tmpp[i] = Hess[i];
        }
        MPI_Allreduce(Hess, tmpp, m * m, MPIU_SCALAR, MPI_SUM, mma_comm);
        memcpy(Hess, tmpp, sizeof(PetscScalar) * m * m);
        delete[] tmpp;
    }
    PetscScalar lamai = 0.0;
    for (PetscInt j = 0; j < m; j++) {
        if (lam[j] < 0.0) {
            lam[j] = 0.0;
        }
        lamai += lam[j] * a[j];
        if (lam[j] > c[j]) {
            Hess[j * m + j] += -1.0;
        }
        Hess[j * m + j] += -mu[j] / lam[j];
    }
    if (lamai > 0.0) {
        for (PetscInt j = 0; j < m; j++) {
            for (PetscInt k = 0; k < m; k++) {
                Hess[j * m + k] += -10.0 * a[j] * a[k];
            }
        }
    }
    PetscScalar HessTrace = 0.0;
    for (PetscInt i = 0; i < m; i++) {
        HessTrace += Hess[i * m + i];
    }
    PetscScalar HessCorr = 1e-4 * HessTrace / m;
    if (-1.0 * HessCorr < 1.0e-7) {
        HessCorr = -1.0e-7;
    }
    for (PetscInt i = 0; i < m; i++) {
        Hess[i * m + i] += HessCorr;
    }
    VecRestoreArray(x, &xv);
    VecRestoreArrays(pij, m, &pijv);
    VecRestoreArrays(qij, m, &qijv);
    VecRestoreArray(L, &Lv);
    VecRestoreArray(U, &Uv);
    VecRestoreArray(q0, &q0v);
    VecRestoreArray(p0, &p0v);
    VecRestoreArray(alpha, &alf);
    VecRestoreArray(beta, &bet);
    delete[] df2;
    delete[] PQ;
    delete[] tmp;
    return ierr;
}

PetscErrorCode MMA::DualLineSearch() {
    PetscErrorCode ierr = 0;

    PetscScalar theta = 1.005;
    for (PetscInt i = 0; i < m; i++) {
        if (theta < -1.01 * s[i] / lam[i]) {
            theta = -1.01 * s[i] / lam[i];
        }
        if (theta < -1.01 * s[i + m] / mu[i]) {
            theta = -1.01 * s[i + m] / mu[i];
        }
    }
    theta = 1.0 / theta;
    for (PetscInt i = 0; i < m; i++) {
        lam[i] = lam[i] + theta * s[i];
        mu[i]  = mu[i] + theta * s[i + m];
    }
    return ierr;
}

PetscScalar MMA::DualResidual(Vec x, PetscScalar epsi) {

    PetscInt nloc;
    VecGetLocalSize(x, &nloc);
    PetscScalar* res = new PetscScalar[2 * m];
    PetscScalar *xv, *Lv, *Uv, **pijv, **qijv;
    VecGetArray(x, &xv);
    VecGetArrays(pij, m, &pijv);
    VecGetArrays(qij, m, &qijv);
    VecGetArray(L, &Lv);
    VecGetArray(U, &Uv);
    for (PetscInt j = 0; j < m; j++) {
        res[j]     = 0.0;
        res[j + m] = 0.0;
        for (PetscInt i = 0; i < nloc; i++) {
            res[j] += pijv[j][i] / (Uv[i] - xv[i]) + qijv[j][i] / (xv[i] - Lv[i]);
        }
    }
    {
        PetscScalar* tmp = new PetscScalar[2 * m];
        for (PetscInt i = 0; i < 2 * m; i++) {
            tmp[i] = 0.0;
        }
        MPI_Allreduce(res, tmp, 2 * m, MPIU_SCALAR, MPI_SUM, mma_comm);
        memcpy(res, tmp, sizeof(PetscScalar) * 2 * m);
        delete[] tmp;
    }
    for (PetscInt j = 0; j < m; j++) {
        res[j] += -b[j] - a[j] * z - y[j] + mu[j];
        res[j + m] += mu[j] * lam[j] - epsi;
    }
    PetscScalar nrI = 0.0;
    for (PetscInt i = 0; i < 2 * m; i++) {
        if (nrI < Abs(res[i])) {
            nrI = Abs(res[i]);
        }
    }
    delete[] res;
    VecRestoreArray(x, &xv);
    VecRestoreArrays(pij, m, &pijv);
    VecRestoreArrays(qij, m, &qijv);
    VecRestoreArray(L, &Lv);
    VecRestoreArray(U, &Uv);
    return nrI;
}

PetscErrorCode MMA::Factorize(PetscScalar* K, PetscInt nn) {
    PetscErrorCode ierr = 0;

    for (PetscInt ss = 0; ss < nn - 1; ss++) {
        for (PetscInt i = ss + 1; i < nn; i++) {
            K[i * nn + ss] = K[i * nn + ss] / K[ss * nn + ss];
            for (PetscInt j = ss + 1; j < nn; j++) {
                K[i * nn + j] = K[i * nn + j] - K[i * nn + ss] * K[ss * nn + j];
            }
        }
    }
    return ierr;
}

PetscErrorCode MMA::Solve(PetscScalar* K, PetscScalar* x, PetscInt nn) {
    PetscErrorCode ierr = 0;

    for (PetscInt i = 1; i < nn; i++) {
        PetscScalar a = 0.0;
        for (PetscInt j = 0; j < i; j++) {
            a = a - K[i * nn + j] * x[j];
        }
        x[i] = x[i] + a;
    }
    x[nn - 1] = x[nn - 1] / K[(nn - 1) * nn + (nn - 1)];
    for (PetscInt i = nn - 2; i >= 0; i--) {
        PetscScalar a = x[i];
        for (PetscInt j = i + 1; j < nn; j++) {
            a = a - K[i * nn + j] * x[j];
        }
        x[i] = a / K[i * nn + i];
    }
    return ierr;
}

PetscScalar MMA::Min(PetscScalar d1, PetscScalar d2) { return d1 < d2 ? d1 : d2; }

PetscScalar MMA::Max(PetscScalar d1, PetscScalar d2) { return d1 > d2 ? d1 : d2; }

PetscInt MMA::Min(PetscInt d1, PetscInt d2) { return d1 < d2 ? d1 : d2; }

PetscInt MMA::Max(PetscInt d1, PetscInt d2) { return d1 > d2 ? d1 : d2; }

PetscScalar MMA::Abs(PetscScalar d1) { return d1 > 0 ? d1 : -1.0 * d1; }





#endif

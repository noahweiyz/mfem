// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Implementation of class LinearForm

#include "darcyform.hpp"

namespace mfem
{

DarcyForm::DarcyForm(FiniteElementSpace *fes_u_, FiniteElementSpace *fes_p_)
   : fes_u(fes_u_), fes_p(fes_p_)
{
   offsets.SetSize(3);
   offsets[0] = 0;
   offsets[1] = fes_u->GetVSize();
   offsets[2] = fes_p->GetVSize();
   offsets.PartialSum();

   width = height = offsets.Last();

   M_u = NULL;
   M_p = NULL;
   B = NULL;

   mB_e = NULL;
   mBt_e = NULL;

   block_op = new BlockOperator(offsets);
}

BilinearForm* DarcyForm::GetFluxMassForm()
{
   if (!M_u) { M_u = new BilinearForm(fes_u); }
   return M_u;
}

const BilinearForm* DarcyForm::GetFluxMassForm() const
{
   MFEM_ASSERT(M_u, "Flux mass form not allocated!");
   return M_u;
}

BilinearForm* DarcyForm::GetPotentialMassForm()
{
   if (!M_p) { M_p = new BilinearForm(fes_p); }
   return M_p;
}

const BilinearForm* DarcyForm::GetPotentialMassForm() const
{
   MFEM_ASSERT(M_p, "Potential mass form not allocated!");
   return M_p;
}

MixedBilinearForm* DarcyForm::GetFluxDivForm()
{
   if (!B) { B = new MixedBilinearForm(fes_u, fes_p); }
   return B;
}

const MixedBilinearForm* DarcyForm::GetFluxDivForm() const
{
   MFEM_ASSERT(B, "Flux div form not allocated!");
   return B;
}

void DarcyForm::SetAssemblyLevel(AssemblyLevel assembly_level)
{
   assembly = assembly_level;

   if (M_u) { M_u->SetAssemblyLevel(assembly); }
   if (M_p) { M_p->SetAssemblyLevel(assembly); }
   if (B) { B->SetAssemblyLevel(assembly); }
}

void DarcyForm::Assemble(int skip_zeros)
{
   if (M_u) { M_u->Assemble(skip_zeros); }
   if (M_p) { M_p->Assemble(skip_zeros); }
   if (B) { B->Assemble(skip_zeros); }
}

void DarcyForm::Finalize(int skip_zeros)
{
   if (M_u)
   {
      M_u->Finalize(skip_zeros);
      block_op->SetDiagonalBlock(0, M_u);
   }

   if (M_p)
   {
      M_p->Finalize(skip_zeros);
      block_op->SetDiagonalBlock(1, M_p);
   }

   if (B)
   {
      B->Finalize(skip_zeros);

      if (!pBt.Ptr()) { ConstructBT(B); }

      block_op->SetBlock(0, 1, pBt.Ptr(), -1.);
      block_op->SetBlock(1, 0, B, -1.);
   }
}

void DarcyForm::FormLinearSystem(const Array<int> &ess_flux_tdof_list,
                                 const Array<int> &ess_pot_tdof_list,
                                 BlockVector &x, BlockVector &b, OperatorHandle &A, Vector &X_, Vector &B_,
                                 int copy_interior)
{
   FormSystemMatrix(ess_flux_tdof_list, ess_pot_tdof_list, A);

   //conforming
   EliminateVDofsInRHS(ess_flux_tdof_list, ess_pot_tdof_list, x, b);

   // A, X and B point to the same data as mat, x and b
   X_.MakeRef(x, 0, x.Size());
   B_.MakeRef(b, 0, b.Size());
   if (!copy_interior)
   {
      x.GetBlock(0).SetSubVectorComplement(ess_flux_tdof_list, 0.0);
      x.GetBlock(1).SetSubVectorComplement(ess_pot_tdof_list, 0.0);
   }
}

void DarcyForm::FormSystemMatrix(const Array<int> &ess_flux_tdof_list,
                                 const Array<int> &ess_pot_tdof_list, OperatorHandle &A)
{
   if (M_u)
   {
      M_u->FormSystemMatrix(ess_flux_tdof_list, pM_u);
      block_op->SetDiagonalBlock(0, pM_u.Ptr());
   }

   if (M_p)
   {
      M_p->FormSystemMatrix(ess_pot_tdof_list, pM_p);
      block_op->SetDiagonalBlock(1, pM_p.Ptr());
   }

   if (B)
   {
      if (assembly != AssemblyLevel::LEGACY && assembly != AssemblyLevel::FULL)
      {
         B->FormRectangularSystemMatrix(ess_flux_tdof_list, ess_pot_tdof_list, pB);

         ConstructBT(pB.Ptr());
      }
      else
      {
         const SparseMatrix *test_P = fes_p->GetConformingProlongation();
         const SparseMatrix *trial_P = fes_u->GetConformingProlongation();

         B->Finalize();

         if (test_P && trial_P)
         {
            pB.Reset(RAP(*test_P, B->SpMat(), *trial_P));
         }
         else if (test_P)
         {
            pB.Reset(TransposeMult(*test_P, B->SpMat()));
         }
         else if (trial_P)
         {
            pB.Reset(mfem::Mult(B->SpMat(), *trial_P));
         }
         else
         {
            pB.Reset(&B->SpMat(), false);
         }

         Array<int> ess_flux_tdof_marker, ess_pot_tdof_marker;
         FiniteElementSpace::ListToMarker(ess_flux_tdof_list, fes_u->GetTrueVSize(),
                                          ess_flux_tdof_marker);
         FiniteElementSpace::ListToMarker(ess_pot_tdof_list, fes_p->GetTrueVSize(),
                                          ess_pot_tdof_marker);

         if (mB_e) { delete mB_e; }
         mB_e = new SparseMatrix(pB->Height(), pB->Width());
         pB.As<SparseMatrix>()->EliminateCols(ess_flux_tdof_marker, *mB_e);
         mB_e->Finalize();

         pBt.Reset(Transpose(*pB.As<SparseMatrix>()));

         if (mBt_e) { delete mBt_e; }
         mBt_e = new SparseMatrix(pBt->Height(), pBt->Width());
         pBt.As<SparseMatrix>()->EliminateCols(ess_pot_tdof_marker, *mBt_e);
         mBt_e->Finalize();
      }

      block_op->SetBlock(0, 1, pBt.Ptr(), -1.);
      block_op->SetBlock(1, 0, pB.Ptr(), -1.);
   }

   A.Reset(block_op, false);
}

void DarcyForm::RecoverFEMSolution(const Vector &X, const BlockVector &b,
                                   BlockVector &x)
{
   BlockVector X_b(const_cast<Vector&>(X), offsets);
   if (M_u)
   {
      M_u->RecoverFEMSolution(X_b.GetBlock(0), b.GetBlock(0), x.GetBlock(0));
   }
   if (M_p)
   {
      M_p->RecoverFEMSolution(X_b.GetBlock(1), b.GetBlock(1), x.GetBlock(1));
   }
}

void DarcyForm::EliminateVDofsInRHS(const Array<int> &vdofs_flux,
                                    const Array<int> &vdofs_pot, const BlockVector &x, BlockVector &b)
{
   if (B)
   {
      if (assembly != AssemblyLevel::LEGACY && assembly != AssemblyLevel::FULL)
      {
         //TODO
         MFEM_ABORT("");
      }
      else
      {
         mB_e->AddMult(x.GetBlock(0), b.GetBlock(1));
         mBt_e->AddMult(x.GetBlock(1), b.GetBlock(0));
      }
   }
   if (M_u)
   {
      M_u->EliminateVDofsInRHS(vdofs_flux, x.GetBlock(0), b.GetBlock(0));
   }
   if (M_p)
   {
      M_p->EliminateVDofsInRHS(vdofs_pot, x.GetBlock(1), b.GetBlock(1));
   }
}

DarcyForm::~DarcyForm()
{
   if (M_u) { delete M_u; }
   if (M_p) { delete M_p; }
   if (B) { delete B; }

   if (mB_e) { delete mB_e; }
   if (mBt_e) { delete mBt_e; }

   delete block_op;
}

const Operator* DarcyForm::ConstructBT(const MixedBilinearForm *B)
{
   pBt.Reset(Transpose(B->SpMat()));
   return pBt.Ptr();
}

const Operator* DarcyForm::ConstructBT(const Operator *opB)
{
   pBt.Reset(new TransposeOperator(opB));
   return pBt.Ptr();
}

}

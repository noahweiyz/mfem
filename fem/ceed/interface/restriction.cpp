// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "restriction.hpp"

#include "util.hpp"

namespace mfem
{

namespace ceed
{

#ifdef MFEM_USE_CEED

static void InitLexicoRestr(const mfem::FiniteElementSpace &fes,
                            bool use_bdr,
                            Ceed ceed,
                            CeedElemRestriction *restr)
{
   const int nelem = use_bdr ? fes.GetNBE() : fes.GetNE();
   const mfem::FiniteElement *fe = use_bdr ? fes.GetBE(0) :
                                   fes.GetFE(0);
   const int P = fe->GetDof();
   const mfem::TensorBasisElement *tfe =
      dynamic_cast<const mfem::TensorBasisElement *>(fe);
   const mfem::Array<int> &dof_map = tfe->GetDofMap();
   CeedInt compstride =
      (fes.GetOrdering() == Ordering::byVDIM) ? 1 : fes.GetNDofs();
   const int stride = (compstride == 1) ? fes.GetVDim() : 1;
   const mfem::Table &el_dof = use_bdr ? fes.GetBdrElementToDofTable() :
                               fes.GetElementToDofTable();
   const int *el_map = el_dof.GetJ();
   mfem::Array<int> tp_el_dof(el_dof.Size_of_connections());
   mfem::Array<bool> tp_el_orients(el_dof.Size_of_connections());
   bool use_orients = false;

   for (int i = 0; i < nelem; i++)
   {
      // No need to handle DofTransformation for tensor-product elements
      for (int j = 0; j < P; j++)
      {
         const int sdid = dof_map[j];  // signed
         const int did = (sdid >= 0) ? sdid : -1 - sdid;
         const int sgid = el_map[did + P * i];  // signed
         const int gid = (sgid >= 0) ? sgid : -1 - sgid;
         tp_el_dof[j + P * i] = stride * gid;
         tp_el_orients[j + P * i] =
            (sgid >= 0 && sdid < 0) || (sgid < 0 && sdid >= 0);
         use_orients = use_orients || tp_el_orients[j + P * i];
      }
   }

   if (use_orients)
   {
      CeedElemRestrictionCreateOriented(ceed, nelem, P, fes.GetVDim(),
                                        compstride, fes.GetVDim() * fes.GetNDofs(),
                                        CEED_MEM_HOST, CEED_COPY_VALUES,
                                        tp_el_dof.GetData(), tp_el_orients.GetData(),
                                        restr);
   }
   else
   {
      CeedElemRestrictionCreate(ceed, nelem, P, fes.GetVDim(),
                                compstride, fes.GetVDim() * fes.GetNDofs(),
                                CEED_MEM_HOST, CEED_COPY_VALUES,
                                tp_el_dof.GetData(), restr);
   }
}

static void InitNativeRestr(const mfem::FiniteElementSpace &fes,
                            bool use_bdr,
                            Ceed ceed,
                            CeedElemRestriction *restr)
{
   const int nelem = use_bdr ? fes.GetNBE() : fes.GetNE();
   const mfem::FiniteElement *fe = use_bdr ? fes.GetBE(0) :
                                   fes.GetFE(0);
   const int P = fe->GetDof();
   CeedInt compstride =
      (fes.GetOrdering() == Ordering::byVDIM) ? 1 : fes.GetNDofs();
   const int stride = (compstride == 1) ? fes.GetVDim() : 1;
   const mfem::Table &el_dof = use_bdr ? fes.GetBdrElementToDofTable() :
                               fes.GetElementToDofTable();
   const int *el_map = el_dof.GetJ();
   mfem::Array<int> tp_el_dof(el_dof.Size_of_connections());
   mfem::Array<bool> tp_el_orients(el_dof.Size_of_connections());
   bool use_orients = false;

   for (int i = 0; i < nelem; i++)
   {
      // DofTransformation support uses InitNativeRestrWithIndices
      for (int j = 0; j < P; j++)
      {
         const int sgid = el_map[j + P * i];  // signed
         const int gid = (sgid >= 0) ? sgid : -1 - sgid;
         tp_el_dof[j + P * i] = stride * gid;
         tp_el_orients[j + P * i] = (sgid < 0);
         use_orients = use_orients || tp_el_orients[j + P * i];
      }
   }

   if (use_orients)
   {
      CeedElemRestrictionCreateOriented(ceed, nelem, P, fes.GetVDim(),
                                        compstride, fes.GetVDim() * fes.GetNDofs(),
                                        CEED_MEM_HOST, CEED_COPY_VALUES,
                                        tp_el_dof.GetData(), tp_el_orients.GetData(),
                                        restr);
   }
   else
   {
      CeedElemRestrictionCreate(ceed, nelem, P, fes.GetVDim(),
                                compstride, fes.GetVDim() * fes.GetNDofs(),
                                CEED_MEM_HOST, CEED_COPY_VALUES,
                                tp_el_dof.GetData(), restr);
   }
}

static void InitLexicoRestrWithIndices(const mfem::FiniteElementSpace &fes,
                                       bool use_bdr,
                                       int nelem,
                                       const int *indices,
                                       Ceed ceed,
                                       CeedElemRestriction *restr)
{
   const mfem::FiniteElement *fe = use_bdr ? fes.GetBE(indices[0]) :
                                   fes.GetFE(indices[0]);
   const int P = fe->GetDof();
   const mfem::TensorBasisElement *tfe =
      dynamic_cast<const mfem::TensorBasisElement *>(fe);
   const mfem::Array<int> &dof_map = tfe->GetDofMap();
   CeedInt compstride =
      (fes.GetOrdering() == Ordering::byVDIM) ? 1 : fes.GetNDofs();
   const int stride = (compstride == 1) ? fes.GetVDim() : 1;
   mfem::Array<int> tp_el_dof(nelem * P), dofs;
   mfem::Array<bool> tp_el_orients(nelem * P);
   bool use_orients = false;

   for (int i = 0; i < nelem; i++)
   {
      // No need to handle DofTransformation for tensor-product elements
      const int elem_index = indices[i];
      mfem::DofTransformation *dof_trans;
      if (use_bdr)
      {
         dof_trans = fes.GetBdrElementDofs(elem_index, dofs);
      }
      else
      {
         dof_trans = fes.GetElementDofs(elem_index, dofs);
      }
      MFEM_VERIFY(!dof_trans || fes.GetMaxElementOrder() == 1,
                  "DofTransformation support for CeedElemRestriction does not exist yet.");
      for (int j = 0; j < P; j++)
      {
         const int sdid = dof_map[j];  // signed
         const int did = (sdid >= 0) ? sdid : -1 - sdid;
         const int sgid = dofs[did];  // signed
         const int gid = (sgid >= 0) ? sgid : -1 - sgid;
         tp_el_dof[j + P * i] = stride * gid;
         tp_el_orients[j + P * i] =
            (sgid >= 0 && sdid < 0) || (sgid < 0 && sdid >= 0);
         use_orients = use_orients || tp_el_orients[j + P * i];
      }
   }

   if (use_orients)
   {
      CeedElemRestrictionCreateOriented(ceed, nelem, P, fes.GetVDim(),
                                        compstride, fes.GetVDim() * fes.GetNDofs(),
                                        CEED_MEM_HOST, CEED_COPY_VALUES,
                                        tp_el_dof.GetData(), tp_el_orients.GetData(),
                                        restr);
   }
   else
   {
      CeedElemRestrictionCreate(ceed, nelem, P, fes.GetVDim(),
                                compstride, fes.GetVDim() * fes.GetNDofs(),
                                CEED_MEM_HOST, CEED_COPY_VALUES,
                                tp_el_dof.GetData(), restr);
   }
}

static void InitNativeRestrWithIndices(const mfem::FiniteElementSpace &fes,
                                       bool use_bdr,
                                       int nelem,
                                       const int *indices,
                                       Ceed ceed,
                                       CeedElemRestriction *restr)
{
   const int i0 = indices ? indices[0] : 0;
   const mfem::FiniteElement *fe = use_bdr ? fes.GetBE(i0) : fes.GetFE(i0);
   const int P = fe->GetDof();
   CeedInt compstride =
      (fes.GetOrdering() == Ordering::byVDIM) ? 1 : fes.GetNDofs();
   const int stride = (compstride == 1) ? fes.GetVDim() : 1;
   mfem::Array<int> tp_el_dof(nelem * P), dofs;
   mfem::Array<bool> tp_el_orients;
   mfem::Array<int> tp_el_curl_orients;
   mfem::Vector el_trans_j;
   mfem::DofTransformation *dof_trans = use_bdr ? fes.GetBdrElementDofs(i0, dofs) :
                                        fes.GetElementDofs(i0, dofs);
   if (!dof_trans)
   {
      tp_el_orients.SetSize(nelem * P);
   }
   else
   {
      tp_el_curl_orients.SetSize(nelem * P * 3, 0.0);
      el_trans_j.SetSize(P);
   }

   for (int i = 0; i < nelem; i++)
   {
      const int elem_index = indices ? indices[i] : i;
      if (use_bdr)
      {
         dof_trans = fes.GetBdrElementDofs(elem_index, dofs);
      }
      else
      {
         dof_trans = fes.GetElementDofs(elem_index, dofs);
      }
      if (!dof_trans)
      {
         for (int j = 0; j < P; j++)
         {
            const int sgid = dofs[j];  // signed
            const int gid = (sgid >= 0) ? sgid : -1 - sgid;
            tp_el_dof[j + P * i] = stride * gid;
            tp_el_orients[j + P * i] = (sgid < 0);
         }
      }
      else
      {
         for (int j = 0; j < P; j++)
         {
            const int sgid = dofs[j];  // signed
            const int gid = (sgid >= 0) ? sgid : -1 - sgid;
            tp_el_dof[j + P * i] = stride * gid;

            // Fill column j of element tridiagonal matrix tp_el_curl_orients
            el_trans_j = 0.0;
            el_trans_j(j) = 1.0;
            dof_trans->InvTransformPrimal(el_trans_j);
            el_trans_j *= (sgid < 0) ? -1.0 : 1.0;
            tp_el_curl_orients[3 * (j + 0 + P * i) + 1] = el_trans_j(j + 0);
            if (j > 0)
            {
               tp_el_curl_orients[3 * (j - 1 + P * i) + 2] = el_trans_j(j - 1);
            }
            if (j < P - 1)
            {
               tp_el_curl_orients[3 * (j + 1 + P * i) + 0] = el_trans_j(j + 1);
            }
#ifdef MFEM_DEBUG
            int nnz = 0;
            for (int k = 0; k < P; k++)
            {
               if (k < j - 1 && k > j + 1 && el_trans_j(k) != 0.0) { nnz++; }
            }
            MFEM_ASSERT(nnz == 0,
                        "Element transformation matrix is not tridiagonal at column "
                        << j << " (nnz = " << nnz << ")!");
#endif
         }
      }
   }

   if (tp_el_curl_orients.Size())
   {
      CeedElemRestrictionCreateCurlOriented(ceed, nelem, P, fes.GetVDim(),
                                            compstride, fes.GetVDim() * fes.GetNDofs(),
                                            CEED_MEM_HOST, CEED_COPY_VALUES,
                                            tp_el_dof.GetData(), tp_el_curl_orients.GetData(),
                                            restr);
   }
   else if (tp_el_orients.Size())
   {
      CeedElemRestrictionCreateOriented(ceed, nelem, P, fes.GetVDim(),
                                        compstride, fes.GetVDim() * fes.GetNDofs(),
                                        CEED_MEM_HOST, CEED_COPY_VALUES,
                                        tp_el_dof.GetData(), tp_el_orients.GetData(),
                                        restr);
   }
   else
   {
      CeedElemRestrictionCreate(ceed, nelem, P, fes.GetVDim(),
                                compstride, fes.GetVDim() * fes.GetNDofs(),
                                CEED_MEM_HOST, CEED_COPY_VALUES,
                                tp_el_dof.GetData(), restr);
   }
}

static void InitRestrictionImpl(const mfem::FiniteElementSpace &fes,
                                bool use_bdr,
                                Ceed ceed,
                                CeedElemRestriction *restr)
{
   const mfem::FiniteElement *fe = use_bdr ? fes.GetBE(0) : fes.GetFE(0);
   const mfem::TensorBasisElement *tfe =
      dynamic_cast<const mfem::TensorBasisElement *>(fe);
   const bool vector = fe->GetRangeType() == mfem::FiniteElement::VECTOR;
   mfem::Array<int> dofs;
   mfem::DofTransformation *dof_trans = use_bdr ? fes.GetBdrElementDofs(0, dofs) :
                                        fes.GetElementDofs(0, dofs);
   if (tfe && tfe->GetDofMap().Size() > 0 && !vector)
   {
      // Lexicographic ordering using dof_map
      InitLexicoRestr(fes, use_bdr, ceed, restr);
   }
   else
   {
      // Native ordering
      if (!dof_trans)
      {
         InitNativeRestr(fes, use_bdr, ceed, restr);
      }
      else
      {
         InitNativeRestrWithIndices(fes, use_bdr, use_bdr ? fes.GetNBE() : fes.GetNE(),
                                    nullptr, ceed, restr);
      }
   }
}

static void InitRestrictionWithIndicesImpl(const mfem::FiniteElementSpace &fes,
                                           bool use_bdr,
                                           int nelem,
                                           const int *indices,
                                           Ceed ceed,
                                           CeedElemRestriction *restr)
{
   const mfem::FiniteElement *fe = use_bdr ? fes.GetBE(indices[0]) :
                                   fes.GetFE(indices[0]);
   const mfem::TensorBasisElement *tfe =
      dynamic_cast<const mfem::TensorBasisElement *>(fe);
   const bool vector = fe->GetRangeType() == mfem::FiniteElement::VECTOR;
   if (tfe && tfe->GetDofMap().Size() > 0 && !vector)
   {
      // Lexicographic ordering using dof_map
      InitLexicoRestrWithIndices(fes, use_bdr, nelem, indices, ceed, restr);
   }
   else
   {
      // Native ordering
      InitNativeRestrWithIndices(fes, use_bdr, nelem, indices, ceed, restr);
   }
}

void InitRestriction(const FiniteElementSpace &fes,
                     bool use_bdr,
                     Ceed ceed,
                     CeedElemRestriction *restr)
{
   // Check for fes -> restriction in hash table
   // {-1, -1, -1} is unique from CEED_STRIDES_BACKEND for non-strided restrictions
   const int nelem = use_bdr ? fes.GetNBE() : fes.GetNE();
   const mfem::FiniteElement *fe = use_bdr ? fes.GetBE(0) : fes.GetFE(0);
   const int P = fe->GetDof();
   const int ncomp = fes.GetVDim();
   RestrKey restr_key(&fes, nelem, P, ncomp, -1, -1, -1);
   auto restr_itr = mfem::internal::ceed_restr_map.find(restr_key);

   // Init or retrieve key values
   if (restr_itr == mfem::internal::ceed_restr_map.end())
   {
      InitRestrictionImpl(fes, use_bdr, ceed, restr);
      mfem::internal::ceed_restr_map[restr_key] = *restr;
   }
   else
   {
      *restr = restr_itr->second;
   }
}

void InitRestrictionWithIndices(const FiniteElementSpace &fes,
                                bool use_bdr,
                                int nelem,
                                const int *indices,
                                Ceed ceed,
                                CeedElemRestriction *restr)
{
   // Check for fes -> restriction in hash table
   // {-1, -1, -1} is unique from CEED_STRIDES_BACKEND for non-strided restrictions
   const mfem::FiniteElement *fe = use_bdr ? fes.GetBE(indices[0]) :
                                   fes.GetFE(indices[0]);
   const int P = fe->GetDof();
   const int ncomp = fes.GetVDim();
   RestrKey restr_key(&fes, nelem, P, ncomp, -1, -1, -1);
   auto restr_itr = mfem::internal::ceed_restr_map.find(restr_key);

   // Init or retrieve key values
   if (restr_itr == mfem::internal::ceed_restr_map.end())
   {
      InitRestrictionWithIndicesImpl(fes, use_bdr, nelem, indices, ceed, restr);
      mfem::internal::ceed_restr_map[restr_key] = *restr;
   }
   else
   {
      *restr = restr_itr->second;
   }
}

void InitStridedRestriction(const mfem::FiniteElementSpace &fes,
                            CeedInt nelem,
                            CeedInt nqpts,
                            CeedInt qdatasize,
                            const CeedInt strides[3],
                            Ceed ceed,
                            CeedElemRestriction *restr)
{
   // Check for fes -> restriction in hash table
   RestrKey restr_key(&fes, nelem, nqpts, qdatasize,
                      strides[0], strides[1], strides[2]);
   auto restr_itr = mfem::internal::ceed_restr_map.find(restr_key);

   // Init or retrieve key values
   if (restr_itr == mfem::internal::ceed_restr_map.end())
   {
      CeedElemRestrictionCreateStrided(ceed, nelem, nqpts, qdatasize,
                                       nelem * nqpts * qdatasize, strides,
                                       restr);
      mfem::internal::ceed_restr_map[restr_key] = *restr;
   }
   else
   {
      *restr = restr_itr->second;
   }
}

#endif

} // namespace ceed

} // namespace mfem

//                  MFEM Example 18 - Serial/Parallel Shared Code
//                      (Implementation of Time-dependent DG Operator)
//
// This code provide example problems for the Euler equations and implements
// the time-dependent DG operator given by the equation:
//
//            (u_t, v)_T - (F(u), ∇ v)_T + <F̂(u, n), [[v]]>_F = 0.
//
// This operator is designed for explicit time stepping methods. Specifically,
// the function DGHyperbolicConservationLaws::Mult implements the following
// transformation:
//
//                             u ↦ M⁻¹(-DF(u) + NF(u))
//
// where M is the mass matrix, DF is the weak divergence of flux, and NF is the
// interface flux. The inverse of the mass matrix is computed element-wise by
// leveraging the block-diagonal structure of the DG mass matrix. Additionally,
// the flux-related terms are computed using the HyperbolicFormIntegrator.
//
// The maximum characteristic speed is determined for each time step. For more
// details, refer to the documentation of DGHyperbolicConservationLaws::Mult.
//

#include <functional>
#include "mfem.hpp"

namespace mfem
{

/// @brief Time dependent DG operator for hyperbolic conservation laws
class DGHyperbolicConservationLaws : public TimeDependentOperator
{
private:
   const int num_equations; // the number of equations
   const int dim;
   FiniteElementSpace &vfes; // vector finite element space
   // Element integration form. Should contain ComputeFlux
   std::unique_ptr<HyperbolicFormIntegrator> formIntegrator;
   // Base Nonlinear Form
   std::unique_ptr<NonlinearForm> nonlinearForm;
   // element-wise inverse mass matrix
   std::vector<DenseMatrix> invmass; // local scalar inverse mass.
   std::vector<DenseMatrix> weakdiv; // local weakdivergence. Trial space is ByDim.
   // global maximum characteristic speed. Updated by form integrators
   mutable double max_char_speed;
   // auxiliary variable used in Mult
   mutable Vector z;

   // Compute element-wise inverse mass matrix
   void ComputeInvMass();
   // Compute element-wise weak-divergence matrix
   void ComputeWeakDivergence();

public:
   /**
    * @brief Construct a new DGHyperbolicConservationLaws object
    *
    * @param vfes_ vector finite element space. Only tested for DG [Pₚ]ⁿ
    * @param formIntegrator_ integrator (F(u,x), grad v)
    * @param preassembleWeakDivergence preassemble weak divergence for faster assembly
    */
   DGHyperbolicConservationLaws(
      FiniteElementSpace &vfes_,
      std::unique_ptr<HyperbolicFormIntegrator> formIntegrator_,
      bool preassembleWeakDivergence=true);
   /**
    * @brief Apply nonlinear form to obtain M⁻¹(DIVF + JUMP HAT(F))
    *
    * @param x current solution vector
    * @param y resulting dual vector to be used in an EXPLICIT solver
    */
   void Mult(const Vector &x, Vector &y) const override;
   // get global maximum characteristic speed to be used in CFL condition
   // where max_char_speed is updated during Mult.
   double GetMaxCharSpeed() { return max_char_speed; }
   void Update();

};

class InEulerFlux : public FluxFunction
{

public:
   /**
    * @brief Construct a new Euler Flux Function with given
    * spatial dimension
    *
    * @param dim spatial dimension

    */
   InEulerFlux(const int dim)
      : FluxFunction(dim, dim) {}

   /**
    * @brief Compute F(ρ, ρu, E)
    *
    * @param state state (ρ, ρu, E) at current integration point
    * @param Tr current element transformation with integration point
    * @param flux F(ρ, ρu, E) = [ρuᵀ; ρuuᵀ + pI; uᵀ(E + p)]
    * @return double maximum characteristic speed, |u| + √(γp/ρ)
    */
   double ComputeFlux(const Vector &state, ElementTransformation &Tr,
                      DenseMatrix &flux) const override;

   /**
    * @brief Compute normal flux, F(ρ, ρu, E)n
    *
    * @param x x (ρ, ρu, E) at current integration point
    * @param normal normal vector, usually not a unit vector
    * @param Tr current element transformation with integration point
    * @param fluxN F(ρ, ρu, E)n = [ρu⋅n; ρu(u⋅n) + pn; (u⋅n)(E + p)]
    * @return double maximum characteristic speed, |u| + √(γp/ρ)
    */
   double ComputeFluxDotN(const Vector &x, const Vector &normal,
                          FaceElementTransformations &Tr,
                          Vector &fluxN) const override;
};

/// The 2nd order AdamsBashforth Method
class AdamsBashforth : public ODESolver
{
private:
   Vector N_1, N_2, z, x_1, x_2;

public:

   void PreviousStep(Vector &pre_x);

   void Init(TimeDependentOperator &f_) override;

   void Step(Vector &x, double &t, double &dt) override;
};

void AdamsBashforth::Init(TimeDependentOperator &f_)
{
   ODESolver::Init(f_);
   int n = f->Width();
   N_1.SetSize(n, mem_type);
   N_2.SetSize(n, mem_type);
   z.SetSize(n, mem_type);
   x_1.SetSize(n, mem_type);
   x_2.SetSize(n, mem_type);
}

void AdamsBashforth::PreviousStep(Vector &pre_x)
{
   x_2=pre_x;
}

void AdamsBashforth::Step(Vector &x, double &t, double &dt)
{
   x_1=x;
   f->SetTime(t);
   f->Mult(x_2, N_2);
   f->Mult(x_1, N_1);
   //x_1.operator*=(4.0/3.0);
   add(x_1,3.0/2.0*dt,N_1,z);
   //z.Add(-1.0/3.0,x_2);
   z.Add(-1.0/2.0*dt,N_2);
   x=z;
   t += dt;
}

class AvgMassOperator : public Operator
{
private:
   Vector avgM;
public:
   AvgMassOperator(FiniteElementSpace *fes): Operator(1, fes->GetVSize()),
      avgM(fes->GetVSize())
   {
      BilinearForm Mass(fes);
      Vector one_gf(fes->GetVSize());
      one_gf = 1.0;
      Mass.AddDomainIntegrator(new MassIntegrator);
      Mass.Assemble();
      Mass.Mult(one_gf, avgM);
   };
   void Mult(const Vector &x, Vector &y) const override
   {
      y = avgM*x;
   }
   void MultTranspose(const Vector &x, Vector &y) const override
   {
      y = avgM;
      y *= x[0];
   }
};

//////////////////////////////////////////////////////////////////
///        HYPERBOLIC CONSERVATION LAWS IMPLEMENTATION         ///
//////////////////////////////////////////////////////////////////

// Implementation of class DGHyperbolicConservationLaws
DGHyperbolicConservationLaws::DGHyperbolicConservationLaws(
   FiniteElementSpace &vfes_,
   std::unique_ptr<HyperbolicFormIntegrator> formIntegrator_,
   bool preassembleWeakDivergence)
   : TimeDependentOperator(vfes_.GetTrueVSize()),
     num_equations(formIntegrator_->num_equations),
     dim(vfes_.GetMesh()->SpaceDimension()),
     vfes(vfes_),
     formIntegrator(std::move(formIntegrator_)),
     z(vfes_.GetTrueVSize())
{
   // Standard local assembly and inversion for energy mass matrices.
   ComputeInvMass();
#ifndef MFEM_USE_MPI
   nonlinearForm.reset(new NonlinearForm(&vfes));
#else
   ParFiniteElementSpace *pvfes = dynamic_cast<ParFiniteElementSpace *>(&vfes);
   if (pvfes)
   {
      nonlinearForm.reset(new ParNonlinearForm(pvfes));
   }
   else
   {
      nonlinearForm.reset(new NonlinearForm(&vfes));
   }
#endif
   if (preassembleWeakDivergence)
   {
      ComputeWeakDivergence();
   }
   else
   {
      nonlinearForm->AddDomainIntegrator(formIntegrator.get());
   }
   nonlinearForm->AddInteriorFaceIntegrator(formIntegrator.get());
   nonlinearForm->UseExternalIntegrators();

}

void DGHyperbolicConservationLaws::ComputeInvMass()
{
   InverseIntegrator inv_mass(new MassIntegrator());

   invmass.resize(vfes.GetNE());
   for (int i=0; i<vfes.GetNE(); i++)
   {
      int dof = vfes.GetFE(i)->GetDof();
      invmass[i].SetSize(dof);
      inv_mass.AssembleElementMatrix(*vfes.GetFE(i),
                                     *vfes.GetElementTransformation(i), invmass[i]);
   }
}

void DGHyperbolicConservationLaws::ComputeWeakDivergence()
{
   TransposeIntegrator weak_div(new GradientIntegrator());
   DenseMatrix weakdiv_bynodes;

   weakdiv.resize(vfes.GetNE());
   for (int i=0; i<vfes.GetNE(); i++)
   {
      int dof = vfes.GetFE(i)->GetDof();
      weakdiv_bynodes.SetSize(dof, dof*dim);
      weak_div.AssembleElementMatrix2(*vfes.GetFE(i), *vfes.GetFE(i),
                                      *vfes.GetElementTransformation(i), weakdiv_bynodes);
      weakdiv[i].SetSize(dof, dof*dim);
      // Reorder so that trial space is ByDim.
      // This makes applying weak divergence to flux value simpler.
      for (int j=0; j<dof; j++)
      {
         for (int d=0; d<dim; d++)
         {
            weakdiv[i].SetCol(j*dim + d, weakdiv_bynodes.GetColumn(d*dof + j));
         }
      }

   }
}


void DGHyperbolicConservationLaws::Mult(const Vector &x, Vector &y) const
{
   // 0. Reset wavespeed computation before operator application.
   formIntegrator->ResetMaxCharSpeed();
   // 1. Apply Nonlinear form to obtain an axiliary result
   //         z = - <F̂(u_h,n), [[v]]>_e
   //    If weak-divergencee is not preassembled, we also have weak-divergence
   //         z = - <F̂(u_h,n), [[v]]>_e + (F(u_h), ∇v)
   nonlinearForm->Mult(x, z);
   if (!weakdiv.empty()) // if weak divergence is pre-assembled
   {
      // Apply weak divergence to F(u_h), and inverse mass to z_loc + weakdiv_loc
      Vector current_state; // view of current state at a node
      DenseMatrix current_flux; // flux of current state
      DenseMatrix flux; // element flux value. Whose column is ordered by dim.
      DenseMatrix current_xmat; // view of current states in an element, dof x num_eq
      DenseMatrix current_zmat; // view of element auxiliary result, dof x num_eq
      DenseMatrix current_ymat; // view of element result, dof x num_eq
      const FluxFunction &fluxFunction = formIntegrator->GetFluxFunction();
      Array<int> vdofs;
      Vector xval, zval;
      for (int i=0; i<vfes.GetNE(); i++)
      {
         ElementTransformation* Tr = vfes.GetElementTransformation(i);
         int dof = vfes.GetFE(i)->GetDof();
         vfes.GetElementVDofs(i, vdofs);
         x.GetSubVector(vdofs, xval);
         current_xmat.UseExternalData(xval.GetData(), dof, num_equations);
         flux.SetSize(num_equations, dim*dof);
         for (int j=0; j<dof; j++) // compute flux for all nodes in the element
         {
            current_xmat.GetRow(j, current_state);
            current_flux.UseExternalData(flux.GetData() + num_equations*dim*j,
                                         num_equations, dof);
            fluxFunction.ComputeFlux(current_state, *Tr, current_flux);
         }
         // Compute weak-divergence and add it to auxiliary result, z
         // Recalling that weakdiv is reordered by dim, we can apply
         // weak-divergence to the transpose of flux.
         z.GetSubVector(vdofs, zval);
         current_zmat.UseExternalData(zval.GetData(), dof, num_equations);
         mfem::AddMult_a_ABt(1.0, weakdiv[i], flux, current_zmat);
         // Apply inverse mass to auxiliary result to obtain the final result
         current_ymat.SetSize(dof, num_equations);
         mfem::Mult(invmass[i], current_zmat, current_ymat);
         y.SetSubVector(vdofs, current_ymat.GetData());
      }
   }
   else
   {
      // Apply block inverse mass
      Vector zval; // z_loc, dof*num_eq

      DenseMatrix current_zmat; // view of element auxiliary result, dof x num_eq
      DenseMatrix current_ymat; // view of element result, dof x num_eq
      Array<int> vdofs;
      for (int i=0; i<vfes.GetNE(); i++)
      {
         int dof = vfes.GetFE(i)->GetDof();
         vfes.GetElementVDofs(i, vdofs);
         z.GetSubVector(vdofs, zval);
         current_zmat.UseExternalData(zval.GetData(), dof, num_equations);
         current_ymat.SetSize(dof, num_equations);
         mfem::Mult(invmass[i], current_zmat, current_ymat);
         y.SetSubVector(vdofs, current_ymat.GetData());
      }
   }
   max_char_speed = formIntegrator->GetMaxCharSpeed();
}

void DGHyperbolicConservationLaws::Update()
{
   nonlinearForm->Update();
   height = nonlinearForm->Height();
   width = height;
   z.SetSize(height);

   ComputeInvMass();
   if (!weakdiv.empty()) {ComputeWeakDivergence();}
}

std::function<void(const Vector&, Vector&)> GetMovingVortexInit(
   const double radius, const double Minf, const double beta,
   const double gas_constant, const double specific_heat_ratio)
{
   return [specific_heat_ratio,
           gas_constant, Minf, radius, beta](const Vector &x, Vector &y)
   {
      MFEM_ASSERT(x.Size() == 2, "");

      const double xc = 0.2, yc = 0.2;

      // Nice units
      const double vel_inf = 1.;
      const double den_inf = 1.;

      // Derive remainder of background state from this and Minf
      const double pres_inf = (den_inf / specific_heat_ratio) *
                              (vel_inf / Minf) * (vel_inf / Minf);
      const double temp_inf = pres_inf / (den_inf * gas_constant);

      double r2rad = 0.0;
      double r2rad2= 0.0;
      r2rad += (x(0) - xc) * (x(0) - xc);
      r2rad += (x(1) - yc) * (x(1) - yc);
      r2rad /= (radius * radius);
      r2rad2 += (x(0) + xc) * (x(0) + xc);
      r2rad2 += (x(1) + yc) * (x(1) + yc);
      r2rad2 /= (radius * radius);

      const double shrinv1 = 1.0 / (specific_heat_ratio - 1.);

      const double velX =
         vel_inf * (1 - beta * (x(1) - yc) / radius * exp(-0.5 * r2rad));
      const double velY =
         vel_inf * beta * (x(0) - xc) / radius * exp(-0.5 * r2rad);
      const double vel2 = velX * velX + velY * velY;

      const double specific_heat =
         gas_constant * specific_heat_ratio * shrinv1;
      const double temp = temp_inf - 0.5 * (vel_inf * beta) *
                          (vel_inf * beta) / specific_heat *
                          exp(-r2rad);

      const double den = den_inf * pow(temp / temp_inf, shrinv1);
      const double pres = den * gas_constant * temp;
      const double energy = shrinv1 * pres / den + 0.5 * vel2;

      y(0) = -(x(1)-0.5)*exp(-0.5*r2rad)-(x(1)+0.5)*exp(-0.5*r2rad2);
      y(1) = (x(0)-0.5)*exp(-0.5*r2rad)+(x(0)+0.5)*exp(-0.5*r2rad2);
   };
}

Mesh EulerMesh(const int problem)
{
   switch (problem)
   {
      case 1:
      case 2:
      case 3:
         return Mesh("../data/periodic-square.mesh");
         break;
      case 4:
         return Mesh("../data/periodic-segment.mesh");
         break;
      default:
         MFEM_ABORT("Problem Undefined");
   }
}

// Initial condition
VectorFunctionCoefficient EulerInitialCondition(const int problem,
                                                const double specific_heat_ratio,
                                                const double gas_constant)
{
   switch (problem)
   {
      case 1: // fast moving vortex
         return VectorFunctionCoefficient(
                   2, GetMovingVortexInit(0.1, 0.5, 1. / 5., gas_constant,
                                          specific_heat_ratio));
      case 2: // slow moving vortex
         return VectorFunctionCoefficient(
                   4, GetMovingVortexInit(0.2, 0.05, 1. / 50., gas_constant,
                                          specific_heat_ratio));
      case 3: // moving sine wave
         return VectorFunctionCoefficient(4, [](const Vector &x, Vector &y)
         {
            MFEM_ASSERT(x.Size() == 2, "");
            const double density = 1.0 + 0.2 * sin(M_PI*(x(0) + x(1)));
            const double velocity_x = 0.7;
            const double velocity_y = 0.3;
            const double pressure = 1.0;
            const double energy =
               pressure / (1.4 - 1.0) +
               density * 0.5 * (velocity_x * velocity_x + velocity_y * velocity_y);

            y(0) = density;
            y(1) = density * velocity_x;
            y(2) = density * velocity_y;
            y(3) = energy;
         });
      case 4:
         return VectorFunctionCoefficient(3, [](const Vector &x, Vector &y)
         {
            MFEM_ASSERT(x.Size() == 1, "");
            const double density = 1.0 + 0.2 * sin(M_PI * 2 * x(0));
            const double velocity_x = 1.0;
            const double pressure = 1.0;
            const double energy =
               pressure / (1.4 - 1.0) + density * 0.5 * (velocity_x * velocity_x);

            y(0) = density;
            y(1) = density * velocity_x;
            y(2) = energy;
         });
      default:
         MFEM_ABORT("Problem Undefined");
   }
}

double InEulerFlux::ComputeFlux(const Vector &U,
                                ElementTransformation &Tr,
                                DenseMatrix &FU) const
{

   // 2. Compute Flux
   for (int d = 0; d < dim; d++)
   {
      for (int i = 0; i < dim; i++)
      {
         FU(i,d)=U(i)*U(d);
      }
   }

   // 3. Compute maximum characteristic speed

   // sound speed, √(γ p / ρ)
   const double speed = std::sqrt(U*U);
   // max characteristic speed = fluid speed
   return speed;
}


double InEulerFlux::ComputeFluxDotN(const Vector &x,
                                    const Vector &normal,
                                    FaceElementTransformations &Tr,
                                    Vector &FUdotN) const
{


   // 2. Compute normal flux

   for (int d = 0; d < dim; d++)
   {
      // (ρuuᵀ + pI)n = ρu*(u⋅n) + pn
      FUdotN(d) = (x * normal) * x(d);
   }

   // 3. Compute maximum characteristic speed
   // fluid speed |u|
   const double speed = std::fabs(x * normal) / std::sqrt(normal*normal);
   // max characteristic speed = fluid speed + sound speed
   return speed;
}
} // namespace mfem

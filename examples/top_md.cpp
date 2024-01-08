//                              MFEM Example 37
//
//
// Compile with: make ex37
//
// Sample runs:
//     ex37 -alpha 10
//     ex37 -alpha 10 -pv
//     ex37 -lambda 0.1 -mu 0.1
//     ex37 -o 2 -alpha 5.0 -mi 50 -vf 0.4 -ntol 1e-5
//     ex37 -r 6 -o 1 -alpha 25.0 -epsilon 0.02 -mi 50 -ntol 1e-5
//
//
// Description: This example code demonstrates the use of MFEM to solve a
//              density-filtered [3] topology optimization problem. The
//              objective is to minimize the compliance
//
//                  minimize ∫_Ω f⋅u dx over u ∈ [H¹(Ω)]² and ρ ∈ L¹(Ω)
//
//                  subject to
//
//                    -Div(r(ρ̃)Cε(u)) = f       in Ω + BCs
//                    -ϵ²Δρ̃ + ρ̃ = ρ             in Ω + Neumann BCs
//                    0 ≤ ρ ≤ 1                 in Ω
//                    ∫_Ω ρ dx = θ vol(Ω)
//
//              Here, r(ρ̃) = ρ₀ + ρ̃³ (1-ρ₀) is the solid isotropic material
//              penalization (SIMP) law, C is the elasticity tensor for an
//              isotropic linearly elastic material, ϵ > 0 is the design
//              length scale, and 0 < θ < 1 is the volume fraction.
//
//              The problem is discretized and gradients are computing using
//              finite elements [1]. The design is optimized using an entropic
//              mirror descent algorithm introduced by Keith and Surowiec [2]
//              that is tailored to the bound constraint 0 ≤ ρ ≤ 1.
//
//              This example highlights the ability of MFEM to deliver high-
//              order solutions to inverse design problems and showcases how
//              to set up and solve PDE-constrained optimization problems
//              using the so-called reduced space approach.
//
//
// [1] Andreassen, E., Clausen, A., Schevenels, M., Lazarov, B. S., & Sigmund, O.
//    (2011). Efficient topology optimization in MATLAB using 88 lines of
//    code. Structural and Multidisciplinary Optimization, 43(1), 1-16.
// [2] Keith, B. and Surowiec, T. (2023) Proximal Galerkin: A structure-
//     preserving finite element method for pointwise bound constraints.
//     arXiv:2307.12444 [math.NA]
// [3] Lazarov, B. S., & Sigmund, O. (2011). Filters in topology optimization
//     based on Helmholtz‐type differential equations. International Journal
//     for Numerical Methods in Engineering, 86(6), 765-781.

#include "mfem.hpp"
#include <iostream>
#include <fstream>
#include "ex37.hpp"

using namespace std;
using namespace mfem;

/**
 * ---------------------------------------------------------------
 *                      ALGORITHM PREAMBLE
 * ---------------------------------------------------------------
 *
 *  The Lagrangian for this problem is
 *
 *          L(u,ρ,ρ̃,w,w̃) = (f,u) - (r(ρ̃) C ε(u),ε(w)) + (f,w)
 *                       - (ϵ² ∇ρ̃,∇w̃) - (ρ̃,w̃) + (ρ,w̃)
 *
 *  where
 *
 *    r(ρ̃) = ρ₀ + ρ̃³ (1 - ρ₀)       (SIMP rule)
 *
 *    ε(u) = (∇u + ∇uᵀ)/2           (symmetric gradient)
 *
 *    C e = λtr(e)I + 2μe           (isotropic material)
 *
 *  NOTE: The Lame parameters can be computed from Young's modulus E
 *        and Poisson's ratio ν as follows:
 *
 *             λ = E ν/((1+ν)(1-2ν)),      μ = E/(2(1+ν))
 *
 * ---------------------------------------------------------------
 *
 *  Discretization choices:
 *
 *     u ∈ V ⊂ (H¹)ᵈ (order p)
 *     ψ ∈ L² (order p - 1), ρ = sigmoid(ψ)
 *     ρ̃ ∈ H¹ (order p)
 *     w ∈ V  (order p)
 *     w̃ ∈ H¹ (order p)
 *
 * ---------------------------------------------------------------
 *                          ALGORITHM
 * ---------------------------------------------------------------
 *
 *  Update ρ with projected mirror descent via the following algorithm.
 *
 *  1. Initialize ψ = inv_sigmoid(vol_fraction) so that ∫ sigmoid(ψ) = θ vol(Ω)
 *
 *  While not converged:
 *
 *     2. Solve filter equation ∂_w̃ L = 0; i.e.,
 *
 *           (ϵ² ∇ ρ̃, ∇ v ) + (ρ̃,v) = (ρ,v)   ∀ v ∈ H¹.
 *
 *     3. Solve primal problem ∂_w L = 0; i.e.,
 *
 *      (λ r(ρ̃) ∇⋅u, ∇⋅v) + (2 μ r(ρ̃) ε(u), ε(v)) = (f,v)   ∀ v ∈ V.
 *
 *     NB. The dual problem ∂_u L = 0 is the negative of the primal problem due to symmetry.
 *
 *     4. Solve for filtered gradient ∂_ρ̃ L = 0; i.e.,
 *
 *      (ϵ² ∇ w̃ , ∇ v ) + (w̃ ,v) = (-r'(ρ̃) ( λ |∇⋅u|² + 2 μ |ε(u)|²),v)   ∀ v ∈ H¹.
 *
 *     5. Project the gradient onto the discrete latent space; i.e., solve
 *
 *                         (G,v) = (w̃,v)   ∀ v ∈ L².
 *
 *     6. Bregman proximal gradient update; i.e.,
 *
 *                            ψ ← ψ - αG + c,
 *
 *     where α > 0 is a step size parameter and c ∈ R is a constant ensuring
 *
 *                     ∫_Ω sigmoid(ψ - αG + c) dx = θ vol(Ω).
 *
 *  end
 *
 */


enum LineSearchMethod
{
   ArmijoBackTracking,
   BregmanBBBackTracking
};

enum Problem
{
   Cantilever,
   MBB,
   LBracket,
   Cantilever3
};

double checkNormalConeL2(GridFunction &psi, GridFunction &grad,
                         double target_volume)
{
   MappedGridFunctionCoefficient rho_cf(&psi, sigmoid);
   GridFunctionCoefficient grad_cf(&grad);
   ConstantCoefficient mu(0.0);
   const double c = 1;
   SumCoefficient grad_cf_minus_mu(grad_cf, mu, c, 1);
   double mu_l = -1.0 - c*grad.Normlinf();
   double mu_r =  1.0 + c*grad.Normlinf();
   TransformedCoefficient projectedDensity(&rho_cf, &grad_cf_minus_mu, [](double p,
                                                                          double g)
   {
      return max(0.0, min(1.0, p - g));
   });
   GridFunction zero_gf(psi);
   zero_gf = 0.0;
   double volume = 0;
   while (fabs(volume - target_volume) > 1e-09 / target_volume & fabs(
             mu_r - mu_l) > 1e-10)
   {
      mu.constant = 0.5*(mu_l + mu_r);
      volume = zero_gf.ComputeL1Error(projectedDensity);
      if (volume < target_volume)
      {
         mu_r = mu.constant;
      }
      else
      {
         mu_l = mu.constant;
      }
   }
   out << ", " << zero_gf.ComputeL1Error(projectedDensity) << flush;

   TransformedCoefficient rho_diff(&rho_cf, &projectedDensity, [](double x,
                                                                  double y)
   {
      return x - y;
   });
   return zero_gf.ComputeL2Error(rho_diff);
}

double checkNormalConeBregman(GridFunction &psi, GridFunction &grad,
                              double target_volume)
{
   GridFunctionCoefficient psi_cf(&psi);
   GridFunctionCoefficient grad_cf(&grad);
   ConstantCoefficient mu(0.0);
   const double c = 1;
   SumCoefficient grad_cf_minus_mu(grad_cf, mu, c, 1);
   double mu_l = -1.0 - c*grad.Normlinf();
   double mu_r =  1.0 + c*grad.Normlinf();
   TransformedCoefficient projectedDensity(&psi_cf, &grad_cf_minus_mu, [](double p,
                                                                          double g)
   {
      return sigmoid(p - g);
   });
   GridFunction zero_gf(psi);
   zero_gf = 0.0;
   while (mu_r - mu_l > 1e-12)
   {
      mu.constant = 0.5*(mu_l + mu_r);
      double volume = zero_gf.ComputeL1Error(projectedDensity);
      if (volume < target_volume)
      {
         mu_r = mu.constant;
      }
      else
      {
         mu_l = mu.constant;
      }
   }
   TransformedCoefficient rho_diff(&psi_cf, &projectedDensity, [](double x,
                                                                  double y)
   {
      return sigmoid(x) - y;
   });
   return zero_gf.ComputeL2Error(rho_diff);
}

int main(int argc, char *argv[])
{

   // 1. Parse command-line options.
   int ref_levels = 7;
   int order = 1;
   double alpha = 1.0;
   double epsilon = 2e-2;
   double vol_fraction = 0.5;
   int max_it = 2e2;
   double itol = 1e-3;
   double ntol = 1e-6;
   double rho_min = 1e-6;
   double exponent = 3.0;
   double lambda = 1.0;
   double mu = 1.0;
   double c1 = 1e-04;
   bool glvis_visualization = true;
   bool save = true;
   bool paraview = true;

   ostringstream solfile, solfile2, meshfile;

   int lineSearchMethod = LineSearchMethod::BregmanBBBackTracking;
   int problem = Problem::Cantilever;

   OptionsParser args(argc, argv);
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem number: 0) Cantilever, 1) MBB, 2) LBracket.");
   args.AddOption(&lineSearchMethod, "-lm", "--line-method",
                  "Line Search Method: 0) BackTracking, 1) BregmanBB, 2) BregmanBB + BackTracking.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&alpha, "-alpha", "--alpha-step-length",
                  "Step length for gradient descent.");
   args.AddOption(&epsilon, "-epsilon", "--epsilon-thickness",
                  "Length scale for ρ.");
   args.AddOption(&max_it, "-mi", "--max-it",
                  "Maximum number of gradient descent iterations.");
   args.AddOption(&ntol, "-ntol", "--rel-tol",
                  "Normalized exit tolerance.");
   args.AddOption(&itol, "-itol", "--abs-tol",
                  "Increment exit tolerance.");
   args.AddOption(&vol_fraction, "-vf", "--volume-fraction",
                  "Volume fraction for the material density.");
   args.AddOption(&lambda, "-lambda", "--lambda",
                  "Lamé constant λ.");
   args.AddOption(&mu, "-mu", "--mu",
                  "Lamé constant μ.");
   args.AddOption(&rho_min, "-rmin", "--psi-min",
                  "Minimum of density coefficient.");
   args.AddOption(&glvis_visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(mfem::out);
      return 1;
   }
   args.PrintOptions(mfem::out);

   Mesh mesh;
   Array2D<int> ess_bdr;
   Array<int> ess_bdr_filter;
   Vector center(2), force(2);
   double r = 0.05;
   VolumeForceCoefficient vforce_cf(r,center,force);
   string mesh_file;
   switch (problem)
   {
      case Problem::Cantilever:
         mesh = mesh.MakeCartesian2D(3, 1, mfem::Element::Type::QUADRILATERAL, true, 3.0,
                                     1.0);
         ess_bdr.SetSize(3, 4);
         ess_bdr_filter.SetSize(4);
         ess_bdr = 0; ess_bdr_filter = 0;
         ess_bdr(2, 3) = 1;
         center(0) = 2.9; center(1) = 0.5;
         force(0) = 0.0; force(1) = -1.0;
         solfile << "Cantilever-";
         solfile2 << "Cantilever-";
         meshfile << "Cantilever";
         break;
      case Problem::LBracket:
         mesh_file = "../data/lbracket_square.mesh";
         mesh = mesh.LoadFromFile(mesh_file);
         ess_bdr.SetSize(3, 6);
         ess_bdr_filter.SetSize(6);
         ess_bdr = 0; ess_bdr_filter = 0;
         ess_bdr(2, 4) = 1;
         center(0) = 0.95; center(1) = 0.35;
         force(0) = 0.0; force(1) = -1.0;
         solfile << "LBracket-";
         solfile2 << "LBracket-";
         meshfile << "LBracket";
         break;
      case Problem::MBB:
         mesh = mesh.MakeCartesian2D(3, 1, mfem::Element::Type::QUADRILATERAL, true, 3.0,
                                     1.0);
         ess_bdr.SetSize(3, 3);
         ess_bdr_filter.SetSize(4);
         ess_bdr = 0; ess_bdr_filter = 0;
         ess_bdr(0, 0) = 1;
         ess_bdr(1, 1) = 1;
         center(0) = 0.05; center(1) = 0.95;
         force(0) = 0.0; force(1) = -1.0;
         solfile << "MBB-";
         solfile2 << "MBB-";
         meshfile << "MBB";
         break;

      case Problem::Cantilever3:
         mesh = mesh.MakeCartesian3D(4, 1, 2, mfem::Element::Type::HEXAHEDRON, 2.0, 0.25,
                                     0.5);
         ess_bdr.SetSize(4, 7);
         ess_bdr_filter.SetSize(7);
         ess_bdr = 0; ess_bdr_filter = 0;
         ess_bdr(3, 4) = 1;
         center.SetSize(3); force.SetSize(3);
         center(0) = 1.9; center(1) = 0.125; center(2) = 0.25;
         force(0) = 0.0; force(1) = 0.0; force(1) = -1.0;
         vforce_cf.UpdateSize();
         solfile << "Cantilever-";
         solfile2 << "Cantilever-";
         meshfile << "Cantilever";
         break;
      default:
         mfem_error("Undefined problem.");
   }

   int dim = mesh.Dimension();
   double h = std::pow(mesh.GetElementVolume(0), 1.0 / dim);

   // 3. Refine the mesh.
   if (problem == Problem::LBracket) {ref_levels--;}
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh.UniformRefinement();
      h *= 0.5;
   }

   if (problem == Problem::MBB)
   {

      for (int i = 0; i<mesh.GetNBE(); i++)
      {
         Element * be = mesh.GetBdrElement(i);
         Array<int> vertices;
         be->GetVertices(vertices);

         double * coords1 = mesh.GetVertex(vertices[0]);
         double * coords2 = mesh.GetVertex(vertices[1]);

         Vector fc(2);
         fc(0) = 0.5*(coords1[0] + coords2[0]);
         fc(1) = 0.5*(coords1[1] + coords2[1]);

         if (abs(fc(0) - 0.0) < 1e-10)
         {
            // the left edge
            be->SetAttribute(1);
         }
         else if ((fc(0) > (3 - std::pow(2, -ref_levels + 1))) & (fc(1) < 1e-10))
         {
            // all other boundaries
            be->SetAttribute(2);
         }
         else
         {
            be->SetAttribute(3);
         }
      }
   }

   // 4. Define the necessary finite element spaces on the mesh.
   H1_FECollection state_fec(order, dim); // space for u
   H1_FECollection filter_fec(order, dim); // space for ρ̃
   L2_FECollection control_fec(order-1, dim,
                               BasisType::GaussLobatto); // space for ψ
   FiniteElementSpace state_fes(&mesh, &state_fec,dim);
   FiniteElementSpace filter_fes(&mesh, &filter_fec);
   FiniteElementSpace control_fes(&mesh, &control_fec);

   int state_size = state_fes.GetTrueVSize();
   int control_size = control_fes.GetTrueVSize();
   int filter_size = filter_fes.GetTrueVSize();
   mfem::out << "Number of state unknowns: " << state_size << std::endl;
   mfem::out << "Number of filter unknowns: " << filter_size << std::endl;
   mfem::out << "Number of control unknowns: " << control_size << std::endl;

   // 5. Set the initial guess for ρ.
   GridFunction psi(&control_fes);
   GridFunction psi_old(&control_fes);
   psi = inv_sigmoid(vol_fraction);
   psi_old = inv_sigmoid(vol_fraction);

   // ρ = sigmoid(ψ)
   MappedGridFunctionCoefficient rho(&psi, sigmoid);
   // Interpolation of ρ = sigmoid(ψ) in control fes (for ParaView output)
   GridFunction rho_gf(&control_fes);
   // ρ - ρ_old = sigmoid(ψ) - sigmoid(ψ_old)
   DiffMappedGridFunctionCoefficient succ_diff_rho(&psi, &psi_old, sigmoid);
   LinearForm succ_diff_rho_form(&control_fes);
   succ_diff_rho_form.AddDomainIntegrator(new DomainLFIntegrator(succ_diff_rho));

   // 9. Define some tools for later
   ConstantCoefficient one(1.0);
   GridFunction zero_gf(&control_fes);
   zero_gf = 0.0;
   LinearForm vol_form(&control_fes);
   vol_form.AddDomainIntegrator(new DomainLFIntegrator(one));
   vol_form.Assemble();
   double domain_volume = vol_form.Sum();
   const double target_volume = domain_volume * vol_fraction;
   ConstantCoefficient lambda_cf(lambda), mu_cf(mu);
   SIMPElasticCompliance obj(&lambda_cf, &mu_cf, epsilon,
                             &rho, &vforce_cf, target_volume,
                             ess_bdr,
                             &state_fes,
                             &filter_fes, exponent, rho_min);
   obj.SetGridFunction(&psi);
   LineSearchAlgorithm *lineSearch;
   switch (lineSearchMethod)
   {
      case LineSearchMethod::ArmijoBackTracking:
         lineSearch = new BackTracking(obj, succ_diff_rho_form, psi_old,
                                       alpha, 2.0, c1, 10, infinity());
         break;
      case LineSearchMethod::BregmanBBBackTracking:
         lineSearch = new BackTrackingLipschitzBregmanMirror(
            obj, succ_diff_rho_form, *(obj.Gradient()), psi, psi_old, c1, 1.0, 1e-10,
            infinity());
         break;
      default:
         mfem_error("Undefined linesearch method.");
   }
   meshfile << "-" << ref_levels;
   solfile << ref_levels << "-";
   solfile2 << ref_levels << "-";
   solfile << "PMD-0.gf";
   solfile2 << "PMD-f.gf";
   meshfile << ".mesh";

   MappedGridFunctionCoefficient &designDensity = obj.GetDesignDensity();
   GridFunction designDensity_gf(&filter_fes);
   designDensity_gf = pow(vol_fraction, exponent);
   // designDensity_gf.ProjectCoefficient(designDensity);

   GridFunction &u = *obj.GetDisplacement();
   GridFunction &rho_filter = *obj.GetFilteredDensity();

   // 10. Connect to GLVis. Prepare for VisIt output.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sout_SIMP, sout_r;
   if (glvis_visualization)
   {
      sout_SIMP.open(vishost, visport);
      sout_SIMP.precision(8);
      sout_SIMP << "solution\n" << mesh << designDensity_gf
                << "window_title 'Design density r(ρ̃) - PMD "
                << problem << " " << lineSearchMethod << "'\n"
                << "keys Rjl***************\n"
                << flush;
      sout_r.open(vishost, visport);
      sout_r.precision(8);
      sout_r << "solution\n" << mesh << rho_gf
             << "window_title 'Raw density ρ - PMD "
             << problem << " " << lineSearchMethod << "'\n"
             << "keys Rjl***************\n"
             << flush;
   }
   ParaViewDataCollection *pd = NULL;
   if (paraview)
   {
      pd = new ParaViewDataCollection("TopPMD", &mesh);
      pd->SetPrefixPath("ParaView");
      pd->RegisterField("state", &u);
      pd->RegisterField("psi", &psi);
      pd->RegisterField("frho", &rho_filter);
      pd->SetLevelsOfDetail(order);
      pd->SetDataFormat(VTKFormat::BINARY);
      pd->SetHighOrderOutput(true);
      pd->SetCycle(0);
      pd->SetTime(0);
      pd->Save();
   }

   mfem::ParaViewDataCollection paraview_dc("ex37", &mesh);
   ofstream mesh_ofs(meshfile.str().c_str());
   mesh_ofs.precision(8);
   mesh.Print(mesh_ofs);
   // 11. Iterate
   obj.Eval();
   GridFunction d(&control_fes);
   int k;
   for (k = 1; k <= max_it; k++)
   {
      // mfem::out << "\nStep = " << k << std::endl;

      d = *obj.Gradient();
      d.Neg();
      double compliance = lineSearch->Step(psi, d);
      double norm_increment = zero_gf.ComputeLpError(1, succ_diff_rho);
      double normal_cone_L2 = checkNormalConeL2(psi, *obj.Gradient(), target_volume);
      double normal_cone_BD = checkNormalConeBregman(psi, *obj.Gradient(),
                                                     target_volume);
      psi_old = psi;
      mfem::out <<  ", " << compliance << ", ";
      mfem::out << norm_increment << ", " << normal_cone_L2 << ", " << normal_cone_BD
                << std::endl;

      if (glvis_visualization)
      {
         designDensity_gf.ProjectCoefficient(designDensity);
         sout_SIMP << "solution\n" << mesh << designDensity_gf
                   << flush;
         rho_gf.ProjectCoefficient(rho);
         sout_r << "solution\n" << mesh << rho_gf
                << flush;
      }
      if (paraview)
      {
         pd->SetCycle(k);
         pd->SetTime(k);
         pd->Save();
      }
      if (normal_cone_L2 < 5e-05)
      {
         break;
      }
   }
   if (save)
   {
      ofstream sol_ofs(solfile.str().c_str());
      sol_ofs.precision(8);
      sol_ofs << psi;

      ofstream sol_ofs2(solfile2.str().c_str());
      sol_ofs2.precision(8);
      sol_ofs2 << *obj.GetFilteredDensity();
   }
   out << "Total number of iteration = " << k << std::endl;
   delete lineSearch;
   delete pd;

   return 0;
}

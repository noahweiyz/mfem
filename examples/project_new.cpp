//                                MFEM Example 18
//
// Compile with: make ex18
//
// Sample runs:
//
//       ex18 -p 1 -r 2 -o 1 -s 3
//       ex18 -p 1 -r 1 -o 3 -s 4
//       ex18 -p 1 -r 0 -o 5 -s 6
//       ex18 -p 2 -r 1 -o 1 -s 3 -mf
//       ex18 -p 2 -r 0 -o 3 -s 3 -mf
//
// Description:  This example code solves the compressible Euler system of
//               equations, a model nonlinear hyperbolic PDE, with a
//               discontinuous Galerkin (DG) formulation.
//
//                (u_t, v)_T - (F(u), ∇ v)_T + <F̂(u,n), [[v]]>_F = 0
//
//               where (⋅,⋅)_T is volume integration, and <⋅,⋅>_F is face
//               integration, F is the Euler flux function, and F̂ is the
//               numerical flux.
//
//               Specifically, it solves for an exact solution of the equations
//               whereby a vortex is transported by a uniform flow. Since all
//               boundaries are periodic here, the method's accuracy can be
//               assessed by measuring the difference between the solution and
//               the initial condition at a later time when the vortex returns
//               to its initial location.
//
//               Note that as the order of the spatial discretization increases,
//               the timestep must become smaller. This example currently uses a
//               simple estimate derived by Cockburn and Shu for the 1D RKDG
//               method. An additional factor can be tuned by passing the --cfl
//               (or -c shorter) flag.
//
//               The example demonstrates usage of DGHyperbolicConservationLaws
//               that wraps NonlinearFormIntegrators containing element and face
//               integration schemes. In this case the system also involves an
//               external approximate Riemann solver for the DG interface flux.
//               By default, weak-divergence is pre-assembled in element-wise
//               manner, which corresponds to (I_h(F(u_h)), ∇ v). This yields
//               better performance and similar accuracy for the included test
//               problems. This can be turned off and use nonlinear assembly
//               similar to matrix-free assembly when -mf flag is provided.
//               It also demonstrates how to use GLVis for in-situ visualization
//               of vector grid function and how to set top-view.
//
//               We recommend viewing examples 9, 14 and 17 before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include "project_new.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int problem = 1;
   const double specific_heat_ratio = 1.4;
   const double gas_constant = 1.0;
   const double nu=0.0001;

   const double sigma = -1.0;
   double kappa = -1.0;
   double eta = 0.0;

   string mesh_file = "";
   int IntOrderOffset = 1;
   int ref_levels = 1;
   int order = 3;
   int ode_solver_type = 4;
   double t_final = 100.0;
   double dt = -0.01;
   double cfl = 0.3;
   bool visualization = true;
   bool preassembleWeakDiv = true;
   int vis_steps = 1;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use. If not provided, then a periodic square"
                  " mesh will be used.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use. See, EulerInitialCondition.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6.");
   args.AddOption(&t_final, "-tf", "--t-final", "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step. Positive number skips CFL timestep calculation.");
   args.AddOption(&cfl, "-c", "--cfl-number",
                  "CFL number for timestep calculation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&preassembleWeakDiv, "-ea", "--element-assembly-divergence",
                  "-mf", "--matrix-free-divergence",
                  "Weak divergence assembly level\n"
                  "    ea - Element assembly with interpolated F\n"
                  "    mf - Nonlinear assembly in matrix-free manner");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.ParseCheck();
   // set penalty constant if not provided.
   if (kappa < 0)
   {
      kappa = (order+1)*(order+1);
   }

   // 2. Read the mesh from the given mesh file. When the user does not provide
   //    mesh file, use the default mesh file for the problem.
   Mesh mesh = mesh_file.empty() ? EulerMesh(problem) : Mesh(mesh_file);
   const int dim = mesh.Dimension();
   const int num_equations = dim;

   if (problem == 5)
   {
      mesh.Transform([](const Vector &x, Vector &y)
      {
         y = x;
         y *= 0.5;
      });
   }
   // Refine the mesh to increase the resolution. In this example we do
   // 'ref_levels' of uniform refinement, where 'ref_levels' is a command-line
   // parameter.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   // 3. Define the ODE solver used for time integration. Several explicit
   //    Runge-Kutta methods are available.
   ODESolver *ode_solver_0 = NULL;
   AdamsBashforth *ode_solver = NULL;
   ode_solver_0 = new ForwardEulerSolver;
   ode_solver = new AdamsBashforth;
   

   // 4. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   DG_FECollection fec(order, dim);
   // Finite element space for a scalar (thermodynamic quantity)
   FiniteElementSpace fes(&mesh, &fec);
   // Finite element space for a mesh-dim vector quantity (momentum)
   FiniteElementSpace dfes(&mesh, &fec, dim, Ordering::byNODES);

   // This example depends on this ordering of the space.
   MFEM_ASSERT(fes.GetOrdering() == Ordering::byNODES, "");

   cout << "Number of unknowns: " << dfes.GetVSize() << endl;

   // 5. Define the initial conditions, save the corresponding mesh and grid
   //    functions to a file. This can be opened with GLVis with the -gc option.

   // Initialize the state.
   VectorFunctionCoefficient u0 = EulerInitialCondition(problem,
                                                        specific_heat_ratio,
                                                        gas_constant);
   GridFunction mom(&dfes);
   GridFunction mom_1(&dfes);
   GridFunction mom_2(&dfes);
   GridFunction gradu_x(&dfes);
   GridFunction gradu_y(&dfes);
   GridFunction p(&fes);
   GridFunction p_grad(&dfes);
   GridFunction mom_x(&fes,mom.GetData());
   GridFunction mom_y(&fes,mom.GetData()+fes.GetVSize());
   mom.ProjectCoefficient(u0);
   p = 0.0;

   // Output the initial solution.
   {
      ostringstream mesh_name;
      mesh_name << "euler-mesh.mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(precision);
      mesh_ofs << mesh;

      for (int k = 0; k < num_equations; k++)
      {
         GridFunction uk(&fes, mom.GetData() + k * fes.GetNDofs());
         ostringstream sol_name;
         sol_name << "euler-" << k << "-init.gf";
         ofstream sol_ofs(sol_name.str().c_str());
         sol_ofs.precision(precision);
         sol_ofs << uk;
      }
   }


   // 6. Set up the nonlinear form with euler flux and numerical flux
   InEulerFlux flux(dim);
   RusanovFlux numericalFlux(flux);
   DGHyperbolicConservationLaws euler(
      dfes, std::unique_ptr<HyperbolicFormIntegrator>(
         new HyperbolicFormIntegrator(numericalFlux, IntOrderOffset)),
      preassembleWeakDiv);


   // Define pressuer pressure_rhs
   const double gamma_0=1.0;
   LinearForm *b_0 = new LinearForm(&fes);
   DivergenceGridFunctionCoefficient divu(&mom);
   ConstantCoefficient m_0(-gamma_0/dt);
   ProductCoefficient pressure_rhs_0(m_0,divu);
   b_0->AddDomainIntegrator(new DomainLFIntegrator(pressure_rhs_0));

   // Define pressuer pressure_rhs
   const double gamma=1.0;
   LinearForm *b = new LinearForm(&fes);
   ConstantCoefficient m(-gamma/dt);
   ProductCoefficient pressure_rhs(m,divu);
   b->AddDomainIntegrator(new DomainLFIntegrator(pressure_rhs));

   // Define pressuer helm_x_rhs
   ConstantCoefficient one_2(gamma/nu/dt);
   LinearForm *b_3 = new LinearForm(&fes);
   GridFunctionCoefficient cu_x(&mom_x);
   ProductCoefficient helm_x_rhs(one_2,cu_x);
   b_3->AddDomainIntegrator(new DomainLFIntegrator(helm_x_rhs));

   // Define pressuer helm_y_rhs
   LinearForm *b_4 = new LinearForm(&fes);
   GridFunctionCoefficient cu_y(&mom_y);
   ProductCoefficient helm_y_rhs(one_2,cu_y);
   b_4->AddDomainIntegrator(new DomainLFIntegrator(helm_y_rhs));

   GradientGridFunctionCoefficient grad_p(&p);
   p_grad.ProjectCoefficient(grad_p);
   CurlGridFunctionCoefficient vorticity_cf(&mom);
   GridFunction vorticity(&fes);
   AvgMassOperator avg_p(&fes);

   // Define -Δ operator
   BilinearForm *a = new BilinearForm(&fes);
   ConstantCoefficient one(1.0);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));
   a->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
   a->AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
   GSSmoother prec;
   OrthoSolver prec2;
   a->Update();
         a->Assemble();
         a->Finalize();
         SparseMatrix &A = a->SpMat();
         prec.SetOperator(A);
         prec2.SetSolver(prec);
         prec2.SetOperator(A);
   SchurConstrainedSolver csolver(A,avg_p,prec);

   // Define -Δ+I operator
   BilinearForm *ah = new BilinearForm(&fes);
   ah->AddDomainIntegrator(new DiffusionIntegrator(one));
   ah->AddDomainIntegrator(new MassIntegrator(one_2));
   ah->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
   ah->AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
   GSSmoother prec3;


   BilinearForm M(&fes);
   M.AddDomainIntegrator(new MassIntegrator);
   M.Assemble();
   GridFunction one_gf(p);
   one_gf = 1.0;
      
    //   p -= M.InnerProduct(p, one_gf);
    //   ConstrainedSolver
    //   MINRESSolver()

   // 7. Visualize momentum with its magnitude
   socketstream sout;
   if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;

      sout.open(vishost, visport);
      if (!sout)
      {
         visualization = false;
         cout << "Unable to connect to GLVis server at " << vishost << ':'
              << visport << endl;
         cout << "GLVis visualization disabled.\n";
      }
      else
      {
         // Plot magnitude of vector-valued momentum
         vorticity.ProjectCoefficient(vorticity_cf);
         sout << "solution\n" << mesh << mom;
         sout << "view 0 0\n";  // view from top
         sout << "keys jlm\n";  // turn off perspective and light
         //  sout << "pause\n";
         sout << flush;
         cout << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";
      }
   }


   // 8. Time integration

   // When dt is not specified, use CFL condition.
   // Compute h_min and initial maximum characteristic speed
   double hmin = infinity();
   if (cfl > 0)
   {
      // determine global h_min
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         hmin = min(mesh.GetElementSize(i, 1), hmin);
      }
      // Find a safe dt, using a temporary vector. Calling Mult() computes the
      // maximum char speed at all quadrature points on all faces (and all elements with -mf).
      Vector z(mom.Size());
      euler.Mult(mom, z);

      double max_char_speed = euler.GetMaxCharSpeed();
      dt = cfl * hmin / max_char_speed / (2 * order + 1)/2;
   }

   // Start the timer.
   tic_toc.Clear();
   tic_toc.Start();

   // Init time integration
   double t = 0.0;
   euler.SetTime(t);
   ode_solver->Init(euler);
   ode_solver_0->Init(euler);

   // Integrate in time.
   bool done = false;
   for (int ti = 0; !done;)
   {
      double dt_real = min(dt, t_final - t);

      if (ti == 0){
         mom_1=mom;

         // advection step
         ode_solver_0 ->Step(mom, t, dt_real);
         /////////////////////////////////////////////////////////////////////////////
         // pressure projection

         m.constant=-gamma/dt;

         b_0->Update();
         b_0->Assemble();
         prec2.Mult(*b_0,p);
         PCG(A, prec2, *b_0, p, 1, 1000, 1e-12, 0.0);
         //csolver.Mult(*b_0,p);
         
         p -= M.InnerProduct(p, one_gf);

         p_grad.ProjectCoefficient(grad_p);

         // Finalize solution
         // Get u^n+1
         mom.Add(-dt,p_grad);

         //viscous step

         one_2.constant=gamma/nu/dt;

         //BilinearForm *ah = new BilinearForm(&fes);
         ah->AddDomainIntegrator(new DiffusionIntegrator(one));
         ah->AddDomainIntegrator(new MassIntegrator(one_2));
         ah->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
         ah->AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));

         //ah->Update();
         ah->Assemble();
         //ah->Finalize();
         SparseMatrix &Ah = ah->SpMat();
         prec3.SetOperator(Ah);

         GridFunctionCoefficient cu_x(&mom_x);
         ProductCoefficient helm_x_rhs(one_2,cu_x);
         b_3->AddDomainIntegrator(new DomainLFIntegrator(helm_x_rhs));
         //b_3->Update();
         b_3->Assemble();
         PCG(Ah, prec3, *b_3, mom_x, 1, 500, 1e-12, 0.0);

         GridFunctionCoefficient cu_y(&mom_y);
         ProductCoefficient helm_y_rhs(one_2,cu_y);
         b_4->AddDomainIntegrator(new DomainLFIntegrator(helm_y_rhs));
         //b_4->Update();
         b_4->Assemble();
         PCG(Ah, prec3, *b_4, mom_y, 1, 500, 1e-12, 0.0);

         mom_2=mom;
         
      }
      else {
         // advection step
         ode_solver ->PreviousStep(mom_1);
         ode_solver ->Step(mom, t, dt_real);
         /////////////////////////////////////////////////////////////////////////////
         // pressure projection

         m.constant=-gamma/dt;

         b->Update();
         b->Assemble();
         
         prec2.Mult(*b,p);
         PCG(A, prec2, *b, p, 1, 1000, 1e-12, 0.0);
         //csolver.Mult(*b,p);


         p -= M.InnerProduct(p, one_gf);

         p_grad.ProjectCoefficient(grad_p);
         
         // Finalize solution
         // Get u^n+1
         
         mom.Add(-dt/gamma,p_grad);


         one_2.constant=gamma/nu/dt;

         //BilinearForm *ah = new BilinearForm(&fes);
         ah->AddDomainIntegrator(new DiffusionIntegrator(one));
         ah->AddDomainIntegrator(new MassIntegrator(one_2));
         ah->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
         ah->AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));

         //ah->Update();
         ah->Assemble();
         //ah->Finalize();
         SparseMatrix &Ah = ah->SpMat();
         prec3.SetOperator(Ah);

         GridFunctionCoefficient cu_x(&mom_x);
         ProductCoefficient helm_x_rhs(one_2,cu_x);
         b_3->AddDomainIntegrator(new DomainLFIntegrator(helm_x_rhs));
         //b_3->Update();
         b_3->Assemble();
         PCG(Ah, prec3, *b_3, mom_x, 1, 500, 1e-12, 0.0);

         GridFunctionCoefficient cu_y(&mom_y);
         ProductCoefficient helm_y_rhs(one_2,cu_y);
         b_4->AddDomainIntegrator(new DomainLFIntegrator(helm_y_rhs));
         //b_4->Update();
         b_4->Assemble();
         PCG(Ah, prec3, *b_4, mom_y, 1, 500, 1e-12, 0.0);

         mom_1=mom_2;
         mom_2=mom;

      }

      if (cfl > 0) // update time step size with CFL
      {
         double max_char_speed = euler.GetMaxCharSpeed();
         dt = cfl * hmin / max_char_speed / (2 * order + 1)/2;
      }
      ti++;

      done = (t >= t_final - 1e-8 * dt);
      if (done || ti % vis_steps == 0)
      {
         cout << "time step: " << ti << ", time: " << t << endl;
         if (visualization)
         {
            vorticity.ProjectCoefficient(vorticity_cf);
            sout << "solution\n" << mesh << mom << flush;
            sout << "window_title 't = " << t << "'";
         }
      }
   }

   tic_toc.Stop();
   cout << " done, " << tic_toc.RealTime() << "s." << endl;

   // 9. Save the final solution. This output can be viewed later using GLVis:
   //    "glvis -m euler-mesh-final.mesh -g euler-1-final.gf".
   {
      ostringstream mesh_name;
      mesh_name << "euler-mesh-final.mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(precision);
      mesh_ofs << mesh;

      for (int k = 0; k < num_equations; k++)
      {
         GridFunction uk(&fes, mom.GetData() + k * fes.GetNDofs());
         ostringstream sol_name;
         sol_name << "euler-" << k << "-final.gf";
         ofstream sol_ofs(sol_name.str().c_str());
         sol_ofs.precision(precision);
         sol_ofs << uk;
      }
   }

   // 10. Compute the L2 solution error summed for all components.
   const double error = mom.ComputeLpError(2, u0);
   cout << "Solution error: " << error << endl;

   // Free the used memory.
   delete ode_solver;
   delete b;
   delete b_0;
   delete a;

   return 0;
}


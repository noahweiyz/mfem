//                                MFEM Example 18 - Parallel Version
//
// Compile with: make ex18p
//
// Sample runs:
//
//       mpirun -np 4 ex18p -p 1 -rs 2 -rp 1 -o 1 -s 3
//       mpirun -np 4 ex18p -p 1 -rs 1 -rp 1 -o 3 -s 4
//       mpirun -np 4 ex18p -p 1 -rs 1 -rp 1 -o 5 -s 6
//       mpirun -np 4 ex18p -p 2 -rs 1 -rp 1 -o 1 -s 3 -mf
//       mpirun -np 4 ex18p -p 2 -rs 1 -rp 1 -o 3 -s 3 -mf
//
// Description:  This example code solves the compressible Euler system of
//               equations, a model nonlinear hyperbolic PDE, with a
//               discontinuous Galerkin (DG) formulation in parallel.
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
#include "ex18.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 0. Parallel setup
   Mpi::Init(argc, argv);
   const int numProcs = Mpi::WorldSize();
   const int myRank = Mpi::WorldRank();
   Hypre::Init();

   // 1. Parse command-line options.
   int problem = 1;
   const double specific_heat_ratio = 1.4;
   const double gas_constant = 1.0;

   string mesh_file = "";
   int IntOrderOffset = 1;
   int ser_ref_levels = 0;
   int par_ref_levels = 1;
   int order = 3;
   int ode_solver_type = 4;
   double t_final = 2.0;
   double dt = -0.01;
   double cfl = 0.3;
   bool visualization = true;
   bool preassembleWeakDiv = true;
   int vis_steps = 50;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use. If not provided, then a periodic square"
                  " mesh will be used.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use. See options in velocity_function().");
   args.AddOption(&ser_ref_levels, "-rs", "--serial-refine",
                  "Number of times to refine the serial mesh uniformly.");
   args.AddOption(&par_ref_levels, "-rp", "--parallel-refine",
                  "Number of times to refine the parallel mesh uniformly.");
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

   // 2. Read the mesh from the given mesh file. When the user does not provide
   //    mesh file, use the default mesh file for the problem.
   Mesh mesh = mesh_file.empty() ? EulerMesh(problem) : Mesh(mesh_file);
   const int dim = mesh.Dimension();
   const int num_equations = dim + 2;

   if (problem == 5)
   {
      mesh.Transform([](const Vector &x, Vector &y)
      {
         y = x;
         y *= 0.5;
      });
   }

   // Refine the mesh to increase the resolution. In this example we do
   // 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is a
   // command-line parameter.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   // Define a parallel mesh by a partitioning of the serial mesh. Refine this
   // mesh further in parallel to increase the resolution. Once the parallel
   // mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh = ParMesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // Refine the mesh to increase the resolution. In this example we do
   // 'par_ref_levels' of uniform refinement, where 'par_ref_levels' is a
   // command-line parameter.
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh.UniformRefinement();
   }

   // 3. Define the ODE solver used for time integration. Several explicit
   //    Runge-Kutta methods are available.
   ODESolver *ode_solver = NULL;
   switch (ode_solver_type)
   {
      case 1: ode_solver = new ForwardEulerSolver; break;
      case 2: ode_solver = new RK2Solver(1.0); break;
      case 3: ode_solver = new RK3SSPSolver; break;
      case 4: ode_solver = new RK4Solver; break;
      case 6: ode_solver = new RK6Solver; break;
      default:
         cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         return 3;
   }

   // 4. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   DG_FECollection fec(order, dim);
   // Finite element space for a scalar (thermodynamic quantity)
   ParFiniteElementSpace fes(&pmesh, &fec);
   // Finite element space for a mesh-dim vector quantity (momentum)
   ParFiniteElementSpace dfes(&pmesh, &fec, dim, Ordering::byNODES);
   // Finite element space for all variables together (total thermodynamic state)
   ParFiniteElementSpace vfes(&pmesh, &fec, num_equations, Ordering::byNODES);

   // This example depends on this ordering of the space.
   MFEM_ASSERT(fes.GetOrdering() == Ordering::byNODES, "");

   HYPRE_BigInt glob_size = vfes.GlobalTrueVSize();
   if (Mpi::Root())
   {
      cout << "Number of unknowns: " << glob_size << endl;
   }

   // 5. Define the initial conditions, save the corresponding mesh and grid
   //    functions to a file. This can be opened with GLVis with the -gc option.

   // Initialize the state.
   VectorFunctionCoefficient u0 = EulerInitialCondition(problem,
                                                        specific_heat_ratio,
                                                        gas_constant);
   ParGridFunction sol(&vfes);
   sol.ProjectCoefficient(u0);
   ParGridFunction mom(&dfes, sol.GetData() + fes.GetNDofs());
   // Output the initial solution.
   {
      ostringstream mesh_name;
      mesh_name << "euler-mesh." << setfill('0') << setw(6) << Mpi::WorldRank();
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(precision);
      mesh_ofs << pmesh;

      for (int k = 0; k < num_equations; k++)
      {
         ParGridFunction uk(&fes, sol.GetData() + k * fes.GetNDofs());
         ostringstream sol_name;
         sol_name << "euler-" << k << "-init." << setfill('0') << setw(6)
                  << Mpi::WorldRank();
         ofstream sol_ofs(sol_name.str().c_str());
         sol_ofs.precision(precision);
         sol_ofs << uk;
      }
   }

   // 6. Set up the nonlinear form with euler flux and numerical flux
   EulerFlux flux(dim, specific_heat_ratio);
   RusanovFlux numericalFlux(flux);
   DGHyperbolicConservationLaws euler(
      vfes, std::unique_ptr<HyperbolicFormIntegrator>(
         new HyperbolicFormIntegrator(numericalFlux, IntOrderOffset)),
      preassembleWeakDiv);

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
         if (Mpi::Root())
         {
            cout << "Unable to connect to GLVis server at " << vishost << ':'
                 << visport << endl;
            cout << "GLVis visualization disabled.\n";
         }
      }
      else
      {
         sout << "parallel " << numProcs << " " << myRank << "\n";
         sout.precision(precision);
         // Plot magnitude of vector-valued momentum
         sout << "solution\n" << pmesh << mom;
         sout << "view 0 0\n";  // view from top
         sout << "keys jlm\n";  // turn off perspective and light
         sout << "pause\n";
         sout << flush;
         if (Mpi::Root())
         {
            cout << "GLVis visualization paused."
                 << " Press space (in the GLVis window) to resume it.\n";
         }
         MPI_Barrier(pmesh.GetComm());
      }
   }

   // 8. Time integration

   // When dt is not specified, use CFL condition.
   // Compute h_min and initial maximum characteristic speed
   double hmin = infinity();
   if (cfl > 0)
   {
      for (int i = 0; i < pmesh.GetNE(); i++)
      {
         hmin = min(pmesh.GetElementSize(i, 1), hmin);
      }
      MPI_Allreduce(MPI_IN_PLACE, &hmin, 1, MPI_DOUBLE, MPI_MIN,
                    pmesh.GetComm());
      // Find a safe dt, using a temporary vector. Calling Mult() computes the
      // maximum char speed at all quadrature points on all faces.
      Vector z(sol.Size());
      euler.Mult(sol, z);

      double max_char_speed  = euler.GetMaxCharSpeed();
      MPI_Allreduce(MPI_IN_PLACE, &max_char_speed, 1, MPI_DOUBLE, MPI_MAX,
                    pmesh.GetComm());
      dt = cfl * hmin / max_char_speed / (2 * order + 1);
   }

   // Start the timer.
   tic_toc.Clear();
   tic_toc.Start();

   // Init time integration
   double t = 0.0;
   euler.SetTime(t);
   ode_solver->Init(euler);

   // Integrate in time.
   bool done = false;
   for (int ti = 0; !done;)
   {
      double dt_real = min(dt, t_final - t);

      ode_solver->Step(sol, t, dt_real);
      if (cfl > 0) // update time step size with CFL
      {
         double max_char_speed = euler.GetMaxCharSpeed();
         MPI_Allreduce(MPI_IN_PLACE, &max_char_speed, 1, MPI_DOUBLE, MPI_MAX,
                       pmesh.GetComm());
         dt = cfl * hmin / max_char_speed / (2 * order + 1);
      }
      ti++;

      done = (t >= t_final - 1e-8 * dt);
      if (done || ti % vis_steps == 0)
      {
         if (Mpi::Root())
         {
            cout << "time step: " << ti << ", time: " << t << endl;
         }
         if (visualization)
         {
            sout << "parallel " << numProcs << " " << myRank << "\n";
            sout << "solution\n" << pmesh << mom;
            sout << "window_title 't = " << t << "'" << flush;
         }
      }
   }

   tic_toc.Stop();
   if (Mpi::Root())
   {
      cout << " done, " << tic_toc.RealTime() << "s." << endl;
   }

   // 9. Save the final solution. This output can be viewed later using GLVis:
   //    "glvis -np 4 -m euler-mesh-final -g euler-1-final".
   {
      ostringstream mesh_name;
      mesh_name << "euler-mesh-final." << setfill('0') << setw(6)
                << Mpi::WorldRank();
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(precision);
      mesh_ofs << pmesh;

      for (int k = 0; k < num_equations; k++)
      {
         ParGridFunction uk(&fes, sol.GetData() + k * fes.GetNDofs());
         ostringstream sol_name;
         sol_name << "euler-" << k << "-final." << setfill('0') << setw(6)
                  << Mpi::WorldRank();
         ofstream sol_ofs(sol_name.str().c_str());
         sol_ofs.precision(precision);
         sol_ofs << uk;
      }
   }

   // 10. Compute the L2 solution error summed for all components.
   const double error = sol.ComputeLpError(2, u0);
   if (Mpi::Root())
   {
      cout << "Solution error: " << error << endl;
   }

   // Free the used memory.
   delete ode_solver;

   return 0;
}

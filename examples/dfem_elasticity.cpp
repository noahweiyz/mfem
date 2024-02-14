#include "dfem.hpp"

int main(int argc, char *argv[])
{
   Mpi::Init();

   std::cout << std::setprecision(9);

   const char *mesh_file = "../data/star.mesh";
   int polynomial_order = 1;
   int refinements = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&polynomial_order, "-o", "--order", "");
   args.AddOption(&refinements, "-r", "--r", "");
   args.ParseCheck();

   Mesh mesh_serial(mesh_file, 1, 1);
   mesh_serial.SetCurvature(1);
   for (int i = 0; i < refinements; i++)
   {
      mesh_serial.UniformRefinement();
   }
   const int dim = mesh_serial.Dimension();

   ParMesh mesh(MPI_COMM_WORLD, mesh_serial);
   mesh_serial.Clear();

   constexpr int vdim = 2;

   // test_partial_assembly_setup_qf(mesh, 1, polynomial_order);
   // exit(0);

   H1_FECollection h1fec(polynomial_order, dim);
   ParFiniteElementSpace h1fes(&mesh, &h1fec, vdim);

   Array<int> ess_tdof_list;
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;
   h1fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   const IntegrationRule &ir =
      IntRules.Get(h1fes.GetFE(0)->GetGeomType(), 2 * h1fec.GetOrder() + 1);

   std::cout << "nqpts = " << ir.GetNPoints() << std::endl;
   std::cout << "ndofs = " << h1fes.GlobalTrueVSize() << std::endl;

   ParGridFunction u(&h1fes);
   ParGridFunction g(&h1fes);
   ParGridFunction rho(&h1fes);

   auto exact_solution = [](const Vector &coords, Vector &u)
   {
      const double x = coords(0);
      const double y = coords(1);
      u(0) = x*x + y;
      u(1) = x + 0.5*y*y;
      out << "(" << x << ", " << y << ")" << " (" << u(0) << ", " << u(
             1) << ")" << "\n";
   };

   VectorFunctionCoefficient exact_solution_coeff(dim, exact_solution);

   // Objective function
   {
      auto objective = [](tensor<double, 2> u, double rho,
                          tensor<double, 2, 2> J,
                          double w)
      {
         out << u << "\n";
         return sqnorm(u) * det(J) * w;
      };

      ElementOperator objective_eop
      {
         objective,
         std::tuple{
            Value{"displacement"},
            Value{"density"},
            Gradient{"coordinates"},
            Weight{"integration_weight"}},
         // outputs (return values)
         std::tuple{
            One{"integral"}
         }
      };

      DifferentiableForm dop(
      {
         {&u, "displacement"},
      },
      {
         {mesh.GetNodes(), "coordinates"},
         {&rho, "density"},
      },
      mesh);

      dop.AddElementOperator(objective_eop, ir);

      u.ProjectCoefficient(exact_solution_coeff);
      Vector zero;

      // finite difference test
      // DenseMatrix dfdu(qdata.Size(), u.Size());
      Vector y(1);
      Vector utdof;
      u.GetTrueDofs(utdof);
      dop.Mult(utdof, y);

      // Vector fx(qdata);
      // // out << "g: " << fx << "\n";
      // print_vector(fx);
      // out << "\n";

      // for (int i = 0; i < u.Size(); i++)
      // {
      //    double h = 1e-6;
      //    u(i) += h;
      //    dop.Mult(u, qdata);
      //    Vector col(qdata);
      //    col -= fx;
      //    col /= h;
      //    u(i) -= h;
      //    dfdu.SetCol(i, col);
      // }

      // Vector du(u.Size());
      // du.Randomize(1245);
      // Vector df(qdata.Size());
      // dfdu.Mult(du, df);
      // print_vector(df);

      // auto &dfdu_op = dop.GetGradient(u);
      // dfdu_op.Mult(du, df);
      // print_vector(df);
   }

   // auto linear_elastic = [](tensor<dual<double, double>, 2, 2> dudxi,
   //                          tensor<double, 2, 2> J,
   //                          double w)
   // {
   //    using mfem::internal::tensor;
   //    using mfem::internal::IsotropicIdentity;

   //    double lambda, mu;
   //    {
   //       lambda = 1.0;
   //       mu = 1.0;
   //    }
   //    static constexpr auto I = IsotropicIdentity<2>();
   //    auto eps = sym(dudxi * inv(J));
   //    auto JxW = transpose(inv(J)) * det(J) * w;
   //    auto r = (lambda * tr(eps) * I + 2.0 * mu * eps) * JxW;
   //    return r;
   // };

   // ElementOperator qf
   // {
   //    // quadrature function lambda
   //    linear_elastic,
   //    // inputs
   //    std::tuple{
   //       Gradient{"displacement"},
   //       Gradient{"coordinates"},
   //       Weight{"integration_weight"}},
   //    // outputs (return values)
   //    std::tuple{
   //       Gradient{"displacement"}
   //    }
   // };

   // ElementOperator forcing_qf
   // {
   //    [](tensor<double, 2> x, tensor<double, 2, 2> J, double w)
   //    {
   //       double lambda, mu;
   //       {
   //          lambda = 1.0;
   //          mu = 1.0;
   //       }
   //       auto f = x;
   //       f(0) = 4.0*mu + 2.0*lambda;
   //       f(1) = 2.0*mu + lambda;
   //       return f * det(J) * w;
   //    },
   //    // inputs
   //    std::tuple{
   //       Value{"coordinates"},
   //       Gradient{"coordinates"},
   //       Weight{"integration_weight"}},
   //    // outputs
   //    std::tuple{
   //       Value{"displacement"}}
   // };

   // DifferentiableForm dop(
   //    // Solutions
   // {{&u, "displacement"}},
   // // Parameters
   // {
   //    {mesh.GetNodes(), "coordinates"},
   // },
   // mesh);

   // dop.AddElementOperator<AD::Enzyme>(qf, ir);
   // dop.AddElementOperator<AD::None>(forcing_qf, ir);
   // dop.SetEssentialTrueDofs(ess_tdof_list);

   // GMRESSolver gmres(MPI_COMM_WORLD);
   // gmres.SetRelTol(1e-12);
   // gmres.SetMaxIter(5000);
   // gmres.SetPrintLevel(IterativeSolver::PrintLevel().Summary());

   // NewtonSolver newton(MPI_COMM_WORLD);
   // newton.SetSolver(gmres);
   // newton.SetOperator(dop);
   // newton.SetRelTol(1e-12);
   // newton.SetMaxIter(100);
   // // newton.SetAdaptiveLinRtol();
   // newton.SetPrintLevel(1);

   // u = 1e-6;
   // u.ProjectBdrCoefficient(exact_solution_coeff, ess_bdr);
   // Vector x;
   // u.GetTrueDofs(x);

   // Vector zero;
   // newton.Mult(zero, x);

   // u.Distribute(x);

   // std::cout << "|u-u_ex|_L2 = " << u.ComputeL2Error(exact_solution_coeff) << "\n";

   return 0;
}

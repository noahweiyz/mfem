#include "dfem.hpp"

int test_interpolate_linear_scalar(std::string mesh_file,
                                   int refinements,
                                   int polynomial_order)
{
   Mesh mesh_serial = Mesh(mesh_file);
   for (int i = 0; i < refinements; i++)
   {
      mesh_serial.UniformRefinement();
   }
   ParMesh mesh(MPI_COMM_WORLD, mesh_serial);

   mesh.SetCurvature(1);
   const int dim = mesh.Dimension();
   mesh_serial.Clear();

   H1_FECollection h1fec(polynomial_order, dim);
   ParFiniteElementSpace h1fes(&mesh, &h1fec);

   const IntegrationRule &ir =
      IntRules.Get(h1fes.GetFE(0)->GetGeomType(), 2 * h1fec.GetOrder() + 1);

   QuadratureSpace qspace(mesh, ir);
   QuadratureFunction qf(&qspace);

   ParGridFunction f1_g(&h1fes);

   auto mass_qf = [](double u, tensor<double, 2, 2> J, double w)
   {
      return u;
   };

   ElementOperator mass
   {
      mass_qf,
      // inputs
      std::tuple{
         Value{"primary_variable"},
         Gradient{"coordinates"},
         Weight{"integration_weight"}},
      // outputs
      std::tuple{None{"quadrature_data"}}};

   DifferentiableForm dop(
   {
      {&f1_g, "primary_variable"}
   },
   {
      {mesh.GetNodes(), "coordinates"},
      {&qf, "quadrature_data"}
   },
   mesh);

   dop.AddElementOperator(mass, ir);

   auto f1 = [](const Vector &coords)
   {
      const double x = coords(0);
      const double y = coords(1);
      return 2.345 + x + y;
   };

   FunctionCoefficient f1_c(f1);
   f1_g.ProjectCoefficient(f1_c);

   Vector x(f1_g);
   dop.Mult(x, qf);

   Vector f_test(qf.Size());
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      ElementTransformation *T = mesh.GetElementTransformation(e);
      for (int qp = 0; qp < ir.GetNPoints(); qp++)
      {
         const IntegrationPoint &ip = ir.IntPoint(qp);
         T->SetIntPoint(&ip);

         f_test((e * ir.GetNPoints()) + qp) = f1_c.Eval(*T, ip);
      }
   }

   f_test -= qf;
   if (f_test.Norml2() > 1e-12)
   {
      return 1;
   }

   return 0;
}

int test_interpolate_gradient_scalar(std::string mesh_file,
                                     int refinements,
                                     int polynomial_order)
{
   Mesh mesh_serial = Mesh(mesh_file);
   for (int i = 0; i < refinements; i++)
   {
      mesh_serial.UniformRefinement();
   }
   ParMesh mesh(MPI_COMM_WORLD, mesh_serial);
   mesh.SetCurvature(1);
   const int dim = mesh.Dimension();
   mesh_serial.Clear();

   H1_FECollection h1fec(polynomial_order, dim);
   ParFiniteElementSpace h1fes(&mesh, &h1fec);

   const IntegrationRule &ir =
      IntRules.Get(h1fes.GetFE(0)->GetGeomType(), 2 * h1fec.GetOrder() + 1);

   QuadratureSpace qspace(mesh, ir);
   QuadratureFunction qf(&qspace, dim);

   ParGridFunction f1_g(&h1fes);

   auto mass_qf = [](double u, tensor<double, 2> grad_u, tensor<double, 2, 2> J,
                     double w)
   {
      return grad_u * inv(J);
   };

   ElementOperator mass
   {
      mass_qf,
      // inputs
      std::tuple{
         Value{"primary_variable"},
         Gradient{"primary_variable"},
         Gradient{"coordinates"},
         Weight{"integration_weight"}},
      // outputs
      std::tuple{None{"quadrature_data"}}};

   DifferentiableForm dop(
   {{&f1_g, "primary_variable"}},
   {
      {mesh.GetNodes(), "coordinates"},
      {&qf, "quadrature_data"}
   },
   mesh);

   dop.AddElementOperator(mass, ir);

   auto f1 = [](const Vector &coords)
   {
      const double x = coords(0);
      const double y = coords(1);
      return 2.345 + x*y + y;
   };

   FunctionCoefficient f1_c(f1);
   f1_g.ProjectCoefficient(f1_c);

   Vector x(f1_g);
   dop.Mult(x, qf);

   Vector f_test(qf.Size());
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      ElementTransformation *T = mesh.GetElementTransformation(e);
      for (int qp = 0; qp < ir.GetNPoints(); qp++)
      {
         const IntegrationPoint &ip = ir.IntPoint(qp);
         T->SetIntPoint(&ip);

         Vector g(dim);
         f1_g.GetGradient(*T, g);
         for (int d = 0; d < dim; d++)
         {
            int qpo = qp * dim;
            int eo = e * (ir.GetNPoints() * dim);
            f_test(d + qpo + eo) = g(d);
         }
      }
   }

   f_test -= qf;
   if (f_test.Norml2() > 1e-12)
   {
      return 1;
   }
   return 0;
}

int test_interpolate_linear_vector(std::string mesh_file, int refinements,
                                   int polynomial_order)
{
   constexpr int vdim = 2;
   Mesh mesh_serial = Mesh(mesh_file);
   for (int i = 0; i < refinements; i++)
   {
      mesh_serial.UniformRefinement();
   }

   ParMesh mesh(MPI_COMM_WORLD, mesh_serial);
   mesh.SetCurvature(1);
   const int dim = mesh.Dimension();
   mesh_serial.Clear();

   H1_FECollection h1fec(polynomial_order, dim);
   ParFiniteElementSpace h1fes(&mesh, &h1fec, vdim);

   const IntegrationRule &ir =
      IntRules.Get(h1fes.GetFE(0)->GetGeomType(), 2 * h1fec.GetOrder() + 1);

   QuadratureSpace qspace(mesh, ir);
   QuadratureFunction qf(&qspace, vdim);

   ParGridFunction f1_g(&h1fes);

   auto mass_qf = [](tensor<double, vdim> u, tensor<double, 2, 2> J, double w)
   {
      return u;
   };

   ElementOperator mass
   {
      mass_qf,
      // inputs
      std::tuple{
         Value{"primary_variable"},
         Gradient{"coordinates"},
         Weight{"integration_weight"}},
      // outputs
      std::tuple{None{"quadrature_data"}}};

   DifferentiableForm dop(
   {{&f1_g, "primary_variable"}},
   {
      {mesh.GetNodes(), "coordinates"},
      {&qf, "quadrature_data"}
   },
   mesh);

   dop.AddElementOperator(mass, ir);

   auto f1 = [](const Vector &coords, Vector &u)
   {
      const double x = coords(0);
      const double y = coords(1);
      u(0) = 2.345 + x + y;
      u(1) = 12.345 + x + y;
   };

   VectorFunctionCoefficient f1_c(vdim, f1);
   f1_g.ProjectCoefficient(f1_c);

   Vector x(f1_g);
   dop.Mult(x, qf);

   Vector f_test(qf.Size());
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      ElementTransformation *T = mesh.GetElementTransformation(e);
      for (int qp = 0; qp < ir.GetNPoints(); qp++)
      {
         const IntegrationPoint &ip = ir.IntPoint(qp);
         T->SetIntPoint(&ip);

         Vector f(vdim);
         f1_g.GetVectorValue(*T, ip, f);
         for (int d = 0; d < vdim; d++)
         {
            int qpo = qp * vdim;
            int eo = e * (ir.GetNPoints() * vdim);
            f_test(d + qpo + eo) = f(d);
         }
      }
   }

   f_test -= qf;
   if (f_test.Norml2() > 1e-12)
   {
      return 1;
   }
   return 0;
}

int test_interpolate_gradient_vector(std::string mesh_file,
                                     int refinements,
                                     int polynomial_order)
{
   constexpr int vdim = 2;
   Mesh mesh_serial = Mesh(mesh_file, 1, 1);
   for (int i = 0; i < refinements; i++)
   {
      mesh_serial.UniformRefinement();
   }
   ParMesh mesh(MPI_COMM_WORLD, mesh_serial);
   mesh.SetCurvature(1);
   const int dim = mesh.Dimension();
   mesh_serial.Clear();

   H1_FECollection h1fec(polynomial_order, dim);
   ParFiniteElementSpace h1fes(&mesh, &h1fec, vdim);

   const IntegrationRule &ir =
      IntRules.Get(h1fes.GetFE(0)->GetGeomType(), 2 * h1fec.GetOrder() + 1);

   QuadratureSpace qspace(mesh, ir);
   QuadratureFunction qf(&qspace, dim * vdim);

   ParGridFunction f1_g(&h1fes);

   auto mass_qf = [](tensor<double, vdim, 2> grad_u, tensor<double, 2, 2> J,
                     double w)
   {
      // out << grad_u * inv(J) << " ";
      return grad_u * inv(J);
   };

   ElementOperator mass
   {
      mass_qf,
      // inputs
      std::tuple{
         Gradient{"primary_variable"},
         Gradient{"coordinates"},
         Weight{"integration_weight"}},
      // outputs
      std::tuple{None{"quadrature_data"}}};

   DifferentiableForm dop(
   {{&f1_g, "primary_variable"}},
   {
      {mesh.GetNodes(), "coordinates"},
      {&qf, "quadrature_data"}
   },
   mesh);

   dop.AddElementOperator(mass, ir);

   auto f1 = [](const Vector &coords, Vector &u)
   {
      const double x = coords(0);
      const double y = coords(1);
      u(0) = x + y;
      u(1) = x + 0.5*y;
   };

   VectorFunctionCoefficient f1_c(vdim, f1);
   f1_g.ProjectCoefficient(f1_c);

   Vector x(f1_g);
   dop.Mult(x, qf);

   Vector f_test(qf.Size());
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      ElementTransformation *T = mesh.GetElementTransformation(e);
      for (int qp = 0; qp < ir.GetNPoints(); qp++)
      {
         const IntegrationPoint &ip = ir.IntPoint(qp);
         T->SetIntPoint(&ip);

         DenseMatrix g(vdim, dim);
         f1_g.GetVectorGradient(*T, g);
         for (int i = 0; i < vdim; i++)
         {
            for (int j = 0; j < dim; j++)
            {
               int eo = e * (ir.GetNPoints() * dim * vdim);
               int qpo = qp * dim * vdim;
               int idx = (j + (i * dim) + qpo + eo);
               f_test(idx) = g(i, j);
            }
         }
      }
   }

   f_test -= qf;
   if (f_test.Norml2() > 1e-12)
   {
      return 1;
   }
   return 0;
}

int test_partial_assembly_setup_qf(ParMesh &mesh, const int vdim,
                                   const int polynomial_order)
{
   const int dim = mesh.Dimension();
   H1_FECollection h1fec(polynomial_order, dim);
   ParFiniteElementSpace h1fes(&mesh, &h1fec, vdim);

   Array<int> ess_tdof_list;
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;
   h1fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   const IntegrationRule &ir =
      IntRules.Get(h1fes.GetFE(0)->GetGeomType(), 2 * h1fec.GetOrder() + 1);

   QuadratureSpace qspace(mesh, ir);
   QuadratureFunction qf(&qspace, dim * dim);

   qf = 0.0;

   std::cout << "nqpts = " << ir.GetNPoints() << std::endl;
   std::cout << "ndofs = " << h1fes.GlobalTrueVSize() << std::endl;

   ParGridFunction u(&h1fes);

   auto pa_setup = [](tensor<double, 2, 2> J, double w)
   {
      return inv(J) * transpose(inv(J)) * det(J) * w;
      // out << J << "\n";
      // return J;
   };

   ElementOperator eop
   {
      pa_setup,
      std::tuple{
         Gradient{"coordinates"},
         Weight{"integration_weight"}},
      std::tuple{
         None{"quadrature_data"}
      }
   };

   DifferentiableForm dop_pasetup(
      // Solutions
   {
   },
   // Parameters
   {
      {mesh.GetNodes(), "coordinates"},
      {&qf, "quadrature_data"}
   },
   mesh);

   dop_pasetup.AddElementOperator(eop, ir);

   out << "setup" << "\n";
   Vector zero;
   dop_pasetup.Mult(zero, qf);

   auto pa_apply = [](tensor<double, 2> dudxi, tensor<double, 2, 2> qdata)
   {
      return dudxi * qdata;
   };

   ElementOperator eo_apply
   {
      pa_apply,
      std::tuple{
         Gradient{"potential"},
         None{"quadrature_data"}
      },
      std::tuple{
         Gradient{"potential"}
      }
   };

   DifferentiableForm dop_paapply(
      // Solutions
   {{&u, "potential"}},
   // Parameters
   {
      {mesh.GetNodes(), "coordinates"},
      {&qf, "quadrature_data"},
   },
   mesh);

   dop_paapply.AddElementOperator(eo_apply, ir);

   Vector y(u.Size());
   out << "apply" << "\n";
   dop_paapply.Mult(u, y);

   // out << "gradient" << "\n";
   // {
   //    u = 1.0;
   //    DenseMatrix J(u.Size());
   //    auto &dop_grad = dop_paapply.GetGradient(u);

   //    Vector y(u.Size());
   //    u = 0.0;
   //    std::ofstream ostrm("amat_dfem.dat");
   //    for (size_t i = 0; i < u.Size(); i++)
   //    {
   //       u(i) = 1.0;
   //       dop_grad.Mult(u, y);
   //       J.SetRow(i, y);
   //       u(i) = 0.0;
   //    }
   //    for (size_t i = 0; i < u.Size(); i++)
   //    {
   //       for (size_t j = 0; j < u.Size(); j++)
   //       {
   //          ostrm << J(i,j) << " ";
   //       }
   //       ostrm << "\n";
   //    }
   //    ostrm.close();
   //    // exit(0);
   // }

   return 0;
}

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

   int ret;
   ret = test_interpolate_linear_scalar(mesh_file, refinements, polynomial_order);
   out << "test_interpolate_linear_scalar";
   ret ? out << " FAILURE\n" : out << " OK\n";
   ret = test_interpolate_gradient_scalar(mesh_file, refinements,
                                          polynomial_order);
   out << "test_interpolate_gradient_scalar";
   ret ? out << " FAILURE\n" : out << " OK\n";
   ret = test_interpolate_linear_vector(mesh_file, refinements, polynomial_order);
   out << "test_interpolate_linear_vector";
   ret ? out << " FAILURE\n" : out << " OK\n";
   ret = test_interpolate_gradient_vector(mesh_file,
                                          refinements,
                                          polynomial_order);
   out << "test_interpolate_gradient_vector";
   ret ? out << " FAILURE\n" : out << " OK\n";

   return 0;
}

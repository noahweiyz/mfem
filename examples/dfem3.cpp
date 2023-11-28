#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <general/forall.hpp>
#include <mfem.hpp>

using namespace mfem;

using InterpolationPair = std::pair<Array<double>, Array<double>>;

using InterpolationPairTensor =
   std::pair<DeviceTensor<2, const double>, DeviceTensor<3, const double>>;

using Field =
   std::pair<std::variant<const GridFunction *, const ParGridFunction *>,
   std::string>;

const GridFunction &GetGridFunction(const Field &f)
{
   if (std::holds_alternative<const ParGridFunction *>(f.first))
   {
      return *std::get<const ParGridFunction *>(f.first);
   }
   else if (std::holds_alternative<const GridFunction *>(f.first))
   {
      return *std::get<const GridFunction *>(f.first);
   }
   else
   {
      MFEM_ABORT("variant not supported");
   }
}

// This probably won't work in parallel. Need to find a way to make a
// distinction between Par/Non-ParFESpace here.
const FiniteElementSpace &GetFESpace(const Field &f)
{
   return *GetGridFunction(f).FESpace();
}

int GetVDim(const Field &f) { return GetFESpace(f).GetVDim(); }

int GetDimension(const Field &f)
{
   return GetFESpace(f).GetMesh()->Dimension();
}

// template <typename T>
// struct Field(T &, std::string)

class FieldDescriptor
{
public:
   enum InterpolationType { VALUE, GRADIENT, CURL } interpolation;

   FieldDescriptor(InterpolationType interp, std::string name)
      : interpolation(interp), name(name) {}

   std::string name;

   int size_on_qp;

   int dim;

   int vdim;

   friend std::ostream &operator<<(std::ostream &os, const FieldDescriptor &fd)
   {
      os << fd.name << " ";
      switch (fd.interpolation)
      {
         case InterpolationType::VALUE:
            os << "value";
            break;
         case InterpolationType::GRADIENT:
            os << "gradient";
            break;
         case InterpolationType::CURL:
            os << "curl";
            break;
      }
      return os;
   }
};

using TestSpace = std::pair<const FiniteElementSpace *, FieldDescriptor>;

int GetSizeOnQP(const FieldDescriptor &fd, const Field &field)
{
   switch (fd.interpolation)
   {
      case FieldDescriptor::InterpolationType::VALUE:
         return GetVDim(field);
         break;
      case FieldDescriptor::InterpolationType::GRADIENT:
         return GetVDim(field) * GetDimension(field);
         break;
      default:
         MFEM_ABORT("ERROR");
         return -1;
   }
}

class Solution
{
public:
   Solution(GridFunction &u) : u(u) {}
   GridFunction &GetObject() { return u; };
   GridFunction &u;
};

template <typename T> class Parameter
{
public:
   Parameter(T &rho) : rho(rho) {}
   T &GetObject() { return rho; };
   T &rho;
};

template <typename quadrature_function_type> struct QFunction
{
   quadrature_function_type func;
   std::vector<FieldDescriptor> inputs;
   std::vector<FieldDescriptor> outputs;
};

template <typename T>
QFunction(T, std::vector<FieldDescriptor>, std::vector<FieldDescriptor>)
-> QFunction<T>;

void element_restriction(std::vector<Field> fields, const Vector &x,
                         std::vector<Vector> &fields_e)
{
   int offset = 0;
   for (int i = 0; i < fields.size(); i++)
   {
      auto fes = GetFESpace(fields[i]);
      const auto R = fes.GetElementRestriction(ElementDofOrdering::NATIVE);
      const int height = R->Height();
      const Vector x_i(x.GetData() + offset, height);
      fields_e[i].SetSize(height);

      R->Mult(x_i, fields_e[i]);

      offset += height;
   }
}

void element_restriction(std::vector<Field> fields, int offset,
                         std::vector<Vector> &fields_e)
{
   for (int i = 0; i < fields.size(); i++)
   {
      fields_e[i + offset] = GetGridFunction(fields[i]);
   }
}

void element_restriction_transpose(const Vector &r_e,
                                   const FiniteElementSpace *t, Vector &y)
{
   t->GetElementRestriction(ElementDofOrdering::NATIVE)->MultTranspose(r_e, y);
}

template <class F> struct FunctionSignature;

template <typename output_type, typename... input_types>
struct FunctionSignature<output_type(input_types...)>
{
   using return_type = output_type;
   using parameter_types = std::tuple<input_types...>;
};

template <class T> struct create_function_signature;

template <typename output_type, typename T, typename... input_types>
struct create_function_signature<output_type (T::*)(input_types...) const>
{
   using type = FunctionSignature<output_type(input_types...)>;
};

void allocate_qf_arg(const FieldDescriptor &input, double &arg)
{
   // no op
}

void allocate_qf_arg(const FieldDescriptor &input, Vector &arg)
{
   arg.SetSize(input.size_on_qp);
}

void allocate_qf_arg(const FieldDescriptor &input, DenseMatrix &arg)
{
   if (input.interpolation == FieldDescriptor::InterpolationType::GRADIENT)
   {
      arg.SetSize(input.size_on_qp / input.dim);
   }
}

template <typename qf_args, std::size_t... i>
void allocate_qf_args_impl(qf_args &args, std::vector<FieldDescriptor> inputs,
                           std::index_sequence<i...>)
{
   (allocate_qf_arg(inputs[i], std::get<i>(args)), ...);
}

template <typename qf_args>
void allocate_qf_args(qf_args &args, std::vector<FieldDescriptor> inputs)
{
   allocate_qf_args_impl(args, inputs,
                         std::make_index_sequence<std::tuple_size_v<qf_args>> {});
}

// This can also be implemented by a user who wants exotic types in their
// quadrature functions
void prepare_qf_arg(const DeviceTensor<1> &u, double &arg) { arg = u(0); }

void prepare_qf_arg(const DeviceTensor<1> &u, Vector &arg)
{
   for (int i = 0; i < u.GetShape()[0]; i++)
   {
      arg[i] = u(i);
   }
}

void prepare_qf_arg(const DeviceTensor<1> &u, DenseMatrix &arg)
{
   for (int i = 0; i < u.GetShape()[0]; i++)
   {
      arg.Data()[i] = u(i);
   }
}

template <typename arg_type>
void prepare_qf_arg(const DeviceTensor<2> &u, arg_type &arg, int qp)
{
   const auto u_qp = Reshape(&u(0, qp), u.GetShape()[0]);
   prepare_qf_arg(u_qp, arg);
}

template <typename qf_type, typename qf_args, std::size_t... i>
void prepare_qf_args(const qf_type &qf, std::vector<DeviceTensor<2>> &u,
                     qf_args &args, int qp, std::index_sequence<i...>)
{
   // we have several options here
   // - reinterpret_cast
   // - memcpy (copy data of u -> arg with overloading operator= for example)
   (prepare_qf_arg(u[i], std::get<i>(args), qp), ...);
}

Vector prepare_qf_result(double x)
{
   Vector r(1);
   r = x;
   return r;
}

Vector prepare_qf_result(Vector x) { return x; }

template <typename qf_type, typename qf_args>
auto apply_qf(const qf_type &qf, qf_args &args, std::vector<DeviceTensor<2>> &u,
              int qp)
{
   prepare_qf_args(qf, u, args, qp,
                   std::make_index_sequence<std::tuple_size_v<qf_args>> {});

   return prepare_qf_result(std::apply(qf, args));
}

void interpolate(const std::vector<Vector> &fields_e,
                 std::vector<FieldDescriptor> interp,
                 std::vector<int> qfarg_to_field,
                 std::vector<InterpolationPairTensor> &dtqmaps,
                 int el,
                 std::vector<DeviceTensor<2>> &fields_qp)
{
   std::cout << dtqmaps[qfarg_to_field[0]].first(0, 4) << "\n";

   for (int arg = 0; arg < interp.size(); arg++)
   {
      const auto B(dtqmaps[qfarg_to_field[arg]].first);
      const auto G(dtqmaps[qfarg_to_field[arg]].second);

      auto [num_dof, num_qp] = B.GetShape();

      const auto field = Reshape(fields_e[qfarg_to_field[arg]].Read(),
                                 interp[arg].size_on_qp, num_dof);

      if (interp[arg].interpolation ==
          FieldDescriptor::InterpolationType::VALUE)
      {
         for (int dof = 0; dof < num_dof; dof++)
         {
            for (int vdim = 0; vdim < interp[arg].vdim; vdim++)
            {
               for (int qp = 0; qp < num_qp; qp++)
               {
                  double a = fields_qp[arg](vdim, qp);
                  double b = B(dof, qp);
                  double c = field(vdim, dof);
                  // fields_qp[arg](vdim, qp) += B(dof, qp) * field(vdim, dof);
               }
            }
         }
      }
   }
}

std::vector<Field>::const_iterator find_name(const std::vector<Field> &fields,
                                             const std::string &input_name)
{
   auto it = std::find_if(fields.begin(), fields.end(), [&](const Field &field)
   {
      return field.second == input_name;
   });

   return it;
}

int find_name_idx(const std::vector<Field> &fields,
                  const std::string &input_name)
{
   std::vector<Field>::const_iterator it = find_name(fields, input_name);
   if (it == fields.end())
   {
      return -1;
   }
   return (it - fields.begin());
}

class DifferentiableForm // : public BlockNonlinearForm?
{
public:
   DifferentiableForm(std::vector<Field> solutions,
                      std::vector<Field> parameters)
      : solutions(solutions), parameters(parameters)
   {
      dim = GetFESpace(solutions[0]).GetFE(0)->GetDim();
      fields_e.resize(solutions.size() + parameters.size());
      fields.insert(fields.end(), solutions.begin(), solutions.end());
      fields.insert(fields.end(), parameters.begin(), parameters.end());
   }

   template <typename qf_type>
   void AddQFunctionIntegrator(QFunction<qf_type> &qf,
                               const IntegrationRule &ir)
   {
      std::cout << "adding quadrature function with quadrature rule "
                << "\n";

      auto it = find_name(solutions, qf.outputs[0].name);
      if (it != solutions.end())
      {
         if (test_space == nullptr)
         {
            test_space = &GetFESpace(*it);
         }
         else if (test_space != &GetFESpace(*it))
         {
            MFEM_ABORT("");
         }
         qf.outputs[0].size_on_qp = GetSizeOnQP(qf.outputs[0], *it);
      }
      else
      {
         MFEM_ABORT("");
      }

      std::vector<int> qfarg_to_field(qf.inputs.size());
      auto found_callback = [&](int i, int idx, Field &f, int offset)
      {
         int sz = GetSizeOnQP(qf.inputs[i], f);
         qf.inputs[i].dim = GetDimension(f);
         qf.inputs[i].vdim = GetVDim(f);
         qf.inputs[i].size_on_qp = sz;
         qfarg_to_field[i] = idx;
         std::cout << "qf argument " << qf.inputs[i] << " added"
                   << "\n";
      };

      for (int i = 0; i < qfarg_to_field.size(); i++)
      {
         int idx;
         if ((idx = find_name_idx(fields, qf.inputs[i].name)) != -1)
         {
            found_callback(i, idx, fields[idx], 0);
         }
         else
         {
            MFEM_ABORT("error");
         }
      }

      const int num_el = test_space->GetNE();
      const int num_dofs = test_space->GetFE(0)->GetDof();
      const int num_qp = ir.GetNPoints();
      const int test_vdim = test_space->GetFE(0)->GetVDim();

      // Allocate memory for fields on quadrature points
      std::vector<Vector> fields_qp_mem;
      for (int i = 0; i < qf.inputs.size(); i++)
      {
         const int s = qf.inputs[i].size_on_qp * num_qp * num_el;
         fields_qp_mem.emplace_back(Vector(s));
      }

      Vector residual_qp_mem(qf.outputs[0].size_on_qp * num_qp * num_el);

      residual_e.SetSize(
         test_space->GetElementRestriction(ElementDofOrdering::NATIVE)
         ->Height());

      using qf_args = typename create_function_signature<
                      decltype(&qf_type::operator())>::type::parameter_types;

      // This tuple contains objects of every QFunction::func function
      // parameter which might have to be resized.
      qf_args args{};
      allocate_qf_args(args, qf.inputs);

      // Duplicate B/G and assume only a single element type for now
      std::vector<InterpolationPair> dtqmaps;
      for (const auto &field : fields)
      {
         const auto map =
            GetFESpace(field).GetFE(0)->GetDofToQuad(ir, DofToQuad::FULL);
         dtqmaps.push_back({map.B, map.G});
      }

      auto r = [&, args, qfarg_to_field, fields_qp_mem, residual_qp_mem, dtqmaps,
                   test_vdim, num_dofs, num_qp, num_el](Vector &y) mutable
      {

         dtqmaps[1].first.Print(std::cout, dtqmaps[1].first.Size()); // ASAN

         std::vector<InterpolationPairTensor> dtqmaps_tensor;
         for (const auto &map : dtqmaps)
         {
            dtqmaps_tensor.push_back(
            {
               Reshape(map.first.Read(), num_dofs, num_qp),
               Reshape(map.second.Read(), num_dofs, num_qp, dim)
            });
         }

         const auto residual_qp =
            Reshape(residual_qp_mem.ReadWrite(), test_vdim, dim, num_qp, num_el);

         // Fields interpolated to the quadrature points in the order of quadrature
         // function arguments
         std::vector<DeviceTensor<2>> fields_qp;
         for (int i = 0; i < qf.inputs.size(); i++)
         {
            fields_qp.emplace_back(DeviceTensor<2>(
                                      fields_qp_mem[i].ReadWrite(), qf.inputs[i].size_on_qp, num_qp));
         }

         std::cout << "begin element loop\n";
         for (int el = 0; el < num_el; el++)
         {
            std::cout << "element " << el << "\n";

            std::cout << Reshape(dtqmaps[qfarg_to_field[0]].first.Read(), num_dofs,
                                 num_qp)(0, 4) << "\n";
            exit(0);

            // B
            interpolate(fields_e, qf.inputs, qfarg_to_field, dtqmaps_tensor, el,
                        fields_qp);

            for (int qp = 0; qp < num_qp; qp++)
            {
               auto f_qp = apply_qf(qf.func, args, fields_qp, qp);
            }

            // B^T
            // integrate(...);
            std::cout << "\n";
         }

         std::cout << "end element loop\n";
      };

      // replace with emplace_back
      residuals.push_back(r);
   }

   void Mult(const Vector &x, Vector &y) const
   {
      // ASSUME T-Vectors == L-Vectors FOR NOW

      // P
      // prolongation

      // G
      // TODO: Combine those and use fields?
      element_restriction(solutions, x, fields_e);
      element_restriction(parameters, solutions.size(), fields_e);

      // BEGIN GPU
      // B^T Q B x
      residual_e = 0.0;
      for (int i = 0; i < residuals.size(); i++)
      {
         residuals[i](residual_e);
      }
      // END GPU

      // G^T
      element_restriction_transpose(residual_e, test_space, y);

      // P^T
      // prolongation_transpose
   }

   int dim;
   std::vector<Field> solutions;
   std::vector<Field> parameters;

   // solutions and parameters
   std::vector<Field> fields;

   std::vector<std::function<void(Vector &)>> residuals;
   const FiniteElementSpace *test_space = nullptr;
   std::vector<FieldDescriptor> test_space_descriptors;
   mutable Vector residual_e;
   mutable std::vector<Vector> fields_e;
};

int main()
{
   Mpi::Init();

   int dim = 2;
   int polynomial_order = 2;
   int num_elements = 1;

   Mesh mesh_serial = Mesh::MakeCartesian2D(num_elements, num_elements,
                                            Element::Type::QUADRILATERAL);

   ParMesh mesh(MPI_COMM_WORLD, mesh_serial);
   mesh.SetCurvature(1);

   mesh_serial.Clear();

   H1_FECollection h1fec(polynomial_order, dim);
   ParFiniteElementSpace h1fes(&mesh, &h1fec);

   const IntegrationRule &ir =
      IntRules.Get(h1fes.GetFE(0)->GetGeomType(), h1fec.GetOrder() * 2);

   ParGridFunction u(&h1fes);
   ParGridFunction rho(&h1fes);

   rho = 0.123;

   // We really need statically sized objects here
   // auto foo = [](double u, Vector grad_u, double rho)
   auto foo = [](Vector x)
   {
      x.Print(std::cout, x.Size());
      return x;
   };

   QFunction qf
   {
      foo,
      // Inputs
      {
         {FieldDescriptor::InterpolationType::VALUE, "coordinates"}
         // {FieldDescriptor::InterpolationType::GRADIENT, "coordinates"}
         // {FieldDescriptor::InterpolationType::VALUE, "potential"},
         // {FieldDescriptor::InterpolationType::GRADIENT, "potential"},
         // {FieldDescriptor::InterpolationType::VALUE, "material1_density"}
      },
      // Output(s) "integrated against (gradient of) test function"
      {{FieldDescriptor::InterpolationType::VALUE, "potential"}}};

   DifferentiableForm dop(
      // Solutions
      // TODO: Describe only with FESpace?
   {
      // {u_space, "potential", data = nullptr}
      {&u, "potential"}
   },
   // Parameters
   {
      // {rho_space, "material1_density", data = rho}
      {mesh.GetNodes(), "coordinates"},
      {&rho, "material1_density"}
   });

   dop.AddQFunctionIntegrator(qf, ir);

   Vector &x = u.GetTrueVector();
   Vector y(x.Size());
   x = 0.0;
   dop.Mult(x, y);

   y.Print(std::cout, y.Size());

   return 0;
}
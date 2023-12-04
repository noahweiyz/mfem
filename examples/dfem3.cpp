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

struct DofToQuadTensors
{
   DeviceTensor<2, const double> B;
   DeviceTensor<3, const double> G;
};

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
   enum Interpolation { NONE, WEIGHTS, VALUE, GRADIENT, CURL } interpolation;

   FieldDescriptor(Interpolation interp, std::string name, int size_on_qp = -1)
      : interpolation(interp), name(name), size_on_qp(size_on_qp) {}

   std::string name;

   int size_on_qp;

   int dim;

   int vdim;

   friend std::ostream &operator<<(std::ostream &os, const FieldDescriptor &fd)
   {
      os << fd.name << " ";
      switch (fd.interpolation)
      {
         case Interpolation::NONE:
            os << "none";
            break;
         case Interpolation::WEIGHTS:
            os << "weights";
            break;
         case Interpolation::VALUE:
            os << "value";
            break;
         case Interpolation::GRADIENT:
            os << "gradient";
            break;
         case Interpolation::CURL:
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
      case FieldDescriptor::Interpolation::VALUE:
         return GetVDim(field);
         break;
      case FieldDescriptor::Interpolation::GRADIENT:
         return GetVDim(field) * GetDimension(field);
         break;
      default:
         MFEM_ABORT("can't get size on quadrature point for field descriptor");
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

template <
   typename quadrature_function_type,
   unsigned long num_inputs,
   unsigned long num_outputs>
struct QFunction
{
   quadrature_function_type func;
   std::array<FieldDescriptor, num_inputs> inputs;
   std::array<FieldDescriptor, num_outputs> outputs;
};

template <typename T, unsigned long num_inputs, unsigned long num_outputs>
QFunction(T, std::array<FieldDescriptor, num_inputs>,
          std::array<FieldDescriptor, num_outputs>)
-> QFunction<T, num_inputs, num_outputs>;

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
      auto fes = GetFESpace(fields[i]);
      const auto R = fes.GetElementRestriction(ElementDofOrdering::NATIVE);
      const int height = R->Height();
      fields_e[i + offset].SetSize(height);

      R->Mult(GetGridFunction(fields[i]), fields_e[i + offset]);
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
   if (input.interpolation == FieldDescriptor::Interpolation::GRADIENT)
   {
      arg.SetSize(input.size_on_qp / input.dim);
   }
}

template <typename qf_args, unsigned long num_inputs, std::size_t... i>
void allocate_qf_args_impl(qf_args &args,
                           std::array<FieldDescriptor, num_inputs> inputs,
                           std::index_sequence<i...>)
{
   (allocate_qf_arg(inputs[i], std::get<i>(args)), ...);
}

template <typename qf_args, unsigned long num_inputs>
void allocate_qf_args(qf_args &args,
                      std::array<FieldDescriptor, num_inputs> inputs)
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

template <unsigned long num_inputs>
void map_to_quadrature_data(
   int element_idx,
   const std::vector<Vector> &fields_e,
   std::array<FieldDescriptor, num_inputs> qfinputs,
   std::vector<int> qfarg_to_field,
   std::vector<DofToQuadTensors> &dtqmaps,
   DeviceTensor<1, const double> integration_weights,
   std::vector<DeviceTensor<2>> &fields_qp)
{
   for (int arg = 0; arg < qfinputs.size(); arg++)
   {
      if (qfinputs[arg].interpolation ==
          FieldDescriptor::Interpolation::VALUE)
      {
         const auto B(dtqmaps[qfarg_to_field[arg]].B);
         auto [num_dof, num_qp] = B.GetShape();
         const int vdim = qfinputs[arg].vdim;
         const int element_offset = element_idx * num_dof * vdim;
         const auto field = Reshape(fields_e[qfarg_to_field[arg]].Read() +
                                    element_offset,
                                    num_dof, vdim);

         for (int vd = 0; vd < vdim; vd++)
         {
            for (int qp = 0; qp < num_qp; qp++)
            {
               double acc = 0.0;
               for (int dof = 0; dof < num_dof; dof++)
               {
                  acc += B(qp, dof) * field(dof, vd);
               }
               fields_qp[arg](vd, qp) = acc;
            }
         }
      }
      else if (qfinputs[arg].interpolation ==
               FieldDescriptor::Interpolation::GRADIENT)
      {
         const auto G(dtqmaps[qfarg_to_field[arg]].G);
         const auto [num_dof, dim, num_qp] = G.GetShape();
         const int vdim = qfinputs[arg].vdim;
         const int element_offset = element_idx * num_dof * vdim;
         const auto field = Reshape(fields_e[qfarg_to_field[arg]].Read() +
                                    element_offset,
                                    num_dof, vdim);

         auto f = Reshape(&fields_qp[arg][0], vdim, dim, num_qp);
         for (int qp = 0; qp < num_qp; qp++)
         {
            for (int vd = 0; vd < vdim; vd++)
            {
               for (int d = 0; d < dim; d++)
               {
                  double acc = 0.0;
                  for (int dof = 0; dof < num_dof; dof++)
                  {
                     acc += G(qp, d, dof) * field(dof, vd);
                  }
                  f(vd, d, qp) = acc;
               }
            }
         }
      }
      else if (qfinputs[arg].interpolation ==
               FieldDescriptor::Interpolation::WEIGHTS)
      {
         const int num_qp = integration_weights.GetShape()[0];
         auto f = Reshape(&fields_qp[arg][0], num_qp);
         for (int qp = 0; qp < num_qp; qp++)
         {
            f(qp) = integration_weights(qp);
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
                      std::vector<Field> parameters,
                      ParMesh &mesh)
      : solutions(solutions), parameters(parameters), mesh(mesh)
   {
      dim = mesh.Dimension();
      fields_e.resize(solutions.size() + parameters.size());
      fields.insert(fields.end(), solutions.begin(), solutions.end());
      fields.insert(fields.end(), parameters.begin(), parameters.end());
   }

   template <
      typename qf_type,
      unsigned long num_inputs,
      unsigned long num_outputs>
   void AddQFunctionIntegrator(QFunction<qf_type, num_inputs, num_outputs> &qf,
                               const IntegrationRule &ir)
   {
      std::cout << "adding quadrature function with quadrature rule "
                << "\n";

      int test_space_field_idx;
      int residual_size_on_qp;

      if ((test_space_field_idx = find_name_idx(fields, qf.outputs[0].name)) != -1)
      {
         if (test_space == nullptr)
         {
            test_space = &GetFESpace(solutions[test_space_field_idx]);
         }
         else if (test_space != &GetFESpace(solutions[test_space_field_idx]))
         {
            MFEM_ABORT("can't add quadrature function with different test space");
         }
         residual_size_on_qp = GetSizeOnQP(qf.outputs[0],
                                           solutions[test_space_field_idx]);
      }
      else
      {
         if (qf.outputs[0].size_on_qp == -1)
         {
            MFEM_ABORT("need to set size on quadrature point for test space that doesn't refer to a field");
         }
         residual_size_on_qp = qf.outputs[0].size_on_qp;
      }

      std::vector<int> qfarg_to_field(qf.inputs.size());
      for (int i = 0; i < qfarg_to_field.size(); i++)
      {
         int idx;
         // TODO: This isn't ideal...
         if (qf.inputs[i].interpolation == FieldDescriptor::Interpolation::WEIGHTS)
         {
            qf.inputs[i].dim = 1;
            qf.inputs[i].vdim = 1;
            qf.inputs[i].size_on_qp = 1;
            qfarg_to_field[i] = -1;
         }
         else if ((idx = find_name_idx(fields, qf.inputs[i].name)) != -1)
         {
            int sz = GetSizeOnQP(qf.inputs[i], fields[idx]);
            qf.inputs[i].dim = GetDimension(fields[idx]);
            qf.inputs[i].vdim = GetVDim(fields[idx]);
            qf.inputs[i].size_on_qp = sz;
            qfarg_to_field[i] = idx;
         }
         else
         {
            MFEM_ABORT("can't find field for " << qf.inputs[i].name);
         }
      }

      const int num_el = mesh.GetNE();
      const int num_qp = ir.GetNPoints();

      // Allocate memory for fields on quadrature points
      std::vector<Vector> fields_qp_mem;
      for (int i = 0; i < qf.inputs.size(); i++)
      {
         const int s = qf.inputs[i].size_on_qp * num_qp * num_el;
         fields_qp_mem.emplace_back(Vector(s));
      }

      Vector residual_qp_mem(residual_size_on_qp * num_qp * num_el);
      if (test_space)
      {
         residual_e.SetSize(
            test_space->GetElementRestriction(ElementDofOrdering::NATIVE)
            ->Height());
      }
      else
      {
         residual_e.SetSize(residual_qp_mem.Size());
      }

      using qf_args = typename create_function_signature<
                      decltype(&qf_type::operator())>::type::parameter_types;

      // This tuple contains objects of every QFunction::func function
      // parameter which might have to be resized.
      qf_args args{};
      allocate_qf_args(args, qf.inputs);

      Array<double> integration_weights_mem = ir.GetWeights();

      // Duplicate B/G and assume only a single element type for now
      std::vector<DofToQuad> dtqmaps;
      for (const auto &field : fields)
      {
         dtqmaps.emplace_back(GetFESpace(field).GetFE(0)->GetDofToQuad(ir,
                                                                       DofToQuad::FULL));
      }

      residuals.emplace_back(
         [&, args, qfarg_to_field, fields_qp_mem, residual_qp_mem, residual_size_on_qp,
             test_space_field_idx,
             dtqmaps, integration_weights_mem, num_qp, num_el](Vector &y_e) mutable
      {
         std::vector<DofToQuadTensors> dtqmaps_tensor;
         for (const auto &map : dtqmaps)
         {
            dtqmaps_tensor.push_back(
            {
               Reshape(map.B.Read(), num_qp, map.ndof),
               Reshape(map.G.Read(), num_qp, dim, map.ndof)
            });
         }

         const auto residual_qp =
            Reshape(residual_qp_mem.ReadWrite(), residual_size_on_qp, num_qp, num_el);

         // Fields interpolated to the quadrature points in the order of quadrature
         // function arguments
         std::vector<DeviceTensor<2>> fields_qp;
         for (int i = 0; i < qf.inputs.size(); i++)
         {
            fields_qp.emplace_back(DeviceTensor<2>(
                                      fields_qp_mem[i].ReadWrite(), qf.inputs[i].size_on_qp, num_qp));
         }

         DeviceTensor<1, const double> integration_weights(integration_weights_mem.Read(), num_qp);

         std::cout << "begin element loop\n";
         for (int el = 0; el < num_el; el++)
         {
            std::cout << "element " << el << "\n";

            // B
            // prepare fields on quadrature points
            map_to_quadrature_data(el, fields_e, qf.inputs, qfarg_to_field, dtqmaps_tensor,
                                   integration_weights, fields_qp);

            for (int qp = 0; qp < num_qp; qp++)
            {
               auto r_qp = Reshape(&residual_qp(0, qp, el), residual_size_on_qp);

               auto f_qp = apply_qf(qf.func, args, fields_qp, qp);

               for (int i = 0; i < residual_size_on_qp; i++)
               {
                  r_qp(i) = f_qp(i);
               }
            }

            // B^T
            if (test_space_field_idx != -1)
            {
               // integrate(...);
               std::cout << "\n";
            }
            else
            {
               y_e = residual_qp_mem;
            }
         }

         std::cout << "end element loop\n";
      });
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

      if (test_space)
      {
         // G^T
         element_restriction_transpose(residual_e, test_space, y);
         // P^T
         // prolongation_transpose
      }
      else
      {
         // No test space means we return quadrature data
         y = residual_e;
      }
   }

   std::vector<Field> solutions;
   std::vector<Field> parameters;
   ParMesh &mesh;
   int dim;

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
   int polynomial_order = 1;
   int num_elements = 1;

   Mesh mesh_serial = Mesh::MakeCartesian2D(num_elements, num_elements,
                                            Element::Type::QUADRILATERAL);

   ParMesh mesh(MPI_COMM_WORLD, mesh_serial);
   mesh.SetCurvature(1);

   mesh_serial.Clear();

   H1_FECollection h1fec(polynomial_order, dim);
   ParFiniteElementSpace h1fes(&mesh, &h1fec);

   const IntegrationRule &ir =
      IntRules.Get(h1fes.GetFE(0)->GetGeomType(), h1fec.GetOrder() + 1);

   ParGridFunction u(&h1fes);
   ParGridFunction rho(&h1fes);

   rho = 0.123;

   // We really need statically sized objects here
   auto foo = [](Vector x, DenseMatrix J, double w)
   {
      // spatially varying coefficient
      x *= std::abs(J.Det()) * w;
      return x;
   };

   QFunction qf
   {
      foo,
      // Inputs
      std::array{
         FieldDescriptor{FieldDescriptor::Interpolation::VALUE, "coordinates"},
         FieldDescriptor{FieldDescriptor::Interpolation::GRADIENT, "coordinates"},
         FieldDescriptor{FieldDescriptor::Interpolation::WEIGHTS, "integration_weights"},
      },
      // Output(s) "integrated against (gradient of) test function"
      std::array{FieldDescriptor{FieldDescriptor::Interpolation::VALUE, "potential"}}
      // std::array{FieldDescriptor{FieldDescriptor::Interpolation::NONE, "quadrature_data", 2}}
   };

   DifferentiableForm dop(
      // Solutions
   {{&u, "potential"}},
   // Parameters
   {
      {mesh.GetNodes(), "coordinates"},
   },
   mesh);

   dop.AddQFunctionIntegrator(qf, ir);

   Vector &x = u.GetTrueVector();
   Vector y(x.Size());
   x = 0.0;
   dop.Mult(x, y);

   y.Print(std::cout, y.Size());

   return 0;
}
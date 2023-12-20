#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>
#include <type_traits>

#include <general/forall.hpp>
#include <linalg/tensor.hpp>
#include <mfem.hpp>

template <typename T>
constexpr auto get_type_name() -> std::string_view
{
#if defined(__clang__)
   constexpr auto prefix = std::string_view {"[T = "};
   constexpr auto suffix = "]";
   constexpr auto function = std::string_view{__PRETTY_FUNCTION__};
#elif defined(__GNUC__)
   constexpr auto prefix = std::string_view {"with T = "};
   constexpr auto suffix = "; ";
   constexpr auto function = std::string_view{__PRETTY_FUNCTION__};
#elif defined(_MSC_VER)
   constexpr auto prefix = std::string_view {"get_type_name<"};
   constexpr auto suffix = ">(void)";
   constexpr auto function = std::string_view{__FUNCSIG__};
#else
#error Unsupported compiler
#endif

   const auto start = function.find(prefix) + prefix.size();
   const auto end = function.find(suffix);
   const auto size = end - start;

   return function.substr(start, size);
}

using namespace mfem;

using mfem::internal::tensor;

int enzyme_dup;
int enzyme_dupnoneed;
int enzyme_out;
int enzyme_const;

template <typename return_type, typename... T>
return_type __enzyme_fwddiff(void *, T...);

template <typename return_type, typename... T>
return_type __enzyme_autodiff(void *, T...);

struct Independent {};
struct Dependent {};

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

class FieldDescriptor
{
public:
   FieldDescriptor(std::string name) : name(name) {};

   std::string name;

   int size_on_qp;

   int dim;

   int vdim;
};

class None : public FieldDescriptor
{
public:
   None(std::string name) : FieldDescriptor(name) {};
};

class Weight : public FieldDescriptor
{
public:
   Weight(std::string name) : FieldDescriptor(name) {};
};

class Value : public FieldDescriptor
{
public:
   Value(std::string name) : FieldDescriptor(name) {};
};

class Gradient : public FieldDescriptor
{
public:
   Gradient(std::string name) : FieldDescriptor(name) {};
};

template <typename field_descriptor_type>
int GetSizeOnQP(const field_descriptor_type &fd, const Field &f)
{
   if constexpr (std::is_same_v<field_descriptor_type, Value>)
   {
      return GetVDim(f);
   }
   else if constexpr (std::is_same_v<field_descriptor_type, Gradient>)
   {
      return GetVDim(f) * GetDimension(f);
   }
   else
   {
      MFEM_ABORT("can't get size on quadrature point for field descriptor");
   }
}

template <typename quadrature_function_type, typename input_type,
          typename output_type>
struct ElementOperator;

template <typename quadrature_function_type, typename... input_types,
          typename... output_types>
struct ElementOperator<quadrature_function_type, std::tuple<input_types...>,
          std::tuple<output_types...>>
{
   quadrature_function_type func;
   std::tuple<input_types...> inputs;
   std::tuple<output_types...> outputs;
   constexpr ElementOperator(quadrature_function_type func,
                             std::tuple<input_types...> inputs,
                             std::tuple<output_types...> outputs)
      : func(func), inputs(inputs), outputs(outputs) {}
};

template <typename quadrature_function_type, typename... input_types,
          typename... output_types>
ElementOperator(quadrature_function_type, std::tuple<input_types...>,
                std::tuple<output_types...>)
-> ElementOperator<quadrature_function_type, std::tuple<input_types...>,
   std::tuple<output_types...>>;

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

template <typename qf_args_t, std::size_t... Is>
auto create_enzyme_args(qf_args_t qf_args, qf_args_t &qf_shadow_args,
                        std::index_sequence<Is...>)
{
   return std::tuple_cat(std::tie(enzyme_dup, std::get<Is>(qf_args),
                                  std::get<Is>(qf_shadow_args))...);
}

template <typename qf_type, typename arg_type>
auto enzyme_fwddiff_apply(qf_type qf, arg_type &&args, arg_type &&shadow_args)
{
   // std::cout << "shadow\n"
   //           << get_type_name<decltype(shadow_args)>() << std::endl;

   auto arg_indices =
      std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<arg_type>>> {};
   auto enzyme_args = create_enzyme_args(args, shadow_args, arg_indices);

   // std::cout << "enzyme_args\n"
   //           << get_type_name<decltype(enzyme_args)>() << std::endl;

   return std::apply(
             [&](auto &&...args)
   {
      using qf_return_type = typename create_function_signature<
                             decltype(&qf_type::operator())>::type::return_type;
      return __enzyme_fwddiff<qf_return_type>((void *)+qf, args...);
   },
   enzyme_args);
}

void allocate_qf_arg(const Weight &input, double &arg)
{
   // no op
}

void allocate_qf_arg(const Value &input, double &arg)
{
   // no op
}

void allocate_qf_arg(const Value &input, Vector &arg)
{
   arg.SetSize(input.size_on_qp);
}

void allocate_qf_arg(const Gradient &input, Vector &arg)
{
   arg.SetSize(input.size_on_qp / input.vdim);
}

void allocate_qf_arg(const Gradient &input, DenseMatrix &arg)
{
   arg.SetSize(input.size_on_qp / input.vdim);
}

void allocate_qf_arg(const Gradient &input, tensor<double, 2> &arg)
{
   // no op
}

void allocate_qf_arg(const Gradient &input, tensor<double, 2, 2> &arg)
{
   // no op
}

template <typename qf_args, typename input_type, std::size_t... i>
void allocate_qf_args_impl(qf_args &args, input_type inputs,
                           std::index_sequence<i...>)
{
   (allocate_qf_arg(std::get<i>(inputs), std::get<i>(args)), ...);
}

template <typename qf_args, typename input_type>
void allocate_qf_args(qf_args &args, input_type inputs)
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

void prepare_qf_arg(const DeviceTensor<1> &u, tensor<double, 2> &arg)
{
   for (int i = 0; i < u.GetShape()[0]; i++)
   {
      arg(i) = u(i);
   }
}

void prepare_qf_arg(const DeviceTensor<1> &u, tensor<double, 2, 2> &arg)
{
   for (int i = 0; i < 2; i++)
   {
      for (int j = 0; j < 2; j++)
      {
         arg(i, j) = u[(i * 2) + j];
      }
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

Vector prepare_qf_result(tensor<double, 2> x)
{
   Vector r(2);
   for (size_t i = 0; i < 2; i++)
   {
      r(i) = x(i);
   }
   return r;
}

template <typename qf_type, typename qf_args>
auto apply_qf(const qf_type &qf, qf_args &args, std::vector<DeviceTensor<2>> &u,
              int qp)
{
   prepare_qf_args(qf, u, args, qp,
                   std::make_index_sequence<std::tuple_size_v<qf_args>> {});

   return prepare_qf_result(std::apply(qf, args));
}

template <typename qf_type, typename qf_args>
auto apply_qf_fwddiff(const qf_type &qf,
                      qf_args &args,
                      std::vector<DeviceTensor<2>> &u,
                      qf_args &shadow_args,
                      std::vector<DeviceTensor<2>> &v,
                      int qp)
{
   prepare_qf_args(qf, u, args, qp,
                   std::make_index_sequence<std::tuple_size_v<qf_args>> {});

   prepare_qf_args(qf, v, shadow_args, qp,
                   std::make_index_sequence<std::tuple_size_v<qf_args>> {});

   return prepare_qf_result(enzyme_fwddiff_apply(qf, args, shadow_args));
}

template <typename input_type>
void map_field_to_quadrature_data(
   DeviceTensor<2> field_qp, int element_idx, DofToQuadTensors &dtqmaps,
   const Vector &field_e, input_type &input,
   DeviceTensor<1, const double> integration_weights)
{
   if constexpr (std::is_same_v<
                 typename std::remove_reference<decltype(input)>::type,
                 Value>)
   {
      const auto B(dtqmaps.B);
      auto [num_qp, num_dof] = B.GetShape();
      const int vdim = input.vdim;
      const int element_offset = element_idx * num_dof * vdim;
      const auto field = Reshape(field_e.Read() + element_offset, num_dof, vdim);

      for (int vd = 0; vd < vdim; vd++)
      {
         for (int qp = 0; qp < num_qp; qp++)
         {
            double acc = 0.0;
            for (int dof = 0; dof < num_dof; dof++)
            {
               acc += B(qp, dof) * field(dof, vd);
            }
            field_qp(vd, qp) = acc;
         }
      }
   }
   else if constexpr (std::is_same_v<typename std::remove_reference<
                      decltype(input)>::type,
                      Gradient>)
   {
      const auto G(dtqmaps.G);
      const auto [num_qp, dim, num_dof] = G.GetShape();
      const int vdim = input.vdim;
      const int element_offset = element_idx * num_dof * vdim;
      const auto field = Reshape(field_e.Read() + element_offset, num_dof, vdim);

      auto f = Reshape(&field_qp[0], vdim, dim, num_qp);
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
   else if constexpr (std::is_same_v<typename std::remove_reference<
                      decltype(input)>::type,
                      Weight>)
   {
      const int num_qp = integration_weights.GetShape()[0];
      auto f = Reshape(&field_qp[0], num_qp);
      for (int qp = 0; qp < num_qp; qp++)
      {
         f(qp) = integration_weights(qp);
      }
   }
}

template <int num_qfinputs, typename input_type, std::size_t... i>
void map_fields_to_quadrature_data(
   std::vector<DeviceTensor<2>> &fields_qp, int element_idx,
   const std::vector<Vector> &fields_e,
   std::array<int, num_qfinputs> qfarg_to_field,
   std::vector<DofToQuadTensors> &dtqmaps,
   DeviceTensor<1, const double> integration_weights, input_type &qfinputs,
   std::index_sequence<i...>)
{
   (map_field_to_quadrature_data(fields_qp[i], element_idx,
                                 dtqmaps[qfarg_to_field[i]], fields_e[qfarg_to_field[i]],
                                 std::get<i>(qfinputs), integration_weights),
    ...);
}

template <typename output_fd_type>
void map_quadrature_data_to_fields(Vector &y_e, int element_idx, int num_el,
                                   DeviceTensor<3, double> residual_qp,
                                   output_fd_type output_fd,
                                   std::vector<DofToQuadTensors> &dtqmaps,
                                   int test_space_field_idx)
{
   // assuming the quadrature point residual has to "play nice with
   // the test function"
   if constexpr (std::is_same_v<
                 typename std::remove_reference<decltype(output_fd)>::type,
                 Value>)
   {
      const auto B(dtqmaps[test_space_field_idx].B);
      const auto [num_qp, num_dof] = B.GetShape();
      const int vdim = output_fd.vdim;
      auto C = Reshape(&residual_qp(0, 0, element_idx), vdim, num_qp);
      auto y = Reshape(y_e.ReadWrite(), num_dof, vdim, num_el);
      for (int dof = 0; dof < num_dof; dof++)
      {
         for (int vd = 0; vd < vdim; vd++)
         {
            double acc = 0.0;
            for (int qp = 0; qp < num_qp; qp++)
            {
               // |JxW| is assumed to be in C
               acc += B(qp, dof) * C(vd, qp);
            }
            y(dof, vd, element_idx) += acc;
         }
      }
   }
   else if constexpr (std::is_same_v<typename std::remove_reference<
                      decltype(output_fd)>::type,
                      Gradient>)
   {
      const auto G(dtqmaps[test_space_field_idx].G);
      const auto [num_qp, dim, num_dof] = G.GetShape();
      const int vdim = output_fd.vdim;
      auto C = Reshape(&residual_qp(0, 0, element_idx), vdim, dim, num_qp);
      auto y = Reshape(y_e.ReadWrite(), num_dof, vdim, num_el);
      for (int dof = 0; dof < num_dof; dof++)
      {
         for (int vd = 0; vd < vdim; vd++)
         {
            double acc = 0.0;
            for (int d = 0; d < dim; d++)
            {
               for (int qp = 0; qp < num_qp; qp++)
               {
                  acc += G(qp, d, dof) * C(vd, d, qp);
               }
            }
            y(dof, vd, element_idx) += acc;
         }
      }
   }
   else
   {
      MFEM_ABORT("implement this");
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

template <size_t num_qfinputs, typename input_type, std::size_t... i>
void map_qfarg_to_field(std::vector<Field> &fields,
                        std::array<int, num_qfinputs> &map, input_type &inputs,
                        std::index_sequence<i...>)
{
   auto f = [&](auto &input, auto &map)
   {
      int idx;
      if constexpr (std::is_same_v<
                    typename std::remove_reference<decltype(input)>::type,
                    Weight>)
      {
         input.dim = 1;
         input.vdim = 1;
         input.size_on_qp = 1;
         map = -1;
      }
      else if ((idx = find_name_idx(fields, input.name)) != -1)
      {
         int sz = GetSizeOnQP(input, fields[idx]);
         input.dim = GetDimension(fields[idx]);
         input.vdim = GetVDim(fields[idx]);
         input.size_on_qp = sz;
         map = idx;
      }
      else
      {
         MFEM_ABORT("can't find field for " << input.name);
      }
   };

   (f(std::get<i>(inputs), map[i]), ...);
}

template <typename input_type, std::size_t... i>
void map_inputs_to_memory(std::vector<DeviceTensor<2>> &fields_qp,
                          std::vector<Vector> &fields_qp_mem, int num_qp,
                          input_type &inputs, std::index_sequence<i...>)
{
   auto f = [&](auto &input, auto &field_qp_mem)
   {
      fields_qp.emplace_back(
         DeviceTensor<2>(field_qp_mem.ReadWrite(), input.size_on_qp, num_qp));
   };

   (f(std::get<i>(inputs), fields_qp_mem[i]), ...);
}

class DifferentiableForm : public Operator
{
public:
   class JacobianOperator : public Operator
   {
   public:
      JacobianOperator(DifferentiableForm &op) : Operator(op.Height()), op(op) {}

      void Mult(const Vector &x, Vector &y) const override
      {
         op.JacobianMult(x, y);
      }

   protected:
      DifferentiableForm &op;
   };

   DifferentiableForm(std::vector<Field> solutions,
                      std::vector<Field> parameters, ParMesh &mesh)
      : solutions(solutions), parameters(parameters), mesh(mesh)
   {
      dim = mesh.Dimension();
      fields.insert(fields.end(), solutions.begin(), solutions.end());
      fields.insert(fields.end(), parameters.begin(), parameters.end());

      fields_e.resize(solutions.size() + parameters.size());
      directions_e.resize(solutions.size());
   }

   template <
      typename qf_type,
      typename input_type,
      typename output_type>
   void AddElementOperator(ElementOperator<qf_type, input_type, output_type> &qf,
                           const IntegrationRule &ir)
   {
      AddElementOperator(qf, ir, Dependent{});
   }

   template <
      typename qf_type,
      typename input_type,
      typename output_type,
      typename dependency_type>
   void AddElementOperator(ElementOperator<qf_type, input_type, output_type> &qf,
                           const IntegrationRule &ir, dependency_type dependency)
   {
      constexpr size_t num_qfinputs = std::tuple_size_v<input_type>;
      // constexpr size_t num_qfoutputs = std::tuple_size_v<output_type>;

      std::cout << "adding quadrature function with quadrature rule "
                << "\n";

      const FiniteElementSpace *test_space = nullptr;
      int test_space_field_idx;
      int residual_size_on_qp;

      auto output_fd = std::get<0>(qf.outputs);
      if ((test_space_field_idx = find_name_idx(fields, output_fd.name)) != -1)
      {
         if (test_space == nullptr)
         {
            test_space = &GetFESpace(solutions[test_space_field_idx]);
         }
         else if (test_space != &GetFESpace(solutions[test_space_field_idx]))
         {
            MFEM_ABORT("can't add quadrature function with different test space");
         }
         residual_size_on_qp =
            GetSizeOnQP(output_fd, solutions[test_space_field_idx]);
         int sz = GetSizeOnQP(output_fd, fields[test_space_field_idx]);
         output_fd.dim = GetDimension(fields[test_space_field_idx]);
         output_fd.vdim = GetVDim(fields[test_space_field_idx]);
         output_fd.size_on_qp = sz;
      }
      else
      {
         if (output_fd.size_on_qp == -1)
         {
            MFEM_ABORT("need to set size on quadrature point for test space"
                       "that doesn't refer to a field");
         }
         residual_size_on_qp = output_fd.size_on_qp;
      }

      std::array<int, num_qfinputs> qfarg_to_field;
      map_qfarg_to_field(fields, qfarg_to_field, qf.inputs,
                         std::make_index_sequence<num_qfinputs> {});

      const int num_el = mesh.GetNE();
      const int num_qp = ir.GetNPoints();

      // All solutions T-vector sizes make up the height of the operator, since
      // they are explicitly provided in Mult() for example.
      this->height = 0;
      for (auto &f : solutions)
      {
         this->height += GetFESpace(f).GetTrueVSize();
      }

      // Since we know the test space of the integrator, we know the output size
      // of the whole operator so it can be set now.
      this->width = test_space->GetTrueVSize();

      // Creating this here allows to call GradientMult even if there are no
      // dependent ElementOperators.
      jacobian_op.reset(new JacobianOperator(*this));

      // Allocate memory for fields on quadrature points
      std::vector<Vector> fields_qp_mem;
      std::apply(
         [&](auto &&...input)
      {
         (fields_qp_mem.emplace_back(
             Vector(input.size_on_qp * num_qp * num_el)),
          ...);
      },
      qf.inputs);

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

      // This tuple contains objects of every ElementOperator::func function
      // parameter which might have to be resized.
      qf_args args{};
      allocate_qf_args(args, qf.inputs);

      Array<double> integration_weights_mem = ir.GetWeights();

      // Duplicate B/G and assume only a single element type for now
      std::vector<DofToQuad> dtqmaps;
      for (const auto &field : fields)
      {
         dtqmaps.emplace_back(
            GetFESpace(field).GetFE(0)->GetDofToQuad(ir, DofToQuad::FULL));
      }

      residual_integrators.emplace_back(
         [&, args, qfarg_to_field, fields_qp_mem, residual_qp_mem,
             residual_size_on_qp, test_space_field_idx, output_fd, dtqmaps,
             integration_weights_mem, num_qp, num_el](Vector &y_e) mutable
      {
         std::vector<DofToQuadTensors> dtqmaps_tensor;
         for (const auto &map : dtqmaps)
         {
            dtqmaps_tensor.push_back(
            {
               Reshape(map.B.Read(), num_qp, map.ndof),
               Reshape(map.G.Read(), num_qp, dim, map.ndof)});
         }

         const auto residual_qp = Reshape(residual_qp_mem.ReadWrite(),
                                          residual_size_on_qp, num_qp, num_el);

         // Fields interpolated to the quadrature points in the order of
         // quadrature function arguments
         std::vector<DeviceTensor<2>> fields_qp;
         map_inputs_to_memory(fields_qp, fields_qp_mem, num_qp, qf.inputs,
                              std::make_index_sequence<num_qfinputs>{});

         DeviceTensor<1, const double> integration_weights(
            integration_weights_mem.Read(), num_qp);

         for (int el = 0; el < num_el; el++)
         {
            // B
            // prepare fields on quadrature points
            map_fields_to_quadrature_data<num_qfinputs>(
               fields_qp, el, fields_e, qfarg_to_field, dtqmaps_tensor,
               integration_weights, qf.inputs,
               std::make_index_sequence<num_qfinputs> {});

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
               // integrate
               map_quadrature_data_to_fields(y_e, el, num_el, residual_qp,
                                             output_fd, dtqmaps_tensor,
                                             test_space_field_idx);
            }
            else
            {
               MFEM_ABORT("implement this");
            }
         }
      });

      if (test_space)
      {
         element_restriction_transpose = [test_space](Vector &r_e, Vector &y)
         {
            test_space->GetElementRestriction(ElementDofOrdering::NATIVE)
            ->MultTranspose(r_e, y);
         };
      }

      if constexpr (std::is_same_v<dependency_type, Dependent>)
      {
         // Allocate memory for directions on quadrature points
         std::vector<Vector> directions_qp_mem;
         std::apply(
            [&](auto &&...input)
         {
            (directions_qp_mem.emplace_back(
                Vector(input.size_on_qp * num_qp * num_el)),
             ...);
         },
         qf.inputs);

         for (auto &v : directions_qp_mem)
         {
            v = 0.0;
         }

         qf_args shadow_args{};
         allocate_qf_args(shadow_args, qf.inputs);

         jacobian_integrators.emplace_back(
            [&, args, shadow_args, qfarg_to_field,
                fields_qp_mem,
                directions_qp_mem, residual_qp_mem,
                residual_size_on_qp,
                test_space_field_idx, output_fd, dtqmaps,
                integration_weights_mem, num_qp,
                num_el](Vector &y_e) mutable
         {
            std::vector<DofToQuadTensors> dtqmaps_tensor;
            for (const auto &map : dtqmaps)
            {
               dtqmaps_tensor.push_back(
               {
                  Reshape(map.B.Read(), num_qp, map.ndof),
                  Reshape(map.G.Read(), num_qp, dim, map.ndof)});
            }

            const auto residual_qp = Reshape(residual_qp_mem.ReadWrite(),
                                             residual_size_on_qp, num_qp, num_el);

            // Fields interpolated to the quadrature points in the order of quadrature
            // function arguments
            std::vector<DeviceTensor<2>> fields_qp;
            map_inputs_to_memory(fields_qp, fields_qp_mem, num_qp, qf.inputs,
                                 std::make_index_sequence<num_qfinputs>{});

            std::vector<DeviceTensor<2>> directions_qp;
            map_inputs_to_memory(directions_qp, directions_qp_mem, num_qp, qf.inputs,
                                 std::make_index_sequence<num_qfinputs>{});

            DeviceTensor<1, const double> integration_weights(
               integration_weights_mem.Read(), num_qp);

            for (int el = 0; el < num_el; el++)
            {
               // B
               // prepare fields on quadrature points
               map_fields_to_quadrature_data<num_qfinputs>(
                  fields_qp, el, fields_e, qfarg_to_field, dtqmaps_tensor,
                  integration_weights, qf.inputs,
                  std::make_index_sequence<num_qfinputs> {});

               constexpr int primary_variable_idx = 0;
               map_field_to_quadrature_data(directions_qp[primary_variable_idx], el,
                                            dtqmaps_tensor[primary_variable_idx],
                                            directions_e[primary_variable_idx],
                                            std::get<primary_variable_idx>(qf.inputs),
                                            integration_weights);

               for (int qp = 0; qp < num_qp; qp++)
               {
                  auto r_qp = Reshape(&residual_qp(0, qp, el), residual_size_on_qp);

                  auto f_qp = apply_qf_fwddiff(qf.func, args, fields_qp, shadow_args,
                                               directions_qp, qp);

                  for (int i = 0; i < residual_size_on_qp; i++)
                  {
                     r_qp(i) = f_qp(i);
                  }
               }

               // B^T
               if (test_space_field_idx != -1)
               {
                  // integrate
                  map_quadrature_data_to_fields(y_e, el, num_el, residual_qp, output_fd,
                                                dtqmaps_tensor, test_space_field_idx);
               }
               else
               {
                  MFEM_ABORT("implement this");
               }
            }
         });
      }
   }

   void Mult(const Vector &x, Vector &y) const override
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
      for (int i = 0; i < residual_integrators.size(); i++)
      {
         residual_integrators[i](residual_e);
      }
      // END GPU

      if (element_restriction_transpose)
      {
         // G^T
         element_restriction_transpose(residual_e, y);
         // P^T
         // prolongation_transpose
      }
      else
      {
         // No element_restriction_transpose
         // -> no test space
         // -> return quadrature data
         y = residual_e;
      }

      y.SetSubVector(ess_tdof_list, 0.0);
   }

   void SetEssentialTrueDofs(const Array<int> &l) { l.Copy(ess_tdof_list); }

   void JacobianMult(const Vector &x, Vector &y) const
   {
      // apply essential bcs
      current_direction_t = x;
      current_direction_t.SetSubVector(ess_tdof_list, 0.0);

      element_restriction(solutions, current_direction_t, directions_e);
      element_restriction(solutions, current_state_t, fields_e);
      element_restriction(parameters, solutions.size(), fields_e);

      // BEGIN GPU
      // B^T Q B x
      Vector &jvp_e = residual_e;

      jvp_e = 0.0;
      for (int i = 0; i < jacobian_integrators.size(); i++)
      {
         jacobian_integrators[i](jvp_e);
      }
      // END GPU

      // G^T
      element_restriction_transpose(jvp_e, y);
      // P^T
      // prolongation_transpose

      // re-assign the essential degrees of freedom on the final output vector.
      for (int i = 0; i < ess_tdof_list.Size(); i++)
      {
         y[ess_tdof_list[i]] = x[ess_tdof_list[i]];
      }
   }

   Operator &GetGradient(const Vector &x) const override
   {
      current_state_t = x;
      return *jacobian_op;
   }

   std::vector<Field> solutions;
   std::vector<Field> parameters;
   ParMesh &mesh;
   int dim;

   // solutions and parameters
   std::vector<Field> fields;

   std::vector<std::function<void(Vector &)>> residual_integrators;
   std::vector<std::function<void(Vector &)>> jacobian_integrators;
   std::function<void(Vector &, Vector &)> element_restriction_transpose;

   mutable Vector residual_e;
   mutable std::vector<Vector> fields_e;
   mutable std::vector<Vector> directions_e;

   std::shared_ptr<JacobianOperator> jacobian_op;
   mutable Vector current_direction_t, current_state_t;

   Array<int> ess_tdof_list;
};

int main()
{
   Mpi::Init();

   int dim = 2;
   int polynomial_order = 2;
   int num_elements = 4;

   Mesh mesh_serial = Mesh::MakeCartesian2D(num_elements, num_elements,
                                            Element::Type::QUADRILATERAL);
   // Mesh mesh_serial("../data/star.mesh");
   ParMesh mesh(MPI_COMM_WORLD, mesh_serial);
   mesh.SetCurvature(1);

   mesh_serial.Clear();

   H1_FECollection h1fec(polynomial_order, dim);
   ParFiniteElementSpace h1fes(&mesh, &h1fec);

   Array<int> ess_tdof_list;
   if (mesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      h1fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   const IntegrationRule &ir =
      IntRules.Get(h1fes.GetFE(0)->GetGeomType(), 2 * h1fec.GetOrder() + 1);

   std::cout << "nqpts = " << ir.GetNPoints() << std::endl;

   ParGridFunction u(&h1fes);
   ParGridFunction rho(&h1fes);

   rho = 0.123;

   auto foo = [](tensor<double, 2> grad_u, tensor<double, 2, 2> J, double w)
   {
      return grad_u * inv(J) * transpose(inv(J)) * det(J) * w;
   };

   ElementOperator qf
   {
      foo,
      // inputs
      std::tuple{
         Gradient{"potential"},
         Gradient{"coordinates"},
         Weight{"integration_weight"}},
      // outputs
      std::tuple{Gradient{"potential"}}};

   ElementOperator forcing_qf
   {
      [](tensor<double, 2, 2> J, double w) { return -det(J) * w; },
      // inputs
      std::tuple{
         Gradient{"coordinates"},
         Weight{"integration_weight"}},
      // outputs
      std::tuple{
         Value{"potential"}}
   };

   DifferentiableForm dop(
      // Solutions
   {{&u, "potential"}},
   // Parameters
   {
      {mesh.GetNodes(), "coordinates"},
   },
   mesh);

   dop.AddElementOperator(qf, ir);
   dop.AddElementOperator(forcing_qf, ir, Independent{});
   dop.SetEssentialTrueDofs(ess_tdof_list);

   GMRESSolver gmres(MPI_COMM_WORLD);
   gmres.SetAbsTol(1e-12);
   gmres.SetRelTol(1e-12);
   gmres.SetMaxIter(500);
   // gmres.SetPrintLevel(IterativeSolver::PrintLevel().All());

   NewtonSolver newton(MPI_COMM_WORLD);
   newton.SetSolver(gmres);
   newton.SetOperator(dop);
   newton.SetRelTol(1e-8);
   newton.SetMaxIter(10);
   newton.SetPrintLevel(1);

   std::cout << std::setprecision(9);

   Vector &x = u.GetTrueVector();
   Vector y(x.Size());
   x = 1.0;
   x.SetSubVector(ess_tdof_list, 0.0);

   // x = 1.0;
   // x.SetSubVector(ess_tdof_list, 0.0);
   // dop.Mult(x, y);
   // y.Print(std::cout, y.Size());
   // x = 0.85;
   // x.SetSubVector(ess_tdof_list, 0.0);
   // dop.Mult(x, y);
   // y.Print(std::cout, y.Size());

   // Vector dx(x.Size());
   // dx = 1.0;
   // dx.SetSubVector(ess_tdof_list, 0.0);
   // auto &dop_jvp = dop.GetGradient(x);
   // dop_jvp.Mult(dx, y);
   // y.Print(std::cout, y.Size());

   Vector zero;
   newton.Mult(zero, x);

   x.Print(std::cout, x.Size());

   u.Distribute(x);

   if (true)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << u << std::flush;
   }

   return 0;
}
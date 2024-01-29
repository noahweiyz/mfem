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

void print_matrix(DenseMatrix m)
{
   out << "{";
   for (int i = 0; i < m.NumRows(); i++)
   {
      out << "{";
      for (int j = 0; j < m.NumCols(); j++)
      {
         out << m(i, j);
         if (j < m.NumCols() - 1)
         {
            out << ", ";
         }
      }
      if (i < m.NumRows() - 1)
      {
         out << "}, ";
      }
      else
      {
         out << "}";
      }
   }
   out << "} ";
}

void print_vector(Vector v)
{
   out << "{";
   for (int i = 0; i < v.Size(); i++)
   {
      out << v(i);
      if (i < v.Size() - 1)
      {
         out << ", ";
      }
   }
   out << "}";
}

using mfem::internal::tensor;

int enzyme_dup;
int enzyme_dupnoneed;
int enzyme_out;
int enzyme_const;

template <typename return_type, typename... T>
return_type __enzyme_fwddiff(void *, T...);

template <typename return_type, typename... T>
return_type __enzyme_autodiff(void *, T...);

template <typename T>
struct always_false
{
   static constexpr bool value = false;
};

template<class>
inline constexpr bool always_false_v = false;

struct Independent {};
struct Dependent {};

struct DofToQuadTensors
{
   DeviceTensor<2, const double> B;
   DeviceTensor<3, const double> G;
};

struct Field
{
   std::variant<
   const QuadratureFunction *,
         const GridFunction *,
         const ParGridFunction *
         > data;

   std::string label;
};

const Vector &GetFieldData(Field &f)
{
   return *std::visit([](auto&& f) -> const Vector*
   {
      return static_cast<const Vector *>(f);
   }, f.data);
}

const Operator *GetProlongationFromField(const Field &f)
{
   return std::visit([](auto&& arg) -> const Operator*
   {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, const GridFunction *>)
      {
         return arg->FESpace()->GetProlongationMatrix();
      }
      else if constexpr (std::is_same_v<T, const ParGridFunction *>)
      {
         return arg->ParFESpace()->GetProlongationMatrix();
      }
      else if constexpr (std::is_same_v<T, const QuadratureFunction *>)
      {
         return nullptr;
      }
      else
      {
         static_assert(always_false_v<T>, "can't use GetProlongation on type");
      }
   }, f.data);
}

const Operator *GetElementRestriction(const Field &f, ElementDofOrdering o)
{
   return std::visit([&o](auto&& arg) -> const Operator*
   {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, const GridFunction *>)
      {
         return arg->FESpace()->GetElementRestriction(o);
      }
      else if constexpr (std::is_same_v<T, const ParGridFunction *>)
      {
         return arg->ParFESpace()->GetElementRestriction(o);
      }
      else if constexpr (std::is_same_v<T, const QuadratureFunction *>)
      {
         return nullptr;
      }
      else
      {
         static_assert(always_false_v<T>, "can't use GetElementRestriction on type");
      }
   }, f.data);
}

const DofToQuad *GetDofToQuad(const Field &f, const IntegrationRule &ir,
                              DofToQuad::Mode mode)
{
   return std::visit([&ir, &mode](auto&& arg) -> const DofToQuad*
   {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, const GridFunction *>)
      {
         return &arg->FESpace()->GetFE(0)->GetDofToQuad(ir, mode);
      }
      else if constexpr (std::is_same_v<T, const ParGridFunction *>)
      {
         return &arg->FESpace()->GetFE(0)->GetDofToQuad(ir, mode);
      }
      else if constexpr (std::is_same_v<T, const QuadratureFunction *>)
      {
         return nullptr;
      }
      else
      {
         static_assert(always_false_v<T>, "can't use GetDofToQuad on type");
      }
   }, f.data);
}

int GetVSize(const Field &f)
{
   return std::visit([](auto && arg)
   {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, const GridFunction *>)
      {
         return arg->FESpace()->GetVSize();
      }
      else if constexpr (std::is_same_v<T, const ParGridFunction *>)
      {
         return arg->ParFESpace()->GetVSize();
      }
      else if constexpr (std::is_same_v<T, const QuadratureFunction *>)
      {
         return arg->Size();
      }
      else
      {
         static_assert(always_false_v<T>, "can't use GetVSize on type");
      }
   }, f.data);
}

int GetTrueVSize(const Field &f)
{
   return std::visit([](auto && arg)
   {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, const GridFunction *>)
      {
         return arg->FESpace()->GetTrueVSize();
      }
      else if constexpr (std::is_same_v<T, const ParGridFunction *>)
      {
         return arg->ParFESpace()->GetTrueVSize();
      }
      else if constexpr (std::is_same_v<T, const QuadratureFunction *>)
      {
         return arg->Size();
      }
      else
      {
         static_assert(always_false_v<T>, "can't use GetTrueVSize on type");
      }
   }, f.data);
}

int GetVDim(const Field &f)
{
   return std::visit([](auto && arg)
   {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, const GridFunction *>)
      {
         return arg->FESpace()->GetVDim();
      }
      else if constexpr (std::is_same_v<T, const ParGridFunction *>)
      {
         return arg->ParFESpace()->GetVDim();
      }
      else if constexpr (std::is_same_v<T, const QuadratureFunction *>)
      {
         return arg->GetVDim();
      }
      else
      {
         static_assert(always_false_v<T>, "can't use GetVDim on type");
      }
   }, f.data);
}

int GetDimension(const Field &f)
{
   return std::visit([](auto && arg)
   {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, const GridFunction *>)
      {
         return arg->FESpace()->GetMesh()->Dimension();
      }
      else if constexpr (std::is_same_v<T, const ParGridFunction *>)
      {
         return arg->ParFESpace()->GetMesh()->Dimension();
      }
      else if constexpr (std::is_same_v<T, const QuadratureFunction *>)
      {
         return 1;
      }
      else
      {
         static_assert(always_false_v<T>, "can't use GetDimension on type");
      }
   }, f.data);
}

class FieldDescriptor
{
public:
   FieldDescriptor(std::string name) : name(name) {};

   std::string name;

   int size_on_qp = -1;

   int dim = -1;

   int vdim = -1;
};

class None : public FieldDescriptor
{
public:
   None(std::string name) : FieldDescriptor(name) {}
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
   else if constexpr (std::is_same_v<field_descriptor_type, None>)
   {
      return GetVDim(f);
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

void prolongation(std::vector<Field> fields, const Vector &x,
                  std::vector<Vector> &fields_local)
{
   int offset = 0;
   for (int i = 0; i < fields.size(); i++)
   {
      const auto P = GetProlongationFromField(fields[i]);
      if (P != nullptr)
      {
         const int width = P->Width();
         const Vector x_i(x.GetData() + offset, width);
         fields_local[i].SetSize(P->Height());

         P->Mult(x_i, fields_local[i]);
         offset += width;
      }
      else
      {
         const int width = GetTrueVSize(fields[i]);
         fields_local[i].SetSize(width);
         const Vector x_i(x.GetData() + offset, width);
         fields_local[i] = x_i;
         offset += width;
      }
   }
}

void element_restriction(std::vector<Field> fields,
                         const std::vector<Vector> &fields_local,
                         std::vector<Vector> &fields_e)
{
   for (int i = 0; i < fields.size(); i++)
   {
      const auto R = GetElementRestriction(fields[i], ElementDofOrdering::NATIVE);
      if (R != nullptr)
      {
         const int height = R->Height();
         fields_e[i].SetSize(height);
         R->Mult(fields_local[i], fields_e[i]);
      }
      else
      {
         const int height = GetTrueVSize(fields[i]);
         fields_e[i].SetSize(height);
         fields_e[i] = fields_local[i];
      }
   }
}

void element_restriction(std::vector<Field> fields,
                         std::vector<Vector> &fields_e,
                         int field_offset)
{
   for (int i = 0; i < fields.size(); i++)
   {
      const auto R = GetElementRestriction(fields[i], ElementDofOrdering::NATIVE);
      if (R != nullptr)
      {
         const int height = R->Height();
         fields_e[i + field_offset].SetSize(height);
         R->Mult(GetFieldData(fields[i]), fields_e[i + field_offset]);
      }
      else
      {
         fields_e[i + field_offset] = GetFieldData(fields[i]);
      }
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

template <typename T1, typename T2>
void allocate_qf_arg(const T1&, T2&)
{
   static_assert(always_false<T1>::value,
                 "allocate_qf_arg not implemented for requested type combination");
}

void allocate_qf_arg(const None &, double &)
{
   // no op
}

void allocate_qf_arg(const None &, tensor<double, 2, 2> &)
{
   // no op
}

template <int length>
void allocate_qf_arg(const None &, tensor<double, length> &)
{
   // no op
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

template <int length>
void allocate_qf_arg(const Value &, tensor<double, length> &)
{
   // no op
}

void allocate_qf_arg(const Gradient &input, Vector &arg)
{
   arg.SetSize(input.size_on_qp / input.vdim);
}

void allocate_qf_arg(const Gradient &input, DenseMatrix &arg)
{
   arg.SetSize(input.size_on_qp / input.vdim);
}

template <int length>
void allocate_qf_arg(const Gradient &, tensor<double, length> &)
{
   // no op
}

void allocate_qf_arg(const Gradient &, double &)
{
   // no op
}

void allocate_qf_arg(const Gradient &, tensor<double, 2, 2> &)
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

template <int length>
void prepare_qf_arg(const DeviceTensor<1> &u, tensor<double, length> &arg)
{
   for (int i = 0; i < u.GetShape()[0]; i++)
   {
      arg(i) = u(i);
   }
}

template <int dim, int vdim>
void prepare_qf_arg(const DeviceTensor<1> &u, tensor<double, dim, vdim> &arg)
{
   for (int i = 0; i < vdim; i++)
   {
      for (int j = 0; j < dim; j++)
      {
         arg(j, i) = u((i * vdim) + j);
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

// Vector prepare_qf_result(tensor<double, 1> x)
// {
//    Vector r(1);
//    r(0) = x(0);
//    return r;
// }

template <int length>
Vector prepare_qf_result(tensor<double, length> x)
{
   Vector r(length);
   for (size_t i = 0; i < length; i++)
   {
      r(i) = x(i);
   }
   return r;
}

Vector prepare_qf_result(tensor<double, 2, 2> x)
{
   Vector r(4);
   for (size_t i = 0; i < 2; i++)
   {
      for (size_t j = 0; j < 2; j++)
      {
         // TODO: Careful with the indices here!
         r(j + (i * 2)) = x(j, i);
      }
   }
   return r;
}

template <typename T>
Vector prepare_qf_result(T)
{
   static_assert(always_false<T>::value,
                 "prepare_qf_result not implemented for result type");
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
   // TODO: Create separate function for clarity
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
   else if constexpr (std::is_same_v<typename std::remove_reference<
                      decltype(input)>::type,
                      None>)
   {
      const auto B(dtqmaps.B);
      auto [num_qp, num_dof] = B.GetShape();
      const int size_on_qp = input.size_on_qp;
      const int element_offset = element_idx * size_on_qp * num_qp;
      const auto field = Reshape(field_e.Read() + element_offset,
                                 size_on_qp * num_qp);
      auto f = Reshape(&field_qp[0], size_on_qp * num_qp);
      for (int i = 0; i < size_on_qp * num_qp; i++)
      {
         f(i) = field(i);
      }
   }
}

template <int num_qfinputs, typename input_type, std::size_t... i>
void map_fields_to_quadrature_data(
   std::vector<DeviceTensor<2>> &fields_qp, int element_idx,
   const std::vector<Vector> &fields_e,
   std::array<int, num_qfinputs> qfinput_to_field,
   std::vector<DofToQuadTensors> &dtqmaps,
   DeviceTensor<1, const double> integration_weights,
   input_type &qfinputs,
   std::index_sequence<i...>)
{
   (map_field_to_quadrature_data(fields_qp[i], element_idx,
                                 dtqmaps[qfinput_to_field[i]], fields_e[qfinput_to_field[i]],
                                 std::get<i>(qfinputs), integration_weights),
    ...);
}

template <typename input_type>
void map_field_to_quadrature_data_conditional(
   DeviceTensor<2> field_qp, int element_idx, DofToQuadTensors &dtqmaps,
   const Vector &field_e, input_type &input,
   DeviceTensor<1, const double> integration_weights,
   const bool condition)
{
   if (condition)
   {
      map_field_to_quadrature_data(field_qp, element_idx, dtqmaps, field_e, input,
                                   integration_weights);
   }
}

template <int num_qfinputs, typename input_type, std::size_t... i>
void map_fields_to_quadrature_data_conditional(
   std::vector<DeviceTensor<2>> &fields_qp, int element_idx,
   const std::vector<Vector> &fields_e,
   const int field_idx,
   std::vector<DofToQuadTensors> &dtqmaps,
   DeviceTensor<1, const double> integration_weights,
   std::array<bool, num_qfinputs> conditions,
   input_type &qfinputs,
   std::index_sequence<i...>)
{
   (map_field_to_quadrature_data_conditional(fields_qp[i], element_idx,
                                             dtqmaps[field_idx], fields_e[field_idx],
                                             std::get<i>(qfinputs), integration_weights, conditions[i]),
    ...);
}

template <typename output_type>
void map_quadrature_data_to_fields(Vector &y_e, int element_idx, int num_el,
                                   DeviceTensor<3, double> residual_qp,
                                   output_type output,
                                   std::vector<DofToQuadTensors> &dtqmaps,
                                   int test_space_field_idx)
{
   // assuming the quadrature point residual has to "play nice with
   // the test function"
   if constexpr (std::is_same_v<
                 typename std::remove_reference<decltype(output)>::type,
                 Value>)
   {
      const auto B(dtqmaps[test_space_field_idx].B);
      const auto [num_qp, num_dof] = B.GetShape();
      const int vdim = output.vdim;
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
                      decltype(output)>::type, Gradient>)
   {
      const auto G(dtqmaps[test_space_field_idx].G);
      const auto [num_qp, dim, num_dof] = G.GetShape();
      const int vdim = output.vdim;
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
   else if constexpr (std::is_same_v<typename std::remove_reference<
                      decltype(output)>::type, None>)
   {
      const auto B(dtqmaps[test_space_field_idx].B);
      const auto [num_qp, num_dof] = B.GetShape();
      const int size_on_qp = output.vdim;
      auto C = Reshape(&residual_qp(0, 0, element_idx), size_on_qp * num_qp);
      auto y = Reshape(y_e.ReadWrite(), size_on_qp * num_qp, num_el);
      for (int i = 0; i < size_on_qp * num_qp; i++)
      {
         y(i, element_idx) += C(i);
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
      return field.label == input_name;
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
void map_qfinput_to_field(std::vector<Field> &fields,
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
         input.dim = GetDimension(fields[idx]);
         input.vdim = GetVDim(fields[idx]);
         input.size_on_qp = GetSizeOnQP(input, fields[idx]);
         map = idx;
      }
      else
      {
         MFEM_ABORT("can't find field for " << input.name);
      }
   };

   (f(std::get<i>(inputs), map[i]), ...);
}

std::vector<DofToQuadTensors> map_dtqmaps(std::vector<const DofToQuad*>
                                          dtqmaps, int dim)
{
   std::vector<DofToQuadTensors> dtqmaps_tensor;
   for (const auto &map : dtqmaps)
   {
      if (map != nullptr)
      {
         dtqmaps_tensor.push_back(
         {
            Reshape(map->B.Read(), map->nqpt, map->ndof),
            Reshape(map->G.Read(), map->nqpt, dim, map->ndof)
         });
      }
      else
      {
         DeviceTensor<2, const double> B(nullptr, map->nqpt, map->ndof);
         DeviceTensor<3, const double> G(nullptr, map->nqpt, dim, map->ndof);
         dtqmaps_tensor.push_back({B, G});
      }
   }
   return dtqmaps_tensor;
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
      if (solutions.size() > 1)
      {
         MFEM_ABORT("only one trial space allowed at the moment, tried with " <<
                    solutions.size());
      }
      fields.insert(fields.end(), solutions.begin(), solutions.end());
      fields.insert(fields.end(), parameters.begin(), parameters.end());

      solutions_local.resize(solutions.size());
      parameters_local.resize(parameters.size());
      directions_local.resize(solutions.size());
      fields_e.resize(solutions.size() + parameters.size());
      directions_e.resize(solutions.size());
   }

   // template <
   //    typename qf_type,
   //    typename input_type,
   //    typename output_type>
   // void AddElementOperator(ElementOperator<qf_type, input_type, output_type> &qf,
   //                         const IntegrationRule &ir)
   // {
   //    AddElementOperator(qf, ir, Dependent{});
   // }

   template <
      typename qf_type,
      typename input_type,
      typename output_type>
   void AddElementOperator(ElementOperator<qf_type, input_type, output_type> &qf,
                           const IntegrationRule &ir)
   {
      constexpr ElementDofOrdering element_dof_ordering = ElementDofOrdering::NATIVE;
      constexpr size_t num_qfinputs = std::tuple_size_v<input_type>;
      constexpr size_t num_qfoutputs = std::tuple_size_v<output_type>;

      if (num_qfoutputs != 1)
      {
         MFEM_ABORT("only one test space allowed at the moment");
      }

      // TODO: This is currently a hack that fixes the first solution
      // variable to be the dependent variable in the JvP.
      //
      // This index is the field that the residual is differentiated wrt.
      constexpr int primary_variable_idx = 0;

      bool contains_dependent_input = std::apply([&](auto ...input)
      {
         auto check_dependency = [&](auto &input)
         {
            return solutions.size() &&
                   (input.name == solutions[primary_variable_idx].label);
         };
         return (check_dependency(input) || ...);
      },
      qf.inputs);

      const int num_el = mesh.GetNE();
      const int num_qp = ir.GetNPoints();

      std::cout << "adding element operator with " << num_qfinputs << " inputs and "
                << num_qfoutputs << " outputs" << "\n";

      int test_space_field_idx;
      int residual_size_on_qp;

      auto output_fd = std::get<0>(qf.outputs);
      if ((test_space_field_idx = find_name_idx(fields, output_fd.name)) != -1)
      {
         residual_size_on_qp =
            GetSizeOnQP(output_fd, fields[test_space_field_idx]);
         output_fd.dim = GetDimension(fields[test_space_field_idx]);
         output_fd.vdim = GetVDim(fields[test_space_field_idx]);
         output_fd.size_on_qp = residual_size_on_qp;
      }
      else
      {
         MFEM_ABORT("can't figure out residual size on quadrature point level");
      }

      std::array<int, num_qfinputs> qfinput_to_field;
      map_qfinput_to_field(fields, qfinput_to_field, qf.inputs,
                           std::make_index_sequence<num_qfinputs> {});

      // All solutions T-vector sizes make up the height of the operator, since
      // they are explicitly provided in Mult() for example.
      this->height = 0;
      int residual_local_size = 0;
      for (auto &f : solutions)
      {
         this->height += GetTrueVSize(f);
         residual_local_size += GetVSize(f);
      }
      residual_local.SetSize(residual_local_size);

      // TODO: Only works for one test space
      if constexpr (std::is_same_v<typename
                    std::remove_reference<decltype(output_fd)>, None>)
      {
         this->width = residual_size_on_qp * num_qp * num_el;
      }
      else
      {
         this->width = this->height;
      }

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
      residual_e.SetSize(residual_qp_mem.Size());

      using qf_args = typename create_function_signature<
                      decltype(&qf_type::operator())>::type::parameter_types;

      // This tuple contains objects of every ElementOperator::func function
      // parameter which might have to be resized.
      qf_args args{};
      allocate_qf_args(args, qf.inputs);

      Array<double> integration_weights_mem = ir.GetWeights();

      // Duplicate B/G and assume only a single element type for now
      std::vector<const DofToQuad*> dtqmaps;
      for (const auto &field : fields)
      {
         dtqmaps.emplace_back(GetDofToQuad(field, ir, DofToQuad::FULL));
      }

      residual_integrators.emplace_back(
         [&, args, qfinput_to_field, fields_qp_mem, residual_qp_mem,
             residual_size_on_qp, test_space_field_idx, output_fd, dtqmaps,
             integration_weights_mem, num_qp, num_el](Vector &y_e) mutable
      {
         auto dtqmaps_tensor = map_dtqmaps(dtqmaps, dim);

         const auto residual_qp = Reshape(residual_qp_mem.ReadWrite(),
                                          residual_size_on_qp, num_qp, num_el);

         // Fields interpolated to the quadrature points in the order of
         // quadrature function arguments
         std::vector<DeviceTensor<2>> fields_qp;
         map_inputs_to_memory(
         fields_qp, fields_qp_mem, num_qp,qf.inputs, std::make_index_sequence<num_qfinputs>{});

         DeviceTensor<1, const double> integration_weights(
            integration_weights_mem.Read(), num_qp);

         for (int el = 0; el < num_el; el++)
         {
            // B
            // prepare fields on quadrature points
            map_fields_to_quadrature_data<num_qfinputs>(
               fields_qp, el, fields_e,
               qfinput_to_field, dtqmaps_tensor,
            integration_weights, qf.inputs, std::make_index_sequence<num_qfinputs> {});

            for (int qp = 0; qp < num_qp; qp++)
            {
               auto f_qp = apply_qf(qf.func, args, fields_qp, qp);

               auto r_qp = Reshape(&residual_qp(0, qp, el), residual_size_on_qp);
               for (int i = 0; i < residual_size_on_qp; i++)
               {
                  r_qp(i) = f_qp(i);
               }
            }

            // B^T
            map_quadrature_data_to_fields(y_e, el, num_el, residual_qp,
                                          output_fd, dtqmaps_tensor,
                                          test_space_field_idx);

         }
      });

      auto R = GetElementRestriction(fields[test_space_field_idx],
                                     element_dof_ordering);
      if (R == nullptr)
      {
         out << "G^T = Identity" << "\n";
         element_restriction_transpose = [](Vector &r_e, Vector &y)
         {
            y = r_e;
         };
      }
      else
      {
         element_restriction_transpose = [R](Vector &r_e, Vector &y)
         {
            R->MultTranspose(r_e, y);
         };
      }

      auto P = GetProlongationFromField(fields[test_space_field_idx]);
      if (P == nullptr)
      {
         out << "P^T = Identity" << "\n";
         prolongation_transpose = [](Vector &r_local, Vector &y)
         {
            y = r_local;
         };
      }
      else
      {
         prolongation_transpose = [P](Vector &r_local, Vector &y)
         {
            P->MultTranspose(r_local, y);
         };
      }

      if (contains_dependent_input)
      {
         // Check which qf inputs are dependent on the primary variable
         std::array<bool, num_qfinputs> qfinput_is_dependent;
         for (int i = 0; i < qfinput_is_dependent.size(); i++)
         {
            if (qfinput_to_field[i] == primary_variable_idx)
            {
               qfinput_is_dependent[i] = true;
               out << "function input " << i << " is dependent on variable "
                   << fields[qfinput_to_field[i]].label << "\n";
            }
            else
            {
               qfinput_is_dependent[i] = false;
            }
         }

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
            [&, args, shadow_args, qfinput_to_field,
                qfinput_is_dependent,
                fields_qp_mem,
                directions_qp_mem, residual_qp_mem,
                residual_size_on_qp,
                test_space_field_idx, output_fd, dtqmaps,
                integration_weights_mem, num_qp,
                num_el](Vector &y_e) mutable
         {
            auto dtqmaps_tensor = map_dtqmaps(dtqmaps, dim);

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
                  fields_qp, el, fields_e, qfinput_to_field, dtqmaps_tensor,
                  integration_weights, qf.inputs,
               std::make_index_sequence<num_qfinputs> {});

               // prepare directions (shadow memory)
               map_fields_to_quadrature_data_conditional<num_qfinputs>(
                  directions_qp, el,
                  directions_e, primary_variable_idx,
                  dtqmaps_tensor,
                  integration_weights,
                  qfinput_is_dependent,
                  qf.inputs,
                  std::make_index_sequence<num_qfinputs> {});

               // D -> D
               for (int qp = 0; qp < num_qp; qp++)
               {
                  auto f_qp = apply_qf_fwddiff(qf.func, args, fields_qp, shadow_args,
                                               directions_qp, qp);

                  auto r_qp = Reshape(&residual_qp(0, qp, el), residual_size_on_qp);
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
      MFEM_ASSERT(residual_integrators.size(), "form does not contain any operators");

      // P
      prolongation(solutions, x, solutions_local);

      // G
      element_restriction(solutions, solutions_local, fields_e);
      element_restriction(parameters, fields_e, solutions.size());

      // BEGIN GPU
      // B^T Q B x
      residual_e = 0.0;
      for (int i = 0; i < residual_integrators.size(); i++)
      {
         residual_integrators[i](residual_e);
      }
      // END GPU

      // G^T
      element_restriction_transpose(residual_e, residual_local);

      // P^T
      prolongation_transpose(residual_local, y);

      y.SetSubVector(ess_tdof_list, 0.0);
   }

   void SetEssentialTrueDofs(const Array<int> &l) { l.Copy(ess_tdof_list); }

   void JacobianMult(const Vector &x, Vector &y) const
   {
      // apply essential bcs
      current_direction_t = x;
      current_direction_t.SetSubVector(ess_tdof_list, 0.0);

      prolongation(solutions, current_direction_t, directions_local);
      prolongation(solutions, current_state_t, solutions_local);

      element_restriction(solutions, directions_local, directions_e);
      element_restriction(solutions, solutions_local, fields_e);
      element_restriction(parameters, fields_e, solutions.size());

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
      element_restriction_transpose(jvp_e, residual_local);

      // P^T
      prolongation_transpose(residual_local, y);

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
   std::function<void(Vector &, Vector &)> prolongation_transpose;

   mutable std::vector<Vector> solutions_local;
   mutable std::vector<Vector> parameters_local;
   mutable std::vector<Vector> directions_local;
   mutable Vector residual_local;

   mutable std::vector<Vector> fields_e;
   mutable std::vector<Vector> directions_e;
   mutable Vector residual_e;

   std::shared_ptr<JacobianOperator> jacobian_op;
   mutable Vector current_direction_t, current_state_t;

   Array<int> ess_tdof_list;
};

int test_interpolate_linear_scalar()
{
   constexpr int num_elements = 1;
   Mesh mesh_serial = Mesh::MakeCartesian2D(num_elements, num_elements,
                                            Element::QUADRILATERAL);
   ParMesh mesh(MPI_COMM_WORLD, mesh_serial);

   mesh.SetCurvature(1);
   const int dim = mesh.Dimension();
   mesh_serial.Clear();

   int polynomial_order = 1;
   H1_FECollection h1fec(polynomial_order, dim);
   ParFiniteElementSpace h1fes(&mesh, &h1fec);

   const IntegrationRule &ir =
      IntRules.Get(h1fes.GetFE(0)->GetGeomType(), 2 * h1fec.GetOrder() + 1);

   ParGridFunction f1_g(&h1fes);

   auto mass_qf = [](double u, tensor<double, 2, 2> J, double w)
   {
      out << u << " ";
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
      std::tuple{Value{"primary_variable"}}};

   DifferentiableForm dop(
   {{&f1_g, "primary_variable"}},
   {{mesh.GetNodes(), "coordinates"}},
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

   Vector x(f1_g), y(f1_g.Size());
   dop.Mult(x, y);

   out << "\n";
   for (int e = 0; e < num_elements; e++)
   {
      ElementTransformation *T = mesh.GetElementTransformation(e);
      for (int qp = 0; qp < ir.GetNPoints(); qp++)
      {
         const IntegrationPoint &ip = ir.IntPoint(qp);
         T->SetIntPoint(&ip);

         double f = f1_c.Eval(*T, ip);
         out << f << " ";
      }
   }

   return 0;
}

int test_interpolate_gradient_scalar(const int polynomial_order)
{
   constexpr int num_elements = 1;
   // Mesh mesh_serial = Mesh::MakeCartesian2D(num_elements, num_elements,
   //                                          Element::QUADRILATERAL);
   Mesh mesh_serial = Mesh("../data/skewed-square.mesh");
   ParMesh mesh(MPI_COMM_WORLD, mesh_serial);
   mesh.SetCurvature(1);
   const int dim = mesh.Dimension();
   mesh_serial.Clear();

   H1_FECollection h1fec(polynomial_order, dim);
   ParFiniteElementSpace h1fes(&mesh, &h1fec);

   const IntegrationRule &ir =
      IntRules.Get(h1fes.GetFE(0)->GetGeomType(), 2 * h1fec.GetOrder() + 1);

   ParGridFunction f1_g(&h1fes);

   auto mass_qf = [](double u, tensor<double, 2> grad_u, tensor<double, 2, 2> J,
                     double w)
   {
      out << grad_u * transpose(inv(J)) << " ";
      return 0.0;
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
      std::tuple{Value{"primary_variable"}}};

   DifferentiableForm dop(
   {{&f1_g, "primary_variable"}},
   {{mesh.GetNodes(), "coordinates"}},
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

   Vector x(f1_g), y(f1_g.Size());
   dop.Mult(x, y);

   out << "\n";
   for (int e = 0; e < num_elements; e++)
   {
      ElementTransformation *T = mesh.GetElementTransformation(e);
      for (int qp = 0; qp < ir.GetNPoints(); qp++)
      {
         const IntegrationPoint &ip = ir.IntPoint(qp);
         T->SetIntPoint(&ip);

         Vector g(dim);
         f1_g.GetGradient(*T, g);
         out << "{";
         for (int i = 0; i < dim; i++)
         {
            out << g(i);
            if (i < dim - 1)
            {
               out << ", ";
            }
            else
            {
               out << "} ";
            }
         }

         // out << f1_g.GetValue(*T) << " ";
      }
   }

   out << "\n";

   return 0;
}

int test_interpolate_linear_vector(const int refinements, int polynomial_order)
{
   constexpr int vdim = 2;
   Mesh mesh_serial = Mesh::MakeCartesian2D(1, 1,
                                            Element::QUADRILATERAL);
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

   ParGridFunction f1_g(&h1fes);

   auto mass_qf = [](tensor<double, vdim> u, tensor<double, 2, 2> J, double w)
   {
      out << u << " ";
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
      std::tuple{Value{"primary_variable"}}};

   DifferentiableForm dop(
   {{&f1_g, "primary_variable"}},
   {{mesh.GetNodes(), "coordinates"}},
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

   Vector x(f1_g), y(f1_g.Size());
   dop.Mult(x, y);

   out << "\n";
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      ElementTransformation *T = mesh.GetElementTransformation(e);
      for (int qp = 0; qp < ir.GetNPoints(); qp++)
      {
         const IntegrationPoint &ip = ir.IntPoint(qp);
         T->SetIntPoint(&ip);

         Vector f(vdim);
         f1_c.Eval(f, *T, ip);
         out << "{";
         for (int i = 0; i < vdim; i++)
         {
            out << f(i);
            if (i < vdim - 1)
            {
               out << ", ";
            }
            else
            {
               out << "} ";
            }
         }

      }
      out << "\n";
   }

   return 0;
}

int test_interpolate_gradient_vector(const std::string mesh_file,
                                     const int refinements,
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

   ParGridFunction f1_g(&h1fes);

   auto mass_qf = [](tensor<double, vdim, 2> grad_u, tensor<double, 2, 2> J,
                     double w)
   {
      // out << grad_u * transpose(inv(J)) << " ";
      return grad_u;
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
      std::tuple{Value{"primary_variable"}}};

   DifferentiableForm dop(
   {{&f1_g, "primary_variable"}},
   {{mesh.GetNodes(), "coordinates"}},
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

   Vector x(f1_g), y(f1_g.Size());
   dop.Mult(x, y);

   out << "\n";
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      ElementTransformation *T = mesh.GetElementTransformation(e);
      for (int qp = 0; qp < ir.GetNPoints(); qp++)
      {
         const IntegrationPoint &ip = ir.IntPoint(qp);
         T->SetIntPoint(&ip);

         DenseMatrix g(vdim, dim);
         f1_g.GetVectorGradient(*T, g);
         g.Symmetrize();
         DenseMatrix J(dim, dim);
         print_matrix(T->Jacobian());
      }
      out << "\n";
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

   // int ret;
   // ret = test_interpolate_linear_scalar(polynomial_order);
   // ret = test_interpolate_gradient_scalar(polynomial_order);
   // ret = test_interpolate_linear_vector(refinements, polynomial_order);
   // ret = test_interpolate_gradient_vector(mesh_file, refinements,
   //                                        polynomial_order);

   // exit(0);

   // Mesh mesh_serial = Mesh::MakeCartesian2D(num_elements, num_elements,
   //                                          Element::Type::QUADRILATERAL);
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
   };

   VectorFunctionCoefficient exact_solution_coeff(dim, exact_solution);

   auto linear_elastic = [](tensor<double, 2, 2> dudxi,
                            tensor<double, 2, 2> J,
                            double w)
   {
      using mfem::internal::tensor;
      using mfem::internal::IsotropicIdentity;

      double lambda, mu;
      {
         lambda = 1.0;
         mu = 1.0;
      }
      static constexpr auto I = IsotropicIdentity<2>();
      auto eps = sym(dudxi * inv(J));
      auto JxW = transpose(inv(J)) * det(J) * w;
      return (lambda * tr(eps) * I + 2.0 * mu * eps) * JxW;
   };

   ElementOperator qf
   {
      // quadrature function lambda
      linear_elastic,
      // inputs
      std::tuple{
         Gradient{"displacement"},
         Gradient{"coordinates"},
         Weight{"integration_weight"}},
      // outputs (return values)
      std::tuple{
         Gradient{"displacement"}
      }
   };

   ElementOperator forcing_qf
   {
      [](tensor<double, 2> x, tensor<double, 2, 2> J, double w)
      {
         double lambda, mu;
         {
            lambda = 1.0;
            mu = 1.0;
         }
         auto f = x;
         f(0) = 4.0*mu + 2.0*lambda;
         f(1) = 2.0*mu + lambda;
         return f * det(J) * w;
      },
      // inputs
      std::tuple{
         Value{"coordinates"},
         Gradient{"coordinates"},
         Weight{"integration_weight"}},
      // outputs
      std::tuple{
         Value{"displacement"}}
   };

   DifferentiableForm dop(
      // Solutions
   {{&u, "displacement"}},
   // Parameters
   {
      {mesh.GetNodes(), "coordinates"},
   },
   mesh);

   // auto exact_solution = [](const Vector &coords, Vector &u)
   // {
   //    const double x = coords(0);
   //    const double y = coords(1);
   //    u(0) = cos(y)+sin(x);
   //    u(1) = sin(x)-cos(y);
   // };

   // VectorFunctionCoefficient exact_solution_coeff(vdim, exact_solution);

   // auto scalar_diffusion = [](tensor<double, 2, 2> grad_u,
   //                            tensor<double, 2, 2> J, double w)
   // {
   //    auto r = grad_u * inv(J) * transpose(inv(J)) * det(J) * w;
   //    return r;
   // };

   // ElementOperator qf
   // {
   //    scalar_diffusion,
   //    // inputs
   //    std::tuple{
   //       Gradient{"potential"},
   //       Gradient{"coordinates"},
   //       Weight{"integration_weight"}},
   //    // outputs
   //    std::tuple{
   //       Gradient{"potential"}
   //    }
   // };

   // ElementOperator forcing_qf
   // {
   //    [](tensor<double, 2> coords, tensor<double, 2, 2> J, double w)
   //    {
   //       const double x = coords(0);
   //       const double y = coords(1);
   //       const double f0 = cos(y)+sin(x);
   //       const double f1 = sin(x)-cos(y);
   //       return tensor<double, 2> {-f0, -f1} * det(J) * w;
   //    },
   //    // inputs
   //    std::tuple{
   //       Value{"coordinates"},
   //       Gradient{"coordinates"},
   //       Weight{"integration_weight"}},
   //    // outputs
   //    std::tuple{
   //       Value{"potential"}}
   // };

   // // auto mass_qf = [](tensor<double, 2> u, tensor<double, 2, 2> grad_u,
   // //                   tensor<double, 2, 2> J, double w)
   // // {
   // //    return u * det(J) * w;
   // // };

   // // ElementOperator mass
   // // {
   // //    mass_qf,
   // //    // inputs
   // //    std::tuple{
   // //       Value{"potential"},
   // //       Gradient{"potential"},
   // //       Gradient{"coordinates"},
   // //       Weight{"integration_weight"}},
   // //    // outputs
   // //    std::tuple{Value{"potential"}}};

   // DifferentiableForm dop(
   //    // Solutions
   // {
   //    {&u, "potential"},
   // },
   // // Parameters
   // {
   //    {mesh.GetNodes(), "coordinates"},
   // },
   // mesh);

   // // auto J = Reshape(mesh.GetGeometricFactors(ir,
   // //                                           GeometricFactors::JACOBIANS)->J.Read(), ir.GetNPoints(), dim, dim, 1);

   // // for (int i = 0; i < ir.GetNPoints(); i++)
   // // {
   // //    for (int c = 0; c < dim; c++)
   // //    {
   // //       for (int r = 0; r < dim; r++)
   // //       {
   // //          printf("%+.2f ", J(i, r, c, 0));
   // //       }
   // //       printf("\n");
   // //    }
   // //    printf("\n");
   // // }

   dop.AddElementOperator(qf, ir);
   dop.AddElementOperator(forcing_qf, ir);
   dop.SetEssentialTrueDofs(ess_tdof_list);

   // {
   //    DenseMatrix J(u.Size());
   //    u.ProjectCoefficient(exact_solution_coeff);
   //    auto &dop_grad = dop.GetGradient(u);

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

   GMRESSolver gmres(MPI_COMM_WORLD);
   gmres.SetRelTol(1e-12);
   gmres.SetMaxIter(5000);
   gmres.SetPrintLevel(IterativeSolver::PrintLevel().Summary());

   NewtonSolver newton(MPI_COMM_WORLD);
   newton.SetSolver(gmres);
   newton.SetOperator(dop);
   newton.SetRelTol(1e-12);
   newton.SetMaxIter(100);
   // newton.SetAdaptiveLinRtol();
   newton.SetPrintLevel(1);

   u = 1e-6;
   u.ProjectBdrCoefficient(exact_solution_coeff, ess_bdr);
   Vector x;
   u.GetTrueDofs(x);
   Vector y(x.Size());

   // dop.Mult(x, y);
   // exit(0);

   Vector zero;
   newton.Mult(zero, x);

   u.Distribute(x);

   std::cout << "|u-u_ex|_L2 = " << u.ComputeL2Error(exact_solution_coeff) << "\n";

   // if (true)
   // {
   //    char vishost[] = "localhost";
   //    int  visport   = 19916;
   //    {
   //       socketstream sol_sock(vishost, visport);
   //       sol_sock.precision(8);
   //       sol_sock << "solution\n" << mesh << u << std::flush;
   //    }
   //    {
   //       g.ProjectCoefficient(exact_solution_coeff);
   //       for (int i = 0; i < g.Size(); i++)
   //       {
   //          g(i) = abs(g(i) - u(i));
   //       }

   //       socketstream sol_sock(vishost, visport);
   //       sol_sock.precision(8);
   //       sol_sock << "solution\n" << mesh << g << std::flush;
   //    }
   // }

   return 0;
}

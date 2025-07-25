// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics

// expected-error@+1{{alignment attribute is not a power of 2}}
llvm.mlir.global private @invalid_global_alignment(42 : i64) {alignment = 63} : i64

// -----

llvm.func @ctor() {
  llvm.return
}

// expected-error@+1{{ctors, priorities, and data must have the same number of elements}}
llvm.mlir.global_ctors ctors = [@ctor], priorities = [], data = [#llvm.zero]

// -----

llvm.func @dtor() {
  llvm.return
}

// expected-error@+1{{dtors, priorities, and data must have the same number of elements}}
llvm.mlir.global_dtors dtors = [@dtor], priorities = [0 : i32, 32767 : i32], data = [#llvm.zero]

// -----

// expected-error@+1{{'ctor' does not reference a valid LLVM function}}
llvm.mlir.global_ctors ctors = [@ctor], priorities = [0 : i32], data = [#llvm.zero]

// -----

llvm.func @dtor()

// expected-error@+1{{'dtor' does not have a definition}}
llvm.mlir.global_dtors dtors = [@dtor], priorities = [0 : i32], data = [#llvm.zero]

// -----

llvm.func @dtor() {
  llvm.return
}

// expected-error@+1{{data element must be symbol or #llvm.zero}}
llvm.mlir.global_dtors dtors = [@dtor], priorities = [0 : i32], data = [0 : i32]

////////////////////////////////////////////////////////////////////////////////

// Check that parser errors are properly produced and do not crash the compiler.

// -----

func.func @icmp_non_string(%arg0 : i32, %arg1 : i16) {
  // expected-error@+1 {{invalid kind of attribute specified}}
  llvm.icmp 42 %arg0, %arg0 : i32
  return
}

// -----

func.func @icmp_wrong_string(%arg0 : i32, %arg1 : i16) {
  // expected-error@+1 {{'foo' is an incorrect value of the 'predicate' attribute}}
  llvm.icmp "foo" %arg0, %arg0 : i32
  return
}

// -----

func.func @alloca_missing_input_result_type(%size : i64) {
  // expected-error@+1 {{expected trailing function type with one argument and one result}}
  llvm.alloca %size x i32 : () -> ()
}

// -----

func.func @alloca_missing_input_type() {
  // expected-error@+1 {{expected trailing function type with one argument and one result}}
  llvm.alloca %size x i32 : () -> (!llvm.ptr)
}

// -----

func.func @alloca_missing_result_type() {
  // expected-error@+1 {{expected trailing function type with one argument and one result}}
  llvm.alloca %size x i32 : (i64) -> ()
}

// -----

func.func @alloca_non_function_type() {
  // expected-error@+1 {{expected trailing function type with one argument and one result}}
  llvm.alloca %size x i32 : !llvm.ptr
}

// -----

func.func @alloca_non_integer_alignment() {
  // expected-error@+1 {{expected integer alignment}}
  llvm.alloca %size x i32 {alignment = 3.0} : !llvm.ptr
}

// -----

func.func @gep_missing_input_result_type(%pos : i64, %base : !llvm.ptr) {
  // expected-error@+1 {{number of operands and types do not match: got 2 operands and 0 types}}
  llvm.getelementptr %base[%pos] : () -> (), i64
}

// -----

func.func @gep_missing_input_type(%pos : i64, %base : !llvm.ptr) {
  // expected-error@+1 {{number of operands and types do not match: got 2 operands and 0 types}}
  llvm.getelementptr %base[%pos] : () -> (!llvm.ptr), i64
}

// -----

func.func @gep_missing_result_type(%pos : i64, %base : !llvm.ptr) {
  // expected-error@+1 {{op requires one result}}
  llvm.getelementptr %base[%pos] : (!llvm.ptr, i64) -> (), i64
}

// -----

func.func @gep_non_function_type(%pos : i64, %base : !llvm.ptr) {
  // expected-error@+1 {{invalid kind of type specified: expected builtin.function, but found '!llvm.ptr'}}
  llvm.getelementptr %base[%pos] : !llvm.ptr
}

// -----

func.func @gep_too_few_dynamic(%base : !llvm.ptr) {
  // expected-error@+1 {{expected as many dynamic indices as specified in 'rawConstantIndices'}}
  %1 = "llvm.getelementptr"(%base) {elem_type = f32, rawConstantIndices = array<i32: -2147483648>} : (!llvm.ptr) -> !llvm.ptr
}

// -----

func.func @load_non_llvm_type(%foo : memref<f32>) {
  // expected-error@+1 {{op operand #0 must be LLVM pointer type}}
  llvm.load %foo : memref<f32> -> f32
}

// -----

func.func @load_syncscope(%ptr : !llvm.ptr) {
  // expected-error@below {{expected syncscope to be null for non-atomic access}}
  %1 = "llvm.load"(%ptr) {syncscope = "singlethread"} : (!llvm.ptr) -> (f32)
}

// -----

func.func @load_unsupported_ordering(%ptr : !llvm.ptr) {
  // expected-error@below {{unsupported ordering 'release'}}
  %1 = llvm.load %ptr atomic release {alignment = 4 : i64} : !llvm.ptr -> f32
}

// -----

func.func @load_unsupported_type(%ptr : !llvm.ptr) {
  // expected-error@below {{unsupported type 'f80' for atomic access}}
  %1 = llvm.load %ptr atomic monotonic {alignment = 16 : i64} : !llvm.ptr -> f80
}

// -----

func.func @load_unsupported_type(%ptr : !llvm.ptr) {
  // expected-error@below {{unsupported type 'i1' for atomic access}}
  %1 = llvm.load %ptr atomic monotonic {alignment = 16 : i64} : !llvm.ptr -> i1
}

// -----

func.func @load_unsupported_type(%ptr : !llvm.ptr) {
  // expected-error@below {{unsupported type 'i33' for atomic access}}
  %1 = llvm.load %ptr atomic monotonic {alignment = 16 : i64} : !llvm.ptr -> i33
}

// -----

func.func @load_unaligned_atomic(%ptr : !llvm.ptr) {
  // expected-error@below {{expected alignment for atomic access}}
  %1 = llvm.load %ptr atomic monotonic : !llvm.ptr -> f32
}

// -----

func.func @store_syncscope(%val : f32, %ptr : !llvm.ptr) {
  // expected-error@below {{expected syncscope to be null for non-atomic access}}
  "llvm.store"(%val, %ptr) {syncscope = "singlethread"} : (f32, !llvm.ptr) -> ()
}

// -----

func.func @store_unsupported_ordering(%val : f32, %ptr : !llvm.ptr) {
  // expected-error@below {{unsupported ordering 'acquire'}}
  llvm.store %val, %ptr atomic acquire {alignment = 4 : i64} : f32, !llvm.ptr
}

// -----

func.func @store_unsupported_type(%val : f80, %ptr : !llvm.ptr) {
  // expected-error@below {{unsupported type 'f80' for atomic access}}
  llvm.store %val, %ptr atomic monotonic {alignment = 16 : i64} : f80, !llvm.ptr
}

// -----

func.func @store_unsupported_type(%val : i1, %ptr : !llvm.ptr) {
  // expected-error@below {{unsupported type 'i1' for atomic access}}
  llvm.store %val, %ptr atomic monotonic {alignment = 16 : i64} : i1, !llvm.ptr
}

// -----

func.func @store_unsupported_type(%val : i48, %ptr : !llvm.ptr) {
  // expected-error@below {{unsupported type 'i48' for atomic access}}
  llvm.store %val, %ptr atomic monotonic {alignment = 16 : i64} : i48, !llvm.ptr
}

// -----

func.func @store_unaligned_atomic(%val : f32, %ptr : !llvm.ptr) {
  // expected-error@below {{expected alignment for atomic access}}
  llvm.store %val, %ptr atomic monotonic : f32, !llvm.ptr
}

// -----

func.func @invalid_call() {
  // expected-error@+1 {{'llvm.call' op must have either a `callee` attribute or at least an operand}}
  "llvm.call"() {op_bundle_sizes = array<i32>} : () -> ()
  llvm.return
}

// -----

func.func @call_missing_ptr_type(%callee : !llvm.func<i8 (i8)>, %arg : i8) {
  // expected-error@+1 {{expected indirect call to have 2 trailing types}}
  llvm.call %callee(%arg) : (i8) -> (i8)
  llvm.return
}

// -----

func.func private @standard_func_callee()

func.func @call_missing_ptr_type(%arg : i8) {
  // expected-error@+2 {{expected '('}}
  // expected-error@+1 {{expected direct call to have 1 trailing type}}
  llvm.call @standard_func_callee(%arg) : !llvm.ptr, (i8) -> (i8)
  llvm.return
}

// -----

func.func @call_non_pointer_type(%callee : !llvm.func<i8 (i8)>, %arg : i8) {
  // expected-error@+1 {{indirect call expects a pointer as callee: '!llvm.func<i8 (i8)>'}}
  llvm.call %callee(%arg) : !llvm.func<i8 (i8)>, (i8) -> (i8)
  llvm.return
}

// -----

func.func @call_non_function_type(%callee : !llvm.ptr, %arg : i8) {
  // expected-error@+2 {{expected '('}}
  // expected-error@+1 {{expected trailing function type}}
  llvm.call %callee(%arg) : !llvm.ptr, !llvm.func<i8 (i8)>
  llvm.return
}

// -----

func.func @call_void_result_type(%callee : !llvm.ptr, %arg : i8) {
  // expected-error@+1 {{expected a non-void result type}}
  llvm.call %callee(%arg) : !llvm.ptr, (i8) -> (!llvm.void)
  llvm.return
}

// -----

func.func @call_unknown_symbol() {
  // expected-error@+1 {{'llvm.call' op 'missing_callee' does not reference a symbol in the current scope}}
  llvm.call @missing_callee() : () -> ()
  llvm.return
}

// -----

func.func private @standard_func_callee()

func.func @call_non_llvm() {
  // expected-error@+1 {{'llvm.call' op 'standard_func_callee' does not reference a valid LLVM function}}
  llvm.call @standard_func_callee() : () -> ()
  llvm.return
}

// -----

func.func @call_non_llvm_arg(%arg0 : tensor<*xi32>) {
  // expected-error@+1 {{'llvm.call' op operand #0 must be variadic of LLVM dialect-compatible type}}
  "llvm.call"(%arg0) {operandSegmentSizes = array<i32: 1, 0>, op_bundle_sizes = array<i32>} : (tensor<*xi32>) -> ()
  llvm.return
}

// -----

func.func @call_non_llvm_res(%callee : !llvm.ptr) {
  // expected-error@+1 {{'llvm.call' op result #0 must be LLVM dialect-compatible type}}
  llvm.call %callee() : !llvm.ptr, () -> (tensor<*xi32>)
  llvm.return
}

// -----

llvm.func @callee_func(i8) -> ()

func.func @callee_arg_mismatch(%arg0 : i32) {
  // expected-error@+1 {{'llvm.call' op operand type mismatch for operand 0: 'i32' != 'i8'}}
  llvm.call @callee_func(%arg0) : (i32) -> ()
  llvm.return
}

// -----

llvm.func @callee_func() -> (i8)

func.func @callee_return_mismatch() {
  // expected-error@+1 {{'llvm.call' op result type mismatch: 'i32' != 'i8'}}
  %res = llvm.call @callee_func() : () -> (i32)
  llvm.return
}

// -----

func.func @call_too_many_results(%callee : !llvm.ptr) {
  // expected-error@+1 {{expected function with 0 or 1 result}}
  llvm.call %callee() : !llvm.ptr, () -> (i32, i32)
  llvm.return
}

// -----

llvm.func @void_func_result(%arg0: i32) {
  // expected-error@below {{expected no operands}}
  // expected-note@above {{when returning from function}}
  llvm.return %arg0: i32
}

// -----

llvm.func @non_void_func_no_result() -> i32 {
  // expected-error@below {{expected 1 operand}}
  // expected-note@above {{when returning from function}}
  llvm.return
}

// -----

llvm.func @func_result_mismatch(%arg0: f32) -> i32 {
  // expected-error@below {{mismatching result types}}
  // expected-note@above {{when returning from function}}
  llvm.return %arg0 : f32
}

// -----

func.func @constant_wrong_type() {
  // expected-error@+1 {{only supports integer, float, string or elements attributes}}
  llvm.mlir.constant(@constant_wrong_type) : !llvm.ptr
}

// -----

func.func @constant_wrong_type_string() {
  // expected-error@below {{expected array type of 3 i8 elements for the string constant}}
  llvm.mlir.constant("foo") : !llvm.ptr
}

// -----

llvm.func @array_attribute_one_element() -> !llvm.struct<(f64, f64)> {
  // expected-error @+1 {{expected array attribute of size 2}}
  %0 = llvm.mlir.constant([1.0 : f64]) : !llvm.struct<(f64, f64)>
  llvm.return %0 : !llvm.struct<(f64, f64)>
}

// -----

llvm.func @array_attribute_two_different_types() -> !llvm.struct<(f64, f64)> {
  // expected-error @+1 {{struct element at index 1 is of wrong type}}
  %0 = llvm.mlir.constant([1.0 : f64, 1.0 : f32]) : !llvm.struct<(f64, f64)>
  llvm.return %0 : !llvm.struct<(f64, f64)>
}

// -----

llvm.func @struct_wrong_attribute_type() -> !llvm.struct<(f64, f64)> {
  // expected-error @+1 {{expected array attribute for struct type}}
  %0 = llvm.mlir.constant(1.0 : f64) : !llvm.struct<(f64, f64)>
  llvm.return %0 : !llvm.struct<(f64, f64)>
}

// -----

llvm.func @struct_one_element() -> !llvm.struct<(f64)> {
  // expected-error @+1 {{expected array attribute of size 1}}
  %0 = llvm.mlir.constant([1.0 : f64, 1.0 : f64]) : !llvm.struct<(f64)>
  llvm.return %0 : !llvm.struct<(f64)>
}

// -----

llvm.func @struct_two_different_elements() -> !llvm.struct<(f64, f32)> {
  // expected-error @+1 {{struct element at index 1 is of wrong type}}
  %0 = llvm.mlir.constant([1.0 : f64, 1.0 : f64]) : !llvm.struct<(f64, f32)>
  llvm.return %0 : !llvm.struct<(f64, f32)>
}

// -----

llvm.func @struct_wrong_element_types() -> !llvm.struct<(!llvm.array<2 x f64>, !llvm.array<2 x f64>)> {
  // expected-error @+1 {{expected struct element types to be floating point type or integer type}}
  %0 = llvm.mlir.constant([dense<[1.0, 1.0]> : tensor<2xf64>, dense<[1.0, 1.0]> : tensor<2xf64>]) : !llvm.struct<(!llvm.array<2 x f64>, !llvm.array<2 x f64>)>
  llvm.return %0 : !llvm.struct<(!llvm.array<2 x f64>, !llvm.array<2 x f64>)>
}

// -----

llvm.func @const_wrong_number_of_elements() -> vector<5xf64> {
  // expected-error @+1{{type and attribute have a different number of elements: 5 vs. 2}}
  %0 = llvm.mlir.constant(dense<[1.0, 1.0]> : tensor<2xf64>) : vector<5xf64>
  llvm.return %0 : vector<5xf64>
}

// -----

llvm.func @scalable_vec_requires_splat() -> vector<[4]xf64> {
  // expected-error @+1{{scalable vector type requires a splat attribute}}
  %0 = llvm.mlir.constant(dense<[1.0, 1.0, 2.0, 2.0]> : tensor<4xf64>) : vector<[4]xf64>
  llvm.return %0 : vector<[4]xf64>
}


// -----

llvm.func @int_attr_requires_int_type() -> f32 {
  // expected-error @below{{expected integer type}}
  %0 = llvm.mlir.constant(1 : index) : f32
  llvm.return %0 : f32
}

// -----

llvm.func @vector_int_attr_requires_int_type() -> vector<2xf32> {
  // expected-error @below{{expected integer element type}}
  %0 = llvm.mlir.constant(dense<[1, 2]> : vector<2xi32>) : vector<2xf32>
  llvm.return %0 : vector<2xf32>
}

// -----

llvm.func @float_attr_and_type_required_same() -> f16 {
  // expected-error @below{{attribute and type have different float semantics}}
  %cst = llvm.mlir.constant(1.0 : bf16) : f16
  llvm.return %cst : f16
}

// -----

llvm.func @vector_float_attr_and_type_required_same() -> vector<2xf16> {
  // expected-error @below{{attribute and type have different float semantics}}
  %cst = llvm.mlir.constant(dense<[1.0, 2.0]> : vector<2xbf16>) : vector<2xf16>
  llvm.return %cst : vector<2xf16>
}

// -----

llvm.func @incompatible_integer_type_for_float_attr() -> i32 {
  // expected-error @below{{expected integer type of width 16}}
  %cst = llvm.mlir.constant(1.0 : f16) : i32
  llvm.return %cst : i32
}

// -----

llvm.func @vector_incompatible_integer_type_for_float_attr() -> vector<2xi8> {
  // expected-error @below{{expected integer type of width 16}}
  %cst = llvm.mlir.constant(dense<[1.0, 2.0]> : vector<2xf16>) : vector<2xi8>
  llvm.return %cst : vector<2xi8>
}

// -----

llvm.func @vector_with_non_vector_type() -> f32 {
  // expected-error @below{{expected vector or array type}}
  %cst = llvm.mlir.constant(dense<100.0> : vector<1xf64>) : f32
  llvm.return %cst : f32
}

// -----

llvm.func @array_attr_with_invalid_type() -> i32 {
  // expected-error @below{{expected array or struct type for array attribute}}
  %0 = llvm.mlir.constant([1 : i32]) : i32
  llvm.return %0 : i32
}

// -----

llvm.func @elements_attribute_incompatible_nested_array_struct1_type() -> !llvm.array<2 x array<2 x array<2 x struct<(i32)>>>> {
  // expected-error @below{{expected integer element type for integer elements attribute}}
  %0 = llvm.mlir.constant(dense<[[[1, 2], [3, 4]], [[42, 43], [44, 45]]]> : tensor<2x2x2xi32>) : !llvm.array<2 x array<2 x array<2 x struct<(i32)>>>>
  llvm.return %0 : !llvm.array<2 x array<2 x array<2 x struct<(i32)>>>>
}

// -----

llvm.func @elements_attribute_incompatible_nested_array_struct3_type() -> !llvm.array<2 x array<2 x array<2 x struct<(i32, i32, i32)>>>> {
  // expected-error @below{{expected integer element type for integer elements attribute}}
  %0 = llvm.mlir.constant(dense<[[[1, 2], [3, 4]], [[42, 43], [44, 45]]]> : tensor<2x2x2xi32>) : !llvm.array<2 x array<2 x array<2 x struct<(i32, i32, i32)>>>>
  llvm.return %0 : !llvm.array<2 x array<2 x array<2 x struct<(i32, i32, i32)>>>>
}

// -----

llvm.func @invalid_struct_element_type() -> !llvm.struct<(f64, array<2 x i32>)> {
  // expected-error @below{{expected struct element types to be floating point type or integer type}}
  %0 = llvm.mlir.constant([1.0 : f64, dense<[1, 2]> : tensor<2xi32>]) : !llvm.struct<(f64, array<2 x i32>)>
  llvm.return %0 : !llvm.struct<(f64, array<2 x i32>)>
}

// -----

llvm.func @wrong_struct_element_attr_type() -> !llvm.struct<(f64, f64)> {
  // expected-error @below{{expected element of array attribute to be floating point or integer}}
  %0 = llvm.mlir.constant([dense<[1, 2]> : tensor<2xi32>, 2.0 : f64]) : !llvm.struct<(f64, f64)>
  llvm.return %0 : !llvm.struct<(f64, f64)>
}

// -----

llvm.func @struct_wrong_attribute_element_type() -> !llvm.struct<(f64, f64)> {
  // expected-error @below{{struct element at index 0 is of wrong type}}
  %0 = llvm.mlir.constant([1.0 : f32, 1.0 : f32]) : !llvm.struct<(f64, f64)>
  llvm.return %0 : !llvm.struct<(f64, f64)>
}

// -----

func.func @insertvalue_non_llvm_type(%a : i32, %b : i32) {
  // expected-error@+2 {{expected LLVM IR Dialect type}}
  llvm.insertvalue %a, %b[0] : tensor<*xi32>
}

// -----

func.func @insertvalue_struct_out_of_bounds() {
  // expected-error@+2 {{position out of bounds}}
  llvm.insertvalue %a, %b[1] : !llvm.struct<(i32)>
}

// -----

func.func @insertvalue_array_out_of_bounds() {
  // expected-error@+2 {{position out of bounds}}
  llvm.insertvalue %a, %b[1] : !llvm.array<1 x i32>
}

// -----

func.func @insertvalue_wrong_nesting() {
  // expected-error@+2 {{expected LLVM IR structure/array type}}
  llvm.insertvalue %a, %b[0,0] : !llvm.struct<(i32)>
}

// -----

func.func @insertvalue_invalid_type(%a : !llvm.array<1 x i32>) -> !llvm.array<1 x i32> {
  // expected-error@+1 {{'llvm.insertvalue' op Type mismatch: cannot insert '!llvm.array<1 x i32>' into '!llvm.array<1 x i32>'}}
  %b = "llvm.insertvalue"(%a, %a) {position = array<i64: 0>} : (!llvm.array<1 x i32>, !llvm.array<1 x i32>) -> !llvm.array<1 x i32>
  return %b : !llvm.array<1 x i32>
}

// -----

func.func @extractvalue_invalid_type(%a : !llvm.array<4 x vector<8xf32>>) -> !llvm.array<4 x vector<8xf32>> {
  // expected-error@+1 {{'llvm.extractvalue' op Type mismatch: extracting from '!llvm.array<4 x vector<8xf32>>' should produce 'vector<8xf32>' but this op returns '!llvm.array<4 x vector<8xf32>>'}}
  %b = "llvm.extractvalue"(%a) {position = array<i64: 1>}
            : (!llvm.array<4 x vector<8xf32>>) -> !llvm.array<4 x vector<8xf32>>
  return %b : !llvm.array<4 x vector<8xf32>>
}

// -----

func.func @extractvalue_non_llvm_type(%a : i32, %b : tensor<*xi32>) {
  // expected-error@+2 {{expected LLVM IR Dialect type}}
  llvm.extractvalue %b[0] : tensor<*xi32>
}

// -----

func.func @extractvalue_struct_out_of_bounds() {
  // expected-error@+2 {{position out of bounds}}
  llvm.extractvalue %b[1] : !llvm.struct<(i32)>
}

// -----

func.func @extractvalue_array_out_of_bounds() {
  // expected-error@+2 {{position out of bounds}}
  llvm.extractvalue %b[1] : !llvm.array<1 x i32>
}

// -----

func.func @extractvalue_wrong_nesting() {
  // expected-error@+2 {{expected LLVM IR structure/array type}}
  llvm.extractvalue %b[0,0] : !llvm.struct<(i32)>
}

// -----

func.func @invalid_vector_type_1(%arg0: vector<4xf32>, %arg1: i32, %arg2: f32) {
  // expected-error@+1 {{invalid kind of type specified: expected builtin.vector, but found 'f32'}}
  %0 = llvm.extractelement %arg2[%arg1 : i32] : f32
}

// -----

func.func @invalid_vector_type_2(%arg0: vector<4xf32>, %arg1: i32, %arg2: f32) {
  // expected-error@+1 {{invalid kind of type specified: expected builtin.vector, but found 'f32'}}
  %0 = llvm.insertelement %arg2, %arg2[%arg1 : i32] : f32
}

// -----

func.func @invalid_vector_type_3(%arg0: vector<4xf32>, %arg1: i32, %arg2: f32) {
  // expected-error@+1 {{invalid kind of type specified: expected builtin.vector, but found 'f32'}}
  %0 = llvm.shufflevector %arg2, %arg2 [0, 0, 0, 0, 7] : f32
}

// -----

func.func @invalid_vector_type_4(%a : vector<4xf32>, %idx : i32) -> vector<4xf32> {
  // expected-error@+1 {{failed to verify that result type matches vector element type}}
  %b = "llvm.extractelement"(%a, %idx) : (vector<4xf32>, i32) -> vector<4xf32>
  return %b : vector<4xf32>
}

// -----

func.func @invalid_vector_type_5(%a : vector<4xf32>, %idx : i32) -> vector<4xf32> {
  // expected-error@+1 {{failed to verify that argument type matches vector element type}}
  %b = "llvm.insertelement"(%a, %a, %idx) : (vector<4xf32>, vector<4xf32>, i32) -> vector<4xf32>
  return %b : vector<4xf32>
}

// -----

func.func @zero_non_llvm_type() {
  // expected-error@+1 {{'llvm.mlir.zero' op result #0 must be LLVM dialect-compatible type, but got 'tensor<4xi32>'}}
  llvm.mlir.zero : tensor<4xi32>
}

// -----

func.func @nvvm_invalid_shfl_pred_1(%arg0 : i32, %arg1 : i32, %arg2 : i32, %arg3 : i32) {
  // expected-error@+1 {{expected return type to be a two-element struct with i1 as the second element}}
  %0 = nvvm.shfl.sync bfly %arg0, %arg3, %arg1, %arg2 {return_value_and_is_valid} : i32 -> i32
}

// -----

func.func @nvvm_invalid_shfl_pred_2(%arg0 : i32, %arg1 : i32, %arg2 : i32, %arg3 : i32) {
  // expected-error@+1 {{expected return type to be a two-element struct with i1 as the second element}}
  %0 = nvvm.shfl.sync bfly %arg0, %arg3, %arg1, %arg2 {return_value_and_is_valid} : i32 -> !llvm.struct<(i32)>
}

// -----

func.func @nvvm_invalid_shfl_pred_3(%arg0 : i32, %arg1 : i32, %arg2 : i32, %arg3 : i32) {
  // expected-error@+1 {{expected return type to be a two-element struct with i1 as the second element}}
  %0 = nvvm.shfl.sync bfly %arg0, %arg3, %arg1, %arg2 {return_value_and_is_valid} : i32 -> !llvm.struct<(i32, i32)>
}

// -----

func.func @nvvm_invalid_mma_0(%a0 : f16, %a1 : f16,
                         %b0 : vector<2xf16>, %b1 : vector<2xf16>,
                         %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32,
                         %c4 : f32, %c5 : f32, %c6 : f32, %c7 : f32) {
  // expected-error@+1 {{Could not match types for the A operands; expected one of 2xvector<2xf16> but got f16, f16}}
  %0 = nvvm.mma.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7]
    {layoutA=#nvvm.mma_layout<row>, layoutB=#nvvm.mma_layout<col>, shape = #nvvm.shape<m = 8, n = 8, k = 4>} : (f16, vector<2xf16>, f32) -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
  llvm.return %0 : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
}

// -----

func.func @nvvm_invalid_mma_1(%a0 : vector<2xf16>, %a1 : vector<2xf16>,
                         %b0 : vector<2xf16>, %b1 : vector<2xf16>,
                         %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32,
                         %c4 : f32, %c5 : f32, %c6 : f32, %c7 : f32) {
  // expected-error@+1 {{Could not match allowed types for the result; expected one of !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>, !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)> but got !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f16)>}}
  %0 = nvvm.mma.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7]
    {layoutA=#nvvm.mma_layout<row>, layoutB=#nvvm.mma_layout<col>, shape = #nvvm.shape<m = 8, n = 8, k = 4>} : (vector<2xf16>, vector<2xf16>, f32) -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f16)>
  llvm.return %0 : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f16)>
}

// -----

func.func @nvvm_invalid_mma_2(%a0 : vector<2xf16>, %a1 : vector<2xf16>,
                         %b0 : vector<2xf16>, %b1 : vector<2xf16>,
                         %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32,
                         %c4 : f32, %c5 : f32, %c6 : f32, %c7 : f32) {
  // expected-error@+1 {{op requires attribute 'layoutA'}}
  %0 = nvvm.mma.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7]
    {shape = #nvvm.shape<m = 8, n = 8, k = 4>}: (vector<2xf16>, vector<2xf16>, f32) -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
  llvm.return %0 : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
}

// -----

func.func @nvvm_invalid_mma_3(%a0 : vector<2xf16>, %a1 : vector<2xf16>,
                         %b0 : vector<2xf16>, %b1 : vector<2xf16>,
                         %c0 : vector<2xf16>, %c1 : vector<2xf16>) {
  // expected-error@+1 {{unimplemented variant for MMA shape <8, 8, 16>}}
  %0 = nvvm.mma.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1] {layoutA=#nvvm.mma_layout<row>, layoutB=#nvvm.mma_layout<col>, shape = #nvvm.shape<m = 8, n = 8, k = 16>} : (vector<2xf16>, vector<2xf16>, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  llvm.return %0 : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
}

// -----

func.func @nvvm_invalid_mma_8(%a0 : i32, %a1 : i32,
                               %b0 : i32,
                               %c0 : i32, %c1 : i32, %c2 : i32, %c3 : i32) {
  // expected-error@+1 {{op requires b1Op attribute}}
  %0 = nvvm.mma.sync A[%a0, %a1] B[%b0] C[%c0, %c1, %c2, %c3]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     multiplicandAPtxType = #nvvm.mma_type<b1>, multiplicandBPtxType = #nvvm.mma_type<b1>,
     shape = #nvvm.shape<m = 16, n = 8, k = 128>} : (i32, i32, i32) -> !llvm.struct<(i32,i32,i32,i32)>
  llvm.return %0 : !llvm.struct<(i32,i32,i32,i32)>
}

// -----

func.func @atomicrmw_mismatched_operands(%f32_ptr : !llvm.ptr, %f32 : f32) {
  // expected-error@+1 {{op failed to verify that result #0 and operand #1 have the same type}}
  %0 = "llvm.atomicrmw"(%f32_ptr, %f32) {bin_op=11, ordering=1} : (!llvm.ptr, f32) -> i32
  llvm.return
}

// -----

func.func @atomicrmw_expected_float(%i32_ptr : !llvm.ptr, %i32 : i32) {
  // expected-error@+1 {{expected LLVM IR floating point type}}
  %0 = llvm.atomicrmw fadd %i32_ptr, %i32 unordered : !llvm.ptr, i32
  llvm.return
}

// -----

func.func @atomicrmw_scalable_vector(%ptr : !llvm.ptr, %f32_vec : vector<[2]xf32>) {
  // expected-error@+1 {{'val' must be floating point LLVM type or LLVM pointer type or signless integer or LLVM dialect-compatible fixed-length vector type}}
  %0 = llvm.atomicrmw fadd %ptr, %f32_vec unordered : !llvm.ptr, vector<[2]xf32>
  llvm.return
}

// -----

func.func @atomicrmw_vector_expected_float(%ptr : !llvm.ptr, %i32_vec : vector<3xi32>) {
  // expected-error@+1 {{expected LLVM IR floating point type for vector element}}
  %0 = llvm.atomicrmw fadd %ptr, %i32_vec unordered : !llvm.ptr, vector<3xi32>
  llvm.return
}

// -----

func.func @atomicrmw_unexpected_xchg_type(%i1_ptr : !llvm.ptr, %i1 : i1) {
  // expected-error@+1 {{unexpected LLVM IR type for 'xchg' bin_op}}
  %0 = llvm.atomicrmw xchg %i1_ptr, %i1 unordered : !llvm.ptr, i1
  llvm.return
}

// -----

func.func @atomicrmw_expected_int(%f32_ptr : !llvm.ptr, %f32 : f32) {
  // expected-error@+1 {{expected LLVM IR integer type}}
  %0 = llvm.atomicrmw max %f32_ptr, %f32 unordered : !llvm.ptr, f32
  llvm.return
}

// -----

func.func @cmpxchg_mismatched_value_operands(%ptr : !llvm.ptr, %i32 : i32, %i64 : i64) {
  // expected-error@+1 {{op failed to verify that operand #1 and operand #2 have the same type}}
  %0 = "llvm.cmpxchg"(%ptr, %i32, %i64) {success_ordering=2,failure_ordering=2} : (!llvm.ptr, i32, i64) -> !llvm.struct<(i32, i1)>
  llvm.return
}

// -----

func.func @cmpxchg_mismatched_result(%ptr : !llvm.ptr, %i64 : i64) {
  // expected-error@+1 {{op failed to verify that result #0 has an LLVM struct type consisting of the type of operand #2 and a bool}}
  %0 = "llvm.cmpxchg"(%ptr, %i64, %i64) {success_ordering=2,failure_ordering=2} : (!llvm.ptr, i64, i64) -> !llvm.struct<(i64, i64)>
  llvm.return
}

// -----

func.func @cmpxchg_unexpected_type(%i1_ptr : !llvm.ptr, %i1 : i1) {
  // expected-error@+1 {{unexpected LLVM IR type}}
  %0 = llvm.cmpxchg %i1_ptr, %i1, %i1 monotonic monotonic : !llvm.ptr, i1
  llvm.return
}

// -----

func.func @cmpxchg_at_least_monotonic_success(%i32_ptr : !llvm.ptr, %i32 : i32) {
  // expected-error@+1 {{ordering must be at least 'monotonic'}}
  %0 = llvm.cmpxchg %i32_ptr, %i32, %i32 unordered monotonic : !llvm.ptr, i32
  llvm.return
}

// -----

func.func @cmpxchg_at_least_monotonic_failure(%i32_ptr : !llvm.ptr, %i32 : i32) {
  // expected-error@+1 {{ordering must be at least 'monotonic'}}
  %0 = llvm.cmpxchg %i32_ptr, %i32, %i32 monotonic unordered : !llvm.ptr, i32
  llvm.return
}

// -----

func.func @cmpxchg_failure_release(%i32_ptr : !llvm.ptr, %i32 : i32) {
  // expected-error@+1 {{failure ordering cannot be 'release' or 'acq_rel'}}
  %0 = llvm.cmpxchg %i32_ptr, %i32, %i32 acq_rel release : !llvm.ptr, i32
  llvm.return
}

// -----

func.func @cmpxchg_failure_acq_rel(%i32_ptr : !llvm.ptr, %i32 : i32) {
  // expected-error@+1 {{failure ordering cannot be 'release' or 'acq_rel'}}
  %0 = llvm.cmpxchg %i32_ptr, %i32, %i32 acq_rel acq_rel : !llvm.ptr, i32
  llvm.return
}

// -----

llvm.func @foo(i32) -> i32
llvm.func @__gxx_personality_v0(...) -> i32

llvm.func @bad_landingpad(%arg0: !llvm.ptr) -> i32 attributes { personality = @__gxx_personality_v0} {
  %0 = llvm.mlir.constant(3 : i32) : i32
  %1 = llvm.mlir.constant(2 : i32) : i32
  %2 = llvm.invoke @foo(%1) to ^bb1 unwind ^bb2 : (i32) -> i32
^bb1:  // pred: ^bb0
  llvm.return %1 : i32
^bb2:  // pred: ^bb0
  // expected-error@+1 {{clause #0 is not a known constant - null, addressof, bitcast}}
  %3 = llvm.landingpad cleanup (catch %1 : i32) (catch %arg0 : !llvm.ptr) : !llvm.struct<(ptr, i32)>
  llvm.return %0 : i32
}

// -----

llvm.func @foo(i32) -> i32
llvm.func @__gxx_personality_v0(...) -> i32

llvm.func @caller(%arg0: i32) -> i32 attributes { personality = @__gxx_personality_v0} {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.ptr : (i32) -> !llvm.ptr
  // expected-note@+1 {{global addresses expected as operand to bitcast used in clauses for landingpad}}
  %2 = llvm.bitcast %1 : !llvm.ptr to !llvm.ptr
  %3 = llvm.invoke @foo(%0) to ^bb1 unwind ^bb2 : (i32) -> i32
^bb1: // pred: ^bb0
  llvm.return %0 : i32
^bb2: // pred: ^bb0
  // expected-error@+1 {{constant clauses expected}}
  %5 = llvm.landingpad (catch %2 : !llvm.ptr) : !llvm.struct<(ptr, i32)>
  llvm.return %0 : i32
}

// -----

llvm.func @foo(i32) -> i32
llvm.func @__gxx_personality_v0(...) -> i32

llvm.func @caller(%arg0: i32) -> i32 attributes { personality = @__gxx_personality_v0} {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.invoke @foo(%0) to ^bb1 unwind ^bb2 : (i32) -> i32
^bb1: // pred: ^bb0
  llvm.return %0 : i32
^bb2: // pred: ^bb0
  // expected-error@+1 {{landingpad instruction expects at least one clause or cleanup attribute}}
  %2 = llvm.landingpad : !llvm.struct<(ptr, i32)>
  llvm.return %0 : i32
}

// -----

llvm.func @foo(i32) -> i32
llvm.func @__gxx_personality_v0(...) -> i32

// expected-error@below {{'llvm.resume' should have a consistent input type inside a function}}
llvm.func @caller(%arg0: i32) -> i32 attributes { personality = @__gxx_personality_v0 } {
  %0 = llvm.invoke @foo(%arg0) to ^bb1 unwind ^bb2 : (i32) -> i32
^bb1:
  %1 = llvm.invoke @foo(%0) to ^bb3 unwind ^bb4 : (i32) -> i32
^bb2:
  %2 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
  llvm.resume %0 : i32
^bb3:
  llvm.return %1 : i32
^bb4:
  %3 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
  llvm.resume %3 : !llvm.struct<(ptr, i32)>
}

// -----

llvm.func @foo(i32) -> i32
llvm.func @__gxx_personality_v0(...) -> i32

// expected-error@below {{'llvm.landingpad' should have a consistent result type inside a function}}
llvm.func @caller(%arg0: i32) -> i32 attributes { personality = @__gxx_personality_v0 } {
  %0 = llvm.invoke @foo(%arg0) to ^bb1 unwind ^bb2 : (i32) -> i32
^bb1:
  %1 = llvm.invoke @foo(%0) to ^bb3 unwind ^bb4 : (i32) -> i32
^bb2:
  %2 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
  llvm.resume %2 : !llvm.struct<(ptr, i32)>
^bb3:
  llvm.return %1 : i32
^bb4:
  %3 = llvm.landingpad cleanup : i32
  llvm.resume %3 : i32
}

// -----

llvm.func @foo(i32) -> i32

llvm.func @caller(%arg0: i32) -> i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.invoke @foo(%0) to ^bb1 unwind ^bb2 : (i32) -> i32
^bb1: // pred: ^bb0
  llvm.return %0 : i32
^bb2: // pred: ^bb0
  // expected-error@+1 {{llvm.landingpad needs to be in a function with a personality}}
  %2 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
  llvm.resume %2 : !llvm.struct<(ptr, i32)>
}

// -----

func.func @invalid_ordering_in_fence() {
  // expected-error @+1 {{can be given only acquire, release, acq_rel, and seq_cst orderings}}
  llvm.fence syncscope("agent") monotonic
}

// -----

// expected-error @+1 {{invalid data layout descriptor}}
module attributes {llvm.data_layout = "#vjkr32"} {
  func.func @invalid_data_layout()
}

// -----

func.func @switch_superfluous_comma(%arg0 : i64) {
  // expected-error@+3 {{custom op 'llvm.switch' expected integer value}}
  llvm.switch %arg0 : i32, ^bb1 [
    42: ^bb2,
  ]
^bb1:
  llvm.return
^bb2:
  llvm.return
}

// -----

func.func @switch_wrong_number_of_weights(%arg0 : i32) {
  // expected-error@+1 {{expects number of branch weights to match number of successors: 3 vs 2}}
  llvm.switch %arg0 : i32, ^bb1 [
    42: ^bb2(%arg0, %arg0 : i32, i32)
  ] {branch_weights = array<i32: 13, 17, 19>}

^bb1: // pred: ^bb0
  llvm.return

^bb2(%1: i32, %2: i32): // pred: ^bb0
  llvm.return
}

// -----

func.func @switch_case_type_mismatch(%arg0 : i64) {
  // expected-error@below {{expects case value type to match condition value type}}
  "llvm.switch"(%arg0)[^bb1, ^bb2] <{case_operand_segments = array<i32: 0>, case_values = dense<42> : vector<1xi32>, operandSegmentSizes = array<i32: 1, 0, 0>}> : (i64) -> ()
^bb1: // pred: ^bb0
  llvm.return
^bb2: // pred: ^bb0
  llvm.return
}

// -----

// expected-error@below {{expected zero value for 'common' linkage}}
llvm.mlir.global common @non_zero_global_common_linkage(42 : i32) : i32

// -----

// expected-error@below {{expected zero value for 'common' linkage}}
llvm.mlir.global common @non_zero_compound_global_common_linkage(dense<[0, 0, 0, 1, 0]> : vector<5xi32>) : !llvm.array<5 x i32>

// -----

// expected-error@below {{expected array type for 'appending' linkage}}
llvm.mlir.global appending @non_array_type_global_appending_linkage() : i32

// -----

module {
  llvm.func @accessGroups(%arg0 : !llvm.ptr) {
      // expected-error@below {{attribute 'access_groups' failed to satisfy constraint: LLVM dialect access group metadata array}}
      %0 = llvm.load %arg0 { "access_groups" = [@func1] } : !llvm.ptr -> i32
      llvm.return
  }
  llvm.func @func1() {
    llvm.return
  }
}

// -----

module {
  llvm.func @accessGroups(%arg0 : !llvm.ptr, %arg1 : i32, %arg2 : i32) {
      // expected-error@below {{attribute 'access_groups' failed to satisfy constraint: LLVM dialect access group metadata array}}
      %0 = llvm.cmpxchg %arg0, %arg1, %arg2 acq_rel monotonic { "access_groups" = [@metadata::@scope] } : !llvm.ptr, i32
      llvm.return
  }
  llvm.metadata @metadata {
    llvm.func @scope()
  }
}

// -----

module {
  llvm.func @aliasScope(%arg0 : !llvm.ptr, %arg1 : i32, %arg2 : i32) {
      // expected-error@below {{attribute 'alias_scopes' failed to satisfy constraint: LLVM dialect alias scope array}}
      %0 = llvm.cmpxchg %arg0, %arg1, %arg2 acq_rel monotonic { "alias_scopes" = "test" } : !llvm.ptr, i32
      llvm.return
  }
}

// -----

module {
  llvm.func @noAliasScopes(%arg0 : !llvm.ptr) {
      // expected-error@below {{attribute 'noalias_scopes' failed to satisfy constraint: LLVM dialect alias scope array}}
      %0 = llvm.load %arg0 { "noalias_scopes" = "test" } : !llvm.ptr -> i32
      llvm.return
  }
}

// -----

llvm.func @wmmaLoadOp_invalid_mem_space(%arg0: !llvm.ptr<5>, %arg1: i32) {
  // expected-error@+1 {{'nvvm.wmma.load' op expected source pointer in memory space 0, 1, 3}}
  %0 = nvvm.wmma.load %arg0, %arg1
    {eltype = #nvvm.mma_type<f16>, frag = #nvvm.mma_frag<a>, k = 16 : i32, layout = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
    : (!llvm.ptr<5>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
  llvm.return
}

// -----

llvm.func @wmmaLoadOp_invalid_AOp(%arg0: !llvm.ptr<3>, %arg1: i32) {
  // expected-error@+1 {{'nvvm.wmma.load' op expected destination type is a structure of 8 elements of type 'vector<2xf16>'}}
  %0 = nvvm.wmma.load %arg0, %arg1
  {eltype = #nvvm.mma_type<f16>, frag = #nvvm.mma_frag<a>, k = 16 : i32, layout = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
  : (!llvm.ptr<3>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
  llvm.return
}

// -----

llvm.func @wmmaLoadOp_invalid_BOp(%arg0: !llvm.ptr<3>, %arg1: i32) {
  // expected-error@+1 {{'nvvm.wmma.load' op expected destination type is a structure of 8 elements of type 'vector<2xf16>'}}
 %0 = nvvm.wmma.load %arg0, %arg1
 {eltype = #nvvm.mma_type<f16>, frag = #nvvm.mma_frag<b>, k = 16 : i32, layout = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
 : (!llvm.ptr<3>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>

  llvm.return
}

// -----

llvm.func @wmmaLoadOp_invalid_COp(%arg0: !llvm.ptr<3>, %arg1: i32) {
  // expected-error@+1 {{'nvvm.wmma.load' op expected destination type is a structure of 4 elements of type 'vector<2xf16>'}}
 %0 = nvvm.wmma.load %arg0, %arg1
   {eltype = #nvvm.mma_type<f16>, frag = #nvvm.mma_frag<c>, k = 16 : i32, layout = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
   : (!llvm.ptr<3>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>

  llvm.return
}

// -----

llvm.func @wmmaStoreOp_invalid_mem_space(%arg0: !llvm.ptr<5>, %arg1: i32,
                            %arg2: vector<2 x f16>, %arg3: vector<2 x f16>,
                            %arg4: vector<2 x f16>, %arg5: vector<2 xf16>) {
  // expected-error@+1 {{'nvvm.wmma.store' op expected operands to be a source pointer in memory space 0, 1, 3}}
  nvvm.wmma.store %arg0, %arg1, %arg2, %arg3, %arg4, %arg5
    {eltype = #nvvm.mma_type<f16>, k = 16 : i32, layout = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
    : !llvm.ptr<5>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>
  llvm.return
}

// -----

llvm.func @gpu_wmma_mma_op_invalid_operands(%arg0: vector<2 x f16>, %arg1: vector<2 x f16>,
                        %arg2: vector<2 x f16>, %arg3: vector<2 x f16>,
                        %arg4: vector<2 x f16>, %arg5: vector<2 x f16>,
                        %arg6: vector<2 x f16>, %arg7: vector<2 x f16>,
                        %arg8: vector<2 x f16>, %arg9: vector<2 x f16>,
                        %arg10: vector<2 x f16>, %arg11: vector<2 x f16>,
                        %arg12: vector<2 x f16>, %arg13: vector<2 x f16>,
                        %arg14: vector<2 x f16>, %arg15: vector<2 x f16>,
                        %arg16: vector<2 x f16>, %arg17: vector<2 x f16>,
                        %arg18: vector<2 x f16>) {
  // expected-error@+1 {{'nvvm.wmma.mma' op expected 20 arguments}}
  %0 = nvvm.wmma.mma %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18
    {eltypeA = #nvvm.mma_type<f16>, eltypeB = #nvvm.mma_type<f16>, k = 16 : i32, layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
    : (vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>,
       vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>)
      -> !llvm.struct<(vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>)>
  llvm.return
}

// -----

llvm.func @gpu_wmma_mma_op_results(%arg0: vector<2 x f16>, %arg1: vector<2 x f16>,
                        %arg2: vector<2 x f16>, %arg3: vector<2 x f16>,
                        %arg4: vector<2 x f16>, %arg5: vector<2 x f16>,
                        %arg6: vector<2 x f16>, %arg7: vector<2 x f16>,
                        %arg8: vector<2 x f16>, %arg9: vector<2 x f16>,
                        %arg10: vector<2 x f16>, %arg11: vector<2 x f16>,
                        %arg12: vector<2 x f16>, %arg13: vector<2 x f16>,
                        %arg14: vector<2 x f16>, %arg15: vector<2 x f16>,
                        %arg16: vector<2 x f16>, %arg17: vector<2 x f16>,
                        %arg18: vector<2 x f16>, %arg19: vector<2 x f16>) {
  // expected-error@+1 {{'nvvm.wmma.mma' op expected destination type is a structure of 4 elements of type 'vector<2xf16>'}}
  %0 = nvvm.wmma.mma %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19
    {eltypeA = #nvvm.mma_type<f16>, eltypeB = #nvvm.mma_type<f16>, k = 16 : i32, layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
    : (vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>,
       vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>)
      -> !llvm.struct<(vector<2 x f16>, vector<2 x f16>, vector<2 x f16>)>  llvm.return
}

// -----

llvm.func @gpu_wmma_mma_op_invalid_ab_operands(%arg0: vector<2 x f16>, %arg1: vector<2 x f16>,
                        %arg2: vector<2 x f16>, %arg3: vector<2 x f16>,
                        %arg4: vector<2 x f16>, %arg5: vector<2 x f16>,
                        %arg6: vector<2 x f16>, %arg7: vector<2 x f16>,
                        %arg8: vector<2 x f16>, %arg9: vector<2 x f16>,
                        %arg10: vector<2 x f16>, %arg11: vector<2 x f16>,
                        %arg12: vector<2 x f16>, %arg13: vector<2 x f16>,
                        %arg14: vector<2 x f16>, %arg15: f32,
                        %arg16: f32, %arg17: f32, %arg18: f32, %arg19: f32,
                        %arg20: f32, %arg21: f32, %arg22: f32, %arg23: f32) {
  // expected-error@+1 {{'nvvm.wmma.mma' op expected argument 15 to be of type 'vector<2xf16>'}}
  %0 = nvvm.wmma.mma %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23
    {eltypeA = #nvvm.mma_type<f16>, eltypeB = #nvvm.mma_type<f32>, k = 16 : i32, layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
    : (vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, f32, f32, f32, f32, f32, f32, f32, f32, f32) -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
  llvm.return
}

// -----

llvm.func @gpu_wmma_mma_op_invalid_c_operand(%arg0: vector<2 x f16>, %arg1: vector<2 x f16>,
                        %arg2: vector<2 x f16>, %arg3: vector<2 x f16>,
                        %arg4: vector<2 x f16>, %arg5: vector<2 x f16>,
                        %arg6: vector<2 x f16>, %arg7: vector<2 x f16>,
                        %arg8: vector<2 x f16>, %arg9: vector<2 x f16>,
                        %arg10: vector<2 x f16>, %arg11: vector<2 x f16>,
                        %arg12: vector<2 x f16>, %arg13: vector<2 x f16>,
                        %arg14: vector<2 x f16>, %arg15: vector<2xf16>,
                        %arg16: f32, %arg17: f32, %arg18: f32, %arg19: f32,
                        %arg20: f32, %arg21: f32, %arg22: f32, %arg23: vector<2xf16>) {
  // expected-error@+1 {{'nvvm.wmma.mma' op expected argument 23 to be of type 'f32'}}
  %0 = nvvm.wmma.mma %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23
    {eltypeA = #nvvm.mma_type<f16>, eltypeB = #nvvm.mma_type<f32>, k = 16 : i32, layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
    : (vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, f32, f32, f32, f32, f32, f32, f32, vector<2xf16>) -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
  llvm.return
}

// -----

llvm.func @gpu_wmma_mma_op_invalid_result(%arg0: vector<2 x f16>, %arg1: vector<2 x f16>,
                        %arg2: vector<2 x f16>, %arg3: vector<2 x f16>,
                        %arg4: vector<2 x f16>, %arg5: vector<2 x f16>,
                        %arg6: vector<2 x f16>, %arg7: vector<2 x f16>,
                        %arg8: vector<2 x f16>, %arg9: vector<2 x f16>,
                        %arg10: vector<2 x f16>, %arg11: vector<2 x f16>,
                        %arg12: vector<2 x f16>, %arg13: vector<2 x f16>,
                        %arg14: vector<2 x f16>, %arg15: vector<2xf16>,
                        %arg16: f32, %arg17: f32, %arg18: f32, %arg19: f32,
                        %arg20: f32, %arg21: f32, %arg22: f32, %arg23: f32) {
  // expected-error@+1 {{'nvvm.wmma.mma' op expected destination type is a structure of 8 elements of type 'f32'}}
  %0 = nvvm.wmma.mma %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23
    {eltypeA = #nvvm.mma_type<f16>, eltypeB = #nvvm.mma_type<f32>, k = 16 : i32, layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
    : (vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, f32, f32, f32, f32, f32, f32, f32, f32) -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, vector<2xf16>)>
  llvm.return
}

// -----

llvm.func @wmmald_matrix(%arg0: !llvm.ptr) {
  // expected-error@+1 {{'nvvm.ldmatrix' op expected source pointer in memory space 3}}
  %l = nvvm.ldmatrix %arg0 {num = 1 : i32, layout = #nvvm.mma_layout<row>} : (!llvm.ptr) -> i32
  llvm.return
}

// -----

llvm.func @wmmald_matrix(%arg0: !llvm.ptr<3>) {
  // expected-error@+1 {{'nvvm.ldmatrix' op expected num attribute to be 1, 2 or 4}}
  %l = nvvm.ldmatrix %arg0 {num = 3 : i32, layout = #nvvm.mma_layout<row>} : (!llvm.ptr<3>) -> i32
  llvm.return
}

// -----

llvm.func @wmmald_matrix(%arg0: !llvm.ptr<3>) {
  // expected-error@+1 {{'nvvm.ldmatrix' op expected destination type is i32}}
  %l = nvvm.ldmatrix %arg0 {num = 1 : i32, layout = #nvvm.mma_layout<row>} : (!llvm.ptr<3>) -> !llvm.struct<(i32)>
  llvm.return
}

// -----

llvm.func @wmmald_matrix(%arg0: !llvm.ptr<3>) {
  // expected-error@+1 {{'nvvm.ldmatrix' op expected destination type is a structure of 4 elements of type i32}}
  %l = nvvm.ldmatrix %arg0 {num = 4 : i32, layout = #nvvm.mma_layout<row>} : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32)>
  llvm.return
}

// -----

llvm.func @caller() {
  // expected-error @below {{expected function call to produce a value}}
  llvm.call @callee() : () -> ()
  llvm.return
}

llvm.func @callee() -> i32

// -----

llvm.func @caller() {
  // expected-error @below {{calling function with void result must not produce values}}
  %0 = llvm.call @callee() : () -> i32
  llvm.return
}

llvm.func @callee() -> ()

// -----

llvm.func @caller() {
  // expected-error @below {{expected function with 0 or 1 result}}
  %0:2 = llvm.call @callee() : () -> (i32, f32)
  llvm.return
}

llvm.func @callee() -> !llvm.struct<(i32, f32)>

// -----

func.func @bitcast(%arg0: vector<2x3xf32>) {
  // expected-error @below {{op operand #0 must be LLVM-compatible non-aggregate type}}
  llvm.bitcast %arg0 : vector<2x3xf32> to vector<2x3xi32>
  return
}

// -----

func.func @cp_async(%arg0: !llvm.ptr<3>, %arg1: !llvm.ptr<1>) {
  // expected-error @below {{expected byte size to be either 4, 8 or 16.}}
  nvvm.cp.async.shared.global %arg0, %arg1, 32, cache = cg : !llvm.ptr<3>, !llvm.ptr<1>
  return
}

// -----

func.func @cp_async(%arg0: !llvm.ptr<3>, %arg1: !llvm.ptr<1>) {
  // expected-error @below {{CG cache modifier is only support for 16 bytes copy.}}
  nvvm.cp.async.shared.global %arg0, %arg1, 8, cache = cg : !llvm.ptr<3>, !llvm.ptr<1>
  return
}

// -----

func.func @mapa(%a: !llvm.ptr, %b : i32) {
  // expected-error @below {{`res` and `a` should have the same type}}
  %0 = nvvm.mapa %a, %b: !llvm.ptr -> !llvm.ptr<3>
  return
}

// -----

func.func @gep_struct_variable(%arg0: !llvm.ptr, %arg1: i32, %arg2: i32) {
  // expected-error @below {{op expected index 1 indexing a struct to be constant}}
  llvm.getelementptr %arg0[%arg1, %arg1] : (!llvm.ptr, i32, i32) -> !llvm.ptr, !llvm.struct<(i32)>
  return
}

// -----

func.func @gep_out_of_bounds(%ptr: !llvm.ptr, %idx: i64) {
  // expected-error @below {{index 2 indexing a struct is out of bounds}}
  llvm.getelementptr %ptr[%idx, 1, 3] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(i32, struct<(i32, f32)>)>
  return
}

// -----

func.func @non_splat_shuffle_on_scalable_vector(%arg0: vector<[4]xf32>) {
  // expected-error@below {{expected a splat operation for scalable vectors}}
  %0 = llvm.shufflevector %arg0, %arg0 [0, 0, 0, 1] : vector<[4]xf32>
  return
}

// -----

llvm.mlir.global internal @side_effecting_global() : !llvm.struct<(i8)> {
  %0 = llvm.mlir.constant(1 : i64) : i64
  // expected-error@below {{ops with side effects not allowed in global initializers}}
  %1 = llvm.alloca %0 x !llvm.struct<(i8)> : (i64) -> !llvm.ptr
  %2 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i8)>
  llvm.return %2 : !llvm.struct<(i8)>
}

// -----

func.func @insert_vector_invalid_source_vector_size(%arg0 : vector<16385xi8>, %arg1 : vector<[16]xi8>) {
  // expected-error@+1 {{op failed to verify that vectors are not bigger than 2^17 bits.}}
  %0 = llvm.intr.vector.insert %arg0, %arg1[0] : vector<16385xi8> into vector<[16]xi8>
}

// -----

func.func @insert_vector_invalid_dest_vector_size(%arg0 : vector<16xi8>, %arg1 : vector<[16385]xi8>) {
  // expected-error@+1 {{op failed to verify that vectors are not bigger than 2^17 bits.}}
  %0 = llvm.intr.vector.insert %arg0, %arg1[0] : vector<16xi8> into vector<[16385]xi8>
}

// -----

func.func @insert_scalable_into_fixed_length_vector(%arg0 : vector<[8]xf32>, %arg1 : vector<16xf32>) {
  // expected-error@+1 {{op failed to verify that it is not inserting scalable into fixed-length vectors.}}
  %0 = llvm.intr.vector.insert %arg0, %arg1[0] : vector<[8]xf32> into vector<16xf32>
}

// -----

func.func @extract_vector_invalid_source_vector_size(%arg0 : vector<[16385]xi8>) {
  // expected-error@+1 {{op failed to verify that vectors are not bigger than 2^17 bits.}}
  %0 = llvm.intr.vector.extract %arg0[0] : vector<16xi8> from vector<[16385]xi8>
}

// -----

func.func @extract_vector_invalid_result_vector_size(%arg0 : vector<[16]xi8>) {
  // expected-error@+1 {{op failed to verify that vectors are not bigger than 2^17 bits.}}
  %0 = llvm.intr.vector.extract %arg0[0] : vector<16385xi8> from vector<[16]xi8>
}

// -----

func.func @extract_scalable_from_fixed_length_vector(%arg0 : vector<16xf32>) {
  // expected-error@+1 {{op failed to verify that it is not extracting scalable from fixed-length vectors.}}
  %0 = llvm.intr.vector.extract %arg0[0] : vector<[8]xf32> from vector<16xf32>
}


// -----

func.func @vector_interleave2_bad_type0(%vec1: vector<[2]xf16>, %vec2 : vector<[4]xf16>) {
  // expected-error@+1 {{op failed to verify that all of {vec1, vec2} have same type}}
  %0 = "llvm.intr.vector.interleave2"(%vec1, %vec2) : (vector<[2]xf16>, vector<[4]xf16>) -> vector<[8]xf16>
  return
}

// -----

func.func @vector_interleave2_bad_type1(%vec1: vector<[2]xf16>, %vec2 : vector<[2]xf16>) {
  // expected-error@+1 {{op failed to verify that result has twice as many elements as 'vec1'}}
  %0 = "llvm.intr.vector.interleave2"(%vec1, %vec2) : (vector<[2]xf16>, vector<[2]xf16>) -> vector<[8]xf16>
  return
}

// -----

/// result vector type is not scalable.

func.func @vector_interleave2_bad_type2(%vec1: vector<[2]xf16>, %vec2 : vector<[2]xf16>) {
  // expected-error@+1 {{op failed to verify that result has twice as many elements as 'vec1'}}
  %0 = "llvm.intr.vector.interleave2"(%vec1, %vec2) : (vector<[2]xf16>, vector<[2]xf16>) -> vector<4xf16>
  return
}

// -----


/// element type doesn't match.

func.func @vector_interleave2_bad_type3(%vec1: vector<[2]xf16>, %vec2 : vector<[2]xf16>) {
  // expected-error@+1 {{op failed to verify that result has twice as many elements as 'vec1'}}
  %0 = "llvm.intr.vector.interleave2"(%vec1, %vec2) : (vector<[2]xf16>, vector<[2]xf16>) -> vector<[4]xf32>
  return
}

// -----

func.func @invalid_bitcast_ptr_to_i64(%arg : !llvm.ptr) {
  // expected-error@+1 {{can only cast pointers from and to pointers}}
  %1 = llvm.bitcast %arg : !llvm.ptr to i64
}

// -----

func.func @invalid_bitcast_i64_to_ptr() {
  %0 = llvm.mlir.constant(2 : i64) : i64
  // expected-error@+1 {{can only cast pointers from and to pointers}}
  %1 = llvm.bitcast %0 : i64 to !llvm.ptr
}

// -----

func.func @invalid_bitcast_vec_to_ptr(%arg : vector<4x!llvm.ptr>) {
  // expected-error@+1 {{cannot cast vector of pointers to pointer}}
  %0 = llvm.bitcast %arg : vector<4x!llvm.ptr> to !llvm.ptr
}

// -----

func.func @invalid_bitcast_ptr_to_vec(%arg : !llvm.ptr) {
  // expected-error@+1 {{cannot cast pointer to vector of pointers}}
  %0 = llvm.bitcast %arg : !llvm.ptr to vector<4x!llvm.ptr>
}

// -----

func.func @invalid_bitcast_addr_cast(%arg : !llvm.ptr<1>) {
  // expected-error@+1 {{cannot cast pointers of different address spaces, use 'llvm.addrspacecast' instead}}
  %0 = llvm.bitcast %arg : !llvm.ptr<1> to !llvm.ptr
}

// -----

func.func @invalid_bitcast_addr_cast_vec(%arg : vector<4x!llvm.ptr<1>>) {
  // expected-error@+1 {{cannot cast pointers of different address spaces, use 'llvm.addrspacecast' instead}}
  %0 = llvm.bitcast %arg : vector<4x!llvm.ptr<1>> to vector<4x!llvm.ptr>
}

// -----

func.func @invalid_target_ext_alloca() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  // expected-error@+1 {{this target extension type cannot be used in alloca}}
  %1 = llvm.alloca %0 x !llvm.target<"no_alloca"> : (i64) -> !llvm.ptr
}

// -----

func.func @invalid_target_ext_load(%arg0 : !llvm.ptr) {
  // expected-error@+1 {{result #0 must be LLVM type with size, but got '!llvm.target<"no_load">'}}
  %0 = llvm.load %arg0 {alignment = 8 : i64} : !llvm.ptr -> !llvm.target<"no_load">
}

// -----

func.func @invalid_target_ext_atomic(%arg0 : !llvm.ptr) {
  // expected-error@+1 {{unsupported type '!llvm.target<"spirv.Event">' for atomic access}}
  %0 = llvm.load %arg0 atomic monotonic {alignment = 8 : i64} : !llvm.ptr -> !llvm.target<"spirv.Event">
}

// -----

func.func @invalid_target_ext_constant_unsupported() {
  // expected-error@+1 {{target extension type does not support zero-initializer}}
  %0 = llvm.mlir.zero : !llvm.target<"invalid_constant">
  llvm.return
}

// -----

func.func @invalid_target_ext_constant() {
  // expected-error@+1 {{does not support target extension type.}}
  %0 = llvm.mlir.constant(0 : index) : !llvm.target<"spirv.Event">
  llvm.return
}

// -----

llvm.comdat @__llvm_comdat {
  // expected-error@below {{only comdat selector symbols can appear in a comdat region}}
  llvm.return
}

// -----

llvm.mlir.global @not_comdat(0 : i32) : i32
// expected-error@below {{expected comdat symbol}}
llvm.mlir.global @invalid_global_comdat(0 : i32) comdat(@not_comdat) : i32

// -----

// expected-error@below {{expected comdat symbol}}
llvm.func @invalid_func_comdat() comdat(@foo) {
  llvm.return
}

// -----

func.func @invalid_zext_target_size_equal(%arg: i32)  {
  // expected-error@+1 {{integer width of the output type is smaller or equal to the integer width of the input type}}
  %0 = llvm.zext %arg : i32 to i32
}

// -----

func.func @invalid_zext_target_size(%arg: i32)  {
  // expected-error@+1 {{integer width of the output type is smaller or equal to the integer width of the input type}}
  %0 = llvm.zext %arg : i32 to i16
}

// -----

func.func @invalid_zext_target_size_vector(%arg: vector<1xi32>)  {
  // expected-error@+1 {{integer width of the output type is smaller or equal to the integer width of the input type}}
  %0 = llvm.zext %arg : vector<1xi32> to vector<1xi16>
}

// -----

func.func @invalid_zext_target_shape(%arg: vector<1xi32>)  {
  // expected-error@+1 {{input and output vectors are of incompatible shape}}
  %0 = llvm.zext %arg : vector<1xi32> to vector<2xi64>
}

// -----

func.func @invalid_zext_target_type(%arg: i32)  {
  // expected-error@+1 {{input type is an integer but output type is a vector}}
  %0 = llvm.zext %arg : i32 to vector<1xi64>
}

// -----

func.func @invalid_zext_target_type_two(%arg: vector<1xi32>)  {
  // expected-error@+1 {{input type is a vector but output type is an integer}}
  %0 = llvm.zext %arg : vector<1xi32> to i64
}

// -----

llvm.func @non_variadic(%arg: i32)

llvm.func @invalid_var_callee_type(%arg: i32)  {
  // expected-error@below {{expected var_callee_type to be a variadic function type}}
  llvm.call @non_variadic(%arg) vararg(!llvm.func<void (i32)>) : (i32) -> ()
  llvm.return
}

// -----

llvm.func @variadic(%arg: i32, ...)

llvm.func @invalid_var_callee_type_num_parameters(%arg: i32)  {
  // expected-error@below {{expected var_callee_type to have at most 1 parameters}}
  llvm.call @variadic(%arg) vararg(!llvm.func<void (i32, i64, ...)>) : (i32) -> ()
  llvm.return
}

// -----

llvm.func @invalid_var_callee_type_num_parameters_indirect(%callee : !llvm.ptr, %arg: i32)  {
  // expected-error@below {{expected var_callee_type to have at most 1 parameters}}
  llvm.call %callee(%arg) vararg(!llvm.func<void (i32, i64, ...)>) : !llvm.ptr, (i32) -> ()
  llvm.return
}

// -----

llvm.func @variadic(%arg: i32, ...)

llvm.func @invalid_var_callee_type_parameter_type_mismatch(%arg: i32)  {
  // expected-error@below {{var_callee_type parameter type mismatch: 'i64' != 'i32'}}
  llvm.call @variadic(%arg) vararg(!llvm.func<void (i64, ...)>) : (i32) -> ()
  llvm.return
}

// -----

llvm.func @invalid_var_callee_type_parameter_type_mismatch_indirect(%callee : !llvm.ptr, %arg: i32)  {
  // expected-error@below {{var_callee_type parameter type mismatch: 'i64' != 'i32'}}
  llvm.call %callee(%arg) vararg(!llvm.func<void (i64, ...)>) : !llvm.ptr, (i32) -> ()
  llvm.return
}

// -----

llvm.func @variadic(%arg: i32, ...)

llvm.func @invalid_var_callee_type_non_void(%arg: i32)  {
  // expected-error@below {{expected var_callee_type to return void}}
  llvm.call @variadic(%arg) vararg(!llvm.func<i8 (i32, ...)>) : (i32) -> ()
  llvm.return
}

// -----

llvm.func @variadic(%arg: i32, ...) -> i32

llvm.func @invalid_var_callee_type_return_type_mismatch(%arg: i32)  {
  // expected-error@below {{var_callee_type return type mismatch: 'i8' != 'i32'}}
  %0 = llvm.call @variadic(%arg) vararg(!llvm.func<i8 (i32, ...)>) : (i32) -> (i32)
  llvm.return
}

// -----

llvm.func @non_variadic(%arg: i32)

llvm.func @invalid_var_callee_type(%arg: i32)  {
  // expected-error@below {{expected var_callee_type to be a variadic function type}}
  llvm.invoke @non_variadic(%arg) to ^bb2 unwind ^bb1 vararg(!llvm.func<void (i32)>) : (i32) -> ()
^bb1:
  llvm.return
^bb2:
  llvm.return
}

// -----

llvm.func @variadic(%arg: i32, ...)

llvm.func @invalid_var_callee_type_num_parameters(%arg: i32)  {
  // expected-error@below {{expected var_callee_type to have at most 1 parameters}}
  llvm.invoke @variadic(%arg) to ^bb2 unwind ^bb1 vararg(!llvm.func<void (i32, i64, ...)>) : (i32) -> ()
^bb1:
  llvm.return
^bb2:
  llvm.return
}

// -----

llvm.func @invalid_var_callee_type_num_parameters_indirect(%callee : !llvm.ptr, %arg: i32)  {
  // expected-error@below {{expected var_callee_type to have at most 1 parameters}}
  llvm.invoke %callee(%arg) to ^bb2 unwind ^bb1 vararg(!llvm.func<void (i32, i64, ...)>) : !llvm.ptr, (i32) -> ()
^bb1:
  llvm.return
^bb2:
  llvm.return
}

// -----

llvm.func @variadic(%arg: i32, ...)

llvm.func @invalid_var_callee_type_parameter_type_mismatch(%arg: i32)  {
  // expected-error@below {{var_callee_type parameter type mismatch: 'i64' != 'i32'}}
  llvm.invoke @variadic(%arg) to ^bb2 unwind ^bb1 vararg(!llvm.func<void (i64, ...)>) : (i32) -> ()
^bb1:
  llvm.return
^bb2:
  llvm.return
}

// -----

llvm.func @invalid_var_callee_type_parameter_type_mismatch_indirect(%callee : !llvm.ptr, %arg: i32)  {
  // expected-error@below {{var_callee_type parameter type mismatch: 'i64' != 'i32'}}
  llvm.invoke %callee(%arg) to ^bb2 unwind ^bb1 vararg(!llvm.func<void (i64, ...)>) : !llvm.ptr, (i32) -> ()
^bb1:
  llvm.return
^bb2:
  llvm.return
}

// -----

llvm.func @variadic(%arg: i32, ...)

llvm.func @invalid_var_callee_type_non_void(%arg: i32)  {
  // expected-error@below {{expected var_callee_type to return void}}
  llvm.invoke @variadic(%arg) to ^bb2 unwind ^bb1 vararg(!llvm.func<i8 (i32, ...)>) : (i32) -> ()
^bb1:
  llvm.return
^bb2:
  llvm.return
}

// -----

llvm.func @variadic(%arg: i32, ...) -> i32

llvm.func @invalid_var_callee_type_return_type_mismatch(%arg: i32)  {
  // expected-error@below {{var_callee_type return type mismatch: 'i8' != 'i32'}}
  %0 = llvm.invoke @variadic(%arg) to ^bb2 unwind ^bb1 vararg(!llvm.func<i8 (i32, ...)>) : (i32) -> (i32)
^bb1:
  llvm.return
^bb2:
  llvm.return
}

// -----

llvm.func @variadic(...)

llvm.func @invalid_variadic_call(%arg: i32)  {
  // expected-error@+1 {{missing var_callee_type attribute for vararg call}}
  "llvm.call"(%arg) <{callee = @variadic}> {operandSegmentSizes = array<i32: 1, 0>, op_bundle_sizes = array<i32>} : (i32) -> ()
  llvm.return
}

// -----

llvm.func @variadic(...)

llvm.func @invalid_variadic_call(%arg: i32)  {
  // expected-error@+1 {{missing var_callee_type attribute for vararg call}}
  "llvm.call"(%arg) <{callee = @variadic}> {operandSegmentSizes = array<i32: 1, 0>, op_bundle_sizes = array<i32>} : (i32) -> ()
  llvm.return
}

// -----

llvm.func @foo(%arg: !llvm.ptr) {
  // expected-error@+1 {{type '!llvm.ptr' cannot be indexed (index #1)}}
  %0 = llvm.getelementptr %arg[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
  llvm.return
}

// -----

func.func @tma_load(%tmaDescriptor: !llvm.ptr, %dest : !llvm.ptr<3>, %barrier: !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %crd3: i32, %off0: i16, %off1: i16, %ctamask : i16, %cacheHint : i64, %p : i1) {
  // expected-error@+1 {{to use im2col mode, the tensor has to be at least 3-dimensional}}
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tmaDescriptor,  %barrier, box[%crd0,%crd1] im2col[%off0] multicast_mask = %ctamask l2_cache_hint = %cacheHint : !llvm.ptr<3>, !llvm.ptr
  return
}
// -----

func.func @tma_load(%tmaDescriptor: !llvm.ptr, %dest : !llvm.ptr<3>, %barrier: !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %crd3: i32, %off0: i16, %off1: i16, %ctamask : i16, %cacheHint : i64, %p : i1) {
  // expected-error@+1 {{im2col offsets must be 2 less than number of coordinates}}
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tmaDescriptor,  %barrier, box[%crd0,%crd1,%crd2,%crd3] im2col[%off0] multicast_mask = %ctamask l2_cache_hint = %cacheHint : !llvm.ptr<3>, !llvm.ptr
  return
}

// -----

func.func @tma_load(%tmaDescriptor: !llvm.ptr, %dest : !llvm.ptr<3>, %barrier: !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %crd3: i32, %off0: i16, %off1: i16, %ctamask : i16, %cacheHint : i64, %p : i1) {
  // expected-error@+1 {{expects coordinates between 1 to 5 dimension}}
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tmaDescriptor,  %barrier, box[]: !llvm.ptr<3>, !llvm.ptr
  return
}

// -----

func.func @tma_load(%tmaDescriptor: !llvm.ptr, %dest : !llvm.ptr<3>, %barrier: !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %crd3: i32, %off0: i16, %off1: i16, %ctamask : i16, %cacheHint : i64, %p : i1) {
  // expected-error@+1 {{expects coordinates between 1 to 5 dimension}}
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tmaDescriptor,  %barrier, box[%crd0,%crd1,%crd2,%crd3,%crd0,%crd1,%crd2,%crd3]: !llvm.ptr<3>, !llvm.ptr
  return
}

// -----

// expected-error @below {{no_inline and always_inline attributes are incompatible}}
llvm.func @alwaysinline_noinline() attributes { always_inline, no_inline } {
  llvm.return
}

// -----

// expected-error @below {{'llvm.func' op with optimize_none must also be no_inline}}
llvm.func @optnone_requires_noinline() attributes { optimize_none } {
  llvm.return
}

// -----

llvm.func @foo()
llvm.func @wrong_number_of_bundle_types() {
  %0 = llvm.mlir.constant(0 : i32) : i32
  // expected-error@+1 {{expected 1 types for operand bundle operands for operand bundle #0, but actually got 2}}
  llvm.call @foo() ["tag"(%0 : i32, i32)] : () -> ()
  llvm.return
}

// -----

llvm.func @wrong_number_of_bundle_types_intrin(%arg0: i32) -> i32 {
  %0 = llvm.mlir.constant(0 : i32) : i32
  // expected-error@+1 {{expected 1 types for operand bundle operands for operand bundle #0, but actually got 2}}
  %1 = llvm.call_intrinsic "llvm.riscv.sha256sig0"(%arg0) ["tag"(%0 : i32, i32)] : (i32 {llvm.signext}) -> (i32)
  llvm.return %1 : i32
}

// -----

llvm.func @foo()
llvm.func @wrong_number_of_bundle_tags() {
  %0 = llvm.mlir.constant(0 : i32) : i32
  %1 = llvm.mlir.constant(1 : i32) : i32
  // expected-error@+1 {{expected 2 operand bundle tags, but actually got 1}}
  "llvm.call"(%0, %1) <{ op_bundle_tags = ["tag"] }> {
    callee = @foo,
    operandSegmentSizes = array<i32: 0, 2>,
    op_bundle_sizes = array<i32: 1, 1>
  } : (i32, i32) -> ()
  llvm.return
}

// -----

llvm.mlir.global external @x(42 : i32) : i32

// expected-error@+1 {{expects type to be a valid element type for an LLVM global alias}}
llvm.mlir.alias external @y : !llvm.label {
  %0 = llvm.mlir.addressof @x : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// -----

llvm.mlir.global external @x(42 : i32) : i32

// expected-error@+1 {{linkage not supported in aliases, available options}}
llvm.mlir.alias appending @y2 : i32 {
  %0 = llvm.mlir.addressof @x : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// -----

// expected-error@+1 {{initializer region must always return a pointer}}
llvm.mlir.alias external @y3 : i32 {
  %c = llvm.mlir.constant(42 : i64) : i64
  llvm.return %c : i64
}

// -----

llvm.mlir.global external @x(42 : i32) : i32

llvm.mlir.alias external @y4 : i32 {
  %0 = llvm.mlir.addressof @x : !llvm.ptr
  // expected-error@+1 {{ops with side effects are not allowed in alias initializers}}
  %2 = llvm.load %0 : !llvm.ptr -> i32
  llvm.return %0 : !llvm.ptr
}

// -----

llvm.mlir.global external @x(42 : i32) : i32

llvm.mlir.alias external @y5 : i32 {
  // expected-error@+1 {{pointer address space must match address space}}
  %0 = llvm.mlir.addressof @x : !llvm.ptr<4>
  llvm.return %0 : !llvm.ptr<4>
}

// -----

module {
  llvm.func @foo()

  // expected-error@below {{only integer and string values are currently supported}}
  llvm.module_flags [#llvm.mlir.module_flag<error, "yolo", @foo>]
}

// -----

module {
  // expected-error@below {{'CG Profile' key expects an array of '#llvm.cgprofile_entry'}}
  llvm.module_flags [#llvm.mlir.module_flag<append, "CG Profile", [
    "yo"
  ]>]
}

// -----

module {
  // expected-error@below {{'CG Profile' key expects an array of '#llvm.cgprofile_entry'}}
  llvm.module_flags [#llvm.mlir.module_flag<append, "CG Profile", 3 : i64>]
}

// -----

module {
  // expected-error@below {{'ProfileSummary' key expects a '#llvm.profile_summary' attribute}}
  llvm.module_flags [#llvm.mlir.module_flag<append, "ProfileSummary", 3 : i64>]
}

// -----

llvm.module_flags [#llvm.mlir.module_flag<error, "ProfileSummary",
     // expected-error@below {{expected one of [SampleProfile, InstrProf, CSInstrProf] for LLVM ProfileSummary format kinds, got: YoloFmt}}
     #llvm.profile_summary<format = "YoloFmt", total_count = 263646, max_count = 86427,
     // expected-error@above {{failed to parse ModuleFlagProfileSummaryAttr parameter 'format' which is to be a `ProfileSummaryFormatKind`}}
       max_internal_count = 86427, max_function_count = 4691,
       num_counts = 3712, num_functions = 796,
       is_partial_profile = 0,
       partial_profile_ratio = 0.000000e+00 : f64,
       detailed_summary =
         <cut_off = 10000, min_count = 86427, num_counts = 1>,
         <cut_off = 100000, min_count = 86427, num_counts = 1>
      // expected-error@below {{failed to parse ModuleFlagAttr parameter}}
>>]

// -----

llvm.func @t0() -> !llvm.ptr {
  %0 = llvm.blockaddress <function = @t0, tag = <id = 1>> : !llvm.ptr
  llvm.blocktag <id = 1>
  llvm.br ^bb1
^bb1:
  // expected-error@+1 {{duplicate block tag '1' in the same function}}
  llvm.blocktag <id = 1>
  llvm.return %0 : !llvm.ptr
}

// -----

llvm.func @t1() -> !llvm.ptr {
  // expected-error@+1 {{expects an existing block label target in the referenced function}}
  %0 = llvm.blockaddress <function = @t1, tag = <id = 1>> : !llvm.ptr
  llvm.br ^bb1
^bb1:
  llvm.return %0 : !llvm.ptr
}

// -----

llvm.func @gep_inbounds_flag_usage(%ptr: !llvm.ptr, %idx: i64) {
  // expected-error@+1 {{'inbounds_flag' cannot be used directly}}
  llvm.getelementptr inbounds_flag %ptr[%idx, 0, %idx] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<(array<10 x f32>)>
  llvm.return
}

// -----

llvm.mlir.global @bad_struct_array_init_size() : !llvm.array<2x!llvm.struct<(i32, f32)>> {
  // expected-error@below {{'llvm.mlir.constant' op array attribute size does not match array type size in dimension 0: 1 vs. 2}}
  %0 = llvm.mlir.constant([[42 : i32, 1.000000e+00 : f32]]) : !llvm.array<2x!llvm.struct<(i32, f32)>>
  llvm.return %0 : !llvm.array<2x!llvm.struct<(i32, f32)>>
}

// -----

llvm.mlir.global @bad_struct_array_init_nesting() : !llvm.array<1x!llvm.array<1x!llvm.array<1x!llvm.struct<(i32)>>>> {
  // expected-error@below {{'llvm.mlir.constant' op nested attribute for sub-array in dimension 1 at index 0 must be a zero, or undef, or array attribute}}
  %0 = llvm.mlir.constant([[1 : i32]]) : !llvm.array<1x!llvm.array<1x!llvm.array<1x!llvm.struct<(i32)>>>>
  llvm.return %0 : !llvm.array<1x!llvm.array<1x!llvm.array<1x!llvm.struct<(i32)>>>>
}

// -----

llvm.mlir.global @bad_struct_array_init_elements() : !llvm.array<1x!llvm.struct<(i32, f32)>> {
  // expected-error@below {{'llvm.mlir.constant' op nested array attribute size for struct element at index 0 must match struct size: 1 vs. 2}}
  %0 = llvm.mlir.constant([[1 : i32]]) : !llvm.array<1x!llvm.struct<(i32, f32)>>
  llvm.return %0 : !llvm.array<1x!llvm.struct<(i32, f32)>>
}

// -----

llvm.mlir.global internal constant @bad_array_attr_simple_type() : !llvm.array<2 x f64> {
  // expected-error@below {{'llvm.mlir.constant' op for array with an array attribute must have a struct element type}}
  %0 = llvm.mlir.constant([2.5, 7.4]) : !llvm.array<2 x f64>
  llvm.return %0 : !llvm.array<2 x f64>
}

// -----

llvm.func @inlineAsmMustTail(%arg0: i32, %arg1 : !llvm.ptr) {
  // expected-error@+1 {{op tail call kind 'musttail' is not supported}}
  %8 = llvm.inline_asm tail_call_kind = <musttail> "foo", "=r,=r,r" %arg0 : (i32) -> !llvm.struct<(i8, i8)>
  llvm.return
}

// -----

llvm.func @invalid_xevm_prefetch(%arg0: !llvm.ptr) {
  // expected-error@+1 {{op operand #0 must be LLVM pointer in address space 1 or LLVM pointer in address space 4}}
  xevm.prefetch %arg0 <{cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>}> : (!llvm.ptr)
  llvm.return
}

// -----

llvm.func @invalid_xevm_mma(%loaded_c_casted: vector<4xf32>, %loaded_a: vector<8xi16>, %loaded_b_casted: vector<8xi32>) -> vector<8xf32> {
  // expected-error@+1 {{op type of C operand must match result type}}
  %c_result = xevm.mma %loaded_a, %loaded_b_casted, %loaded_c_casted {shape = <m = 8, n = 16, k = 16>, types = <d = f32, a = f16, b = f16, c = f32>} : (vector<8xi16>, vector<8xi32>, vector<4xf32>) -> vector<8xf32>
  llvm.return %c_result : vector<8xf32>
}

// -----

llvm.func @invalid_xevm_matrix_1(%c: !llvm.ptr<1>, %base_width_c: i32, %base_height_c: i32, %base_pitch_c: i32, %x: i32, %y: i32, %c_result_casted: vector<8xi32>) {
  // expected-error@+1 {{op expecting tile_width to be between 1 and 8}}
  xevm.blockstore2d %c, %base_width_c, %base_height_c, %base_pitch_c, %x, %y, %c_result_casted <{elem_size_in_bits=64 : i32, tile_width=16 : i32, tile_height=8 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)
  llvm.return
}

// -----

llvm.func @invalid_xevm_matrix_2(%c: !llvm.ptr<1>, %base_width_c: i32, %base_height_c: i32, %base_pitch_c: i32, %x: i32, %y: i32, %c_result_casted: vector<8xi32>) {
  // expected-error@+1 {{op expecting elem_size_in_bits to be 8, 16, 32, or 64}}
  xevm.blockstore2d %c, %base_width_c, %base_height_c, %base_pitch_c, %x, %y, %c_result_casted <{elem_size_in_bits=18 : i32, tile_width=16 : i32, tile_height=8 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)
  llvm.return
}

// -----

llvm.func @invalid_xevm_matrix_3(%a: !llvm.ptr<1>, %base_width_a: i32, %base_height_a: i32, %base_pitch_a: i32, %x: i32, %y: i32) -> vector<8xi16> {
  // expected-error@+1 {{op result size of 128 bits does not match the expected size of 208 bits}}
  %loaded_a = xevm.blockload2d %a, %base_width_a, %base_height_a, %base_pitch_a, %x, %y <{elem_size_in_bits=16 : i32, tile_width=26 : i32, tile_height=8 : i32, v_blocks=1 : i32, transpose=false, pack_register=false, cache_control=#xevm.load_cache_control<L1uc_L2uc_L3uc>}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi16>
  llvm.return %loaded_a : vector<8xi16>
}

// -----

llvm.func external @resolve_foo() -> !llvm.ptr attributes {dso_local}
// expected-error@+1 {{'llvm.mlir.ifunc' op resolver must be a definition}}
llvm.mlir.ifunc external @foo : !llvm.func<void (ptr, i32)>, !llvm.ptr @resolve_foo {dso_local}

// -----

llvm.mlir.global external @resolve_foo() : !llvm.ptr
// expected-error@+1 {{'llvm.mlir.ifunc' op must have a function resolver}}
llvm.mlir.ifunc external @foo : !llvm.func<void (ptr, i32)>, !llvm.ptr @resolve_foo {dso_local}

// -----

llvm.func external @resolve_foo() -> !llvm.ptr
// expected-error@+1 {{'llvm.mlir.ifunc' op 'common' linkage not supported in ifuncs}}
llvm.mlir.ifunc common @foo : !llvm.func<void (ptr, i32)>, !llvm.ptr @resolve_foo {dso_local}

// -----

llvm.mlir.global external @resolve_foo() : !llvm.ptr
llvm.mlir.alias external @alias_resolver : !llvm.ptr {
  %0 = llvm.mlir.addressof @resolve_foo : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}
// expected-error@+1 {{'llvm.mlir.ifunc' op must have a function resolver}}
llvm.mlir.ifunc external @foo : !llvm.func<void (ptr, i32)>, !llvm.ptr @alias_resolver {dso_local}

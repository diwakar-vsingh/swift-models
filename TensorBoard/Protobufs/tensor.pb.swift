// DO NOT EDIT.
// swift-format-ignore-file
//
// Generated by the Swift generator plugin for the protocol buffer compiler.
// Source: tensorboardX/proto/tensor.proto
//
// For information on using the generated types, please see the documentation:
//   https://github.com/apple/swift-protobuf/

import Foundation
import SwiftProtobuf

// If the compiler emits an error on this type, it is because this file
// was generated by a version of the `protoc` Swift plug-in that is
// incompatible with the version of SwiftProtobuf to which you are linking.
// Please ensure that you are building against the same version of the API
// that was used to generate this file.
fileprivate struct _GeneratedWithProtocGenSwiftVersion: SwiftProtobuf.ProtobufAPIVersionCheck {
  struct _2: SwiftProtobuf.ProtobufAPIVersion_2 {}
  typealias Version = _2
}

/// Protocol buffer representing a tensor.
struct TensorboardX_TensorProto {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  var dtype: TensorboardX_DataType = .dtInvalid

  /// Shape of the tensor.  TODO(touts): sort out the 0-rank issues.
  var tensorShape: TensorboardX_TensorShapeProto {
    get {return _tensorShape ?? TensorboardX_TensorShapeProto()}
    set {_tensorShape = newValue}
  }
  /// Returns true if `tensorShape` has been explicitly set.
  var hasTensorShape: Bool {return self._tensorShape != nil}
  /// Clears the value of `tensorShape`. Subsequent reads from it will return its default value.
  mutating func clearTensorShape() {self._tensorShape = nil}

  /// Version number.
  ///
  /// In version 0, if the "repeated xxx" representations contain only one
  /// element, that element is repeated to fill the shape.  This makes it easy
  /// to represent a constant Tensor with a single value.
  var versionNumber: Int32 = 0

  /// Serialized raw tensor content from either Tensor::AsProtoTensorContent or
  /// memcpy in tensorflow::grpc::EncodeTensorToByteBuffer. This representation
  /// can be used for all tensor types. The purpose of this representation is to
  /// reduce serialization overhead during RPC call by avoiding serialization of
  /// many repeated small items.
  var tensorContent: Data = Data()

  /// DT_HALF. Note that since protobuf has no int16 type, we'll have some
  /// pointless zero padding for each value here.
  var halfVal: [Int32] = []

  /// DT_FLOAT.
  var floatVal: [Float] = []

  /// DT_DOUBLE.
  var doubleVal: [Double] = []

  /// DT_INT32, DT_INT16, DT_INT8, DT_UINT8.
  var intVal: [Int32] = []

  /// DT_STRING
  var stringVal: [Data] = []

  /// DT_COMPLEX64. scomplex_val(2*i) and scomplex_val(2*i+1) are real
  /// and imaginary parts of i-th single precision complex.
  var scomplexVal: [Float] = []

  /// DT_INT64
  var int64Val: [Int64] = []

  /// DT_BOOL
  var boolVal: [Bool] = []

  /// DT_COMPLEX128. dcomplex_val(2*i) and dcomplex_val(2*i+1) are real
  /// and imaginary parts of i-th double precision complex.
  var dcomplexVal: [Double] = []

  /// DT_RESOURCE
  var resourceHandleVal: [TensorboardX_ResourceHandleProto] = []

  var unknownFields = SwiftProtobuf.UnknownStorage()

  init() {}

  fileprivate var _tensorShape: TensorboardX_TensorShapeProto? = nil
}

// MARK: - Code below here is support for the SwiftProtobuf runtime.

fileprivate let _protobuf_package = "tensorboardX"

extension TensorboardX_TensorProto: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  static let protoMessageName: String = _protobuf_package + ".TensorProto"
  static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "dtype"),
    2: .standard(proto: "tensor_shape"),
    3: .standard(proto: "version_number"),
    4: .standard(proto: "tensor_content"),
    13: .standard(proto: "half_val"),
    5: .standard(proto: "float_val"),
    6: .standard(proto: "double_val"),
    7: .standard(proto: "int_val"),
    8: .standard(proto: "string_val"),
    9: .standard(proto: "scomplex_val"),
    10: .standard(proto: "int64_val"),
    11: .standard(proto: "bool_val"),
    12: .standard(proto: "dcomplex_val"),
    14: .standard(proto: "resource_handle_val"),
  ]

  mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeSingularEnumField(value: &self.dtype) }()
      case 2: try { try decoder.decodeSingularMessageField(value: &self._tensorShape) }()
      case 3: try { try decoder.decodeSingularInt32Field(value: &self.versionNumber) }()
      case 4: try { try decoder.decodeSingularBytesField(value: &self.tensorContent) }()
      case 5: try { try decoder.decodeRepeatedFloatField(value: &self.floatVal) }()
      case 6: try { try decoder.decodeRepeatedDoubleField(value: &self.doubleVal) }()
      case 7: try { try decoder.decodeRepeatedInt32Field(value: &self.intVal) }()
      case 8: try { try decoder.decodeRepeatedBytesField(value: &self.stringVal) }()
      case 9: try { try decoder.decodeRepeatedFloatField(value: &self.scomplexVal) }()
      case 10: try { try decoder.decodeRepeatedInt64Field(value: &self.int64Val) }()
      case 11: try { try decoder.decodeRepeatedBoolField(value: &self.boolVal) }()
      case 12: try { try decoder.decodeRepeatedDoubleField(value: &self.dcomplexVal) }()
      case 13: try { try decoder.decodeRepeatedInt32Field(value: &self.halfVal) }()
      case 14: try { try decoder.decodeRepeatedMessageField(value: &self.resourceHandleVal) }()
      default: break
      }
    }
  }

  func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if self.dtype != .dtInvalid {
      try visitor.visitSingularEnumField(value: self.dtype, fieldNumber: 1)
    }
    if let v = self._tensorShape {
      try visitor.visitSingularMessageField(value: v, fieldNumber: 2)
    }
    if self.versionNumber != 0 {
      try visitor.visitSingularInt32Field(value: self.versionNumber, fieldNumber: 3)
    }
    if !self.tensorContent.isEmpty {
      try visitor.visitSingularBytesField(value: self.tensorContent, fieldNumber: 4)
    }
    if !self.floatVal.isEmpty {
      try visitor.visitPackedFloatField(value: self.floatVal, fieldNumber: 5)
    }
    if !self.doubleVal.isEmpty {
      try visitor.visitPackedDoubleField(value: self.doubleVal, fieldNumber: 6)
    }
    if !self.intVal.isEmpty {
      try visitor.visitPackedInt32Field(value: self.intVal, fieldNumber: 7)
    }
    if !self.stringVal.isEmpty {
      try visitor.visitRepeatedBytesField(value: self.stringVal, fieldNumber: 8)
    }
    if !self.scomplexVal.isEmpty {
      try visitor.visitPackedFloatField(value: self.scomplexVal, fieldNumber: 9)
    }
    if !self.int64Val.isEmpty {
      try visitor.visitPackedInt64Field(value: self.int64Val, fieldNumber: 10)
    }
    if !self.boolVal.isEmpty {
      try visitor.visitPackedBoolField(value: self.boolVal, fieldNumber: 11)
    }
    if !self.dcomplexVal.isEmpty {
      try visitor.visitPackedDoubleField(value: self.dcomplexVal, fieldNumber: 12)
    }
    if !self.halfVal.isEmpty {
      try visitor.visitPackedInt32Field(value: self.halfVal, fieldNumber: 13)
    }
    if !self.resourceHandleVal.isEmpty {
      try visitor.visitRepeatedMessageField(value: self.resourceHandleVal, fieldNumber: 14)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  static func ==(lhs: TensorboardX_TensorProto, rhs: TensorboardX_TensorProto) -> Bool {
    if lhs.dtype != rhs.dtype {return false}
    if lhs._tensorShape != rhs._tensorShape {return false}
    if lhs.versionNumber != rhs.versionNumber {return false}
    if lhs.tensorContent != rhs.tensorContent {return false}
    if lhs.halfVal != rhs.halfVal {return false}
    if lhs.floatVal != rhs.floatVal {return false}
    if lhs.doubleVal != rhs.doubleVal {return false}
    if lhs.intVal != rhs.intVal {return false}
    if lhs.stringVal != rhs.stringVal {return false}
    if lhs.scomplexVal != rhs.scomplexVal {return false}
    if lhs.int64Val != rhs.int64Val {return false}
    if lhs.boolVal != rhs.boolVal {return false}
    if lhs.dcomplexVal != rhs.dcomplexVal {return false}
    if lhs.resourceHandleVal != rhs.resourceHandleVal {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}
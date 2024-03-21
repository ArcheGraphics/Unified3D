//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "MiniVec.h"

namespace u3d {
namespace metal {
class Indexer;

class IndexerIterator;

// Maximum number of dimensions of TensorRef.
static constant int64_t MAX_DIMS = 10;

// Maximum number of inputs of an op.
// MAX_INPUTS shall be >= MAX_DIMS to support advanced indexing.
static constant int64_t MAX_INPUTS = 10;

// Maximum number of outputs of an op. This number can be increased when
// necessary.
static constant int64_t MAX_OUTPUTS = 2;

template <int NARGS, typename index_t = uint32_t>
struct OffsetCalculator {
    MiniVec<index_t, NARGS> get(index_t linear_idx) thread const {
        MiniVec<index_t, NARGS> offsets;
        for (int arg = 0; arg < NARGS; arg++) {
            offsets[arg] = 0;
        }
        
        for (int dim = 0; dim < MAX_DIMS; ++dim) {
            if (dim == dims_) {
                break;
            }
            index_t mod = linear_idx % sizes_[dim];
            linear_idx = linear_idx / sizes_[dim];
            
            for (int arg = 0; arg < NARGS; arg++) {
                offsets[arg] += mod * strides_[dim][arg];
            }
        }
        return offsets;
    }
    
    int dims_;
    index_t sizes_[MAX_DIMS];
    index_t strides_[MAX_DIMS][NARGS];
};

/// A minimalistic class that reference a Tensor.
struct TensorRef {
    device char* data_ptr_;
    int64_t dtype_byte_size_ = 0;
    int64_t shape_[MAX_DIMS];
    int64_t byte_strides_[MAX_DIMS];
};

/// Indexer to one Tensor
///
/// Example usage:
///
/// ```cpp
/// // Create a float Tensor and set all elements to 100.
/// std::vector<float> vals{0, 1, 2, 3, 4};
/// Tensor a(vals, SizeVector{5}, core::Float32);
/// TensorIterator iter(a);
/// for (int64_t i = 0; i < iter.NumWorkloads(); ++i) {
///     *static_cast<float*>(iter.GetPtr(i)) = 100.f;
/// }
/// ```
class TensorIterator {
public:
    int64_t NumWorkloads() thread const {
        int64_t num_workloads = 1;
        for (int64_t i = 0; i < ndims_; ++i) {
            num_workloads *= input_.shape_[i];
        }
        return num_workloads;
    }
    
    device char* GetPtr(int64_t workload_idx) thread const {
        if (workload_idx < 0 || workload_idx >= NumWorkloads()) {
            return nullptr;
        }
        int64_t offset = 0;
        workload_idx = workload_idx * input_.dtype_byte_size_;
        for (int64_t i = 0; i < ndims_; ++i) {
            offset += workload_idx / input_.byte_strides_[i] *
            input_.byte_strides_[i];
            workload_idx = workload_idx % input_.byte_strides_[i];
        }
        return input_.data_ptr_ + offset;
    }
    
protected:
    TensorRef input_;
    int64_t ndims_;
};

/// Indexing engine for elementwise ops with broadcasting support.
///
/// Fancy indexing is supported by restriding input tensor and treating the
/// operation as elementwise op.
///
/// After constructing Indexer on the host, the indexing methods can be
/// used from both host and device.
class Indexer {
public:
    /// Get input Tensor data pointer based on \p workload_idx.
    ///
    /// \param input_idx Input tensor index.
    /// \param workload_idx The index of the compute workload, similar to
    /// thread_id, if a thread only processes one workload.
    device char* GetInputPtr(int64_t input_idx, int64_t workload_idx) device const {
        if (input_idx < 0 || input_idx >= num_inputs_) {
            return nullptr;
        }
        return GetWorkloadDataPtr(inputs_[input_idx],
                                  inputs_contiguous_[input_idx], workload_idx);
    }
    
    /// Get input Tensor data pointer based on \p workload_idx.
    ///
    /// \param input_idx Input tensor index.
    /// \param workload_idx The index of the compute workload, similar to
    /// thread_id, if a thread only processes one workload.
    ///
    /// Note: Assumes that sizeof(T) matches the input's dtype size, but does
    /// not check this constraint for performance reasons.
    template <typename T>
    device T* GetInputPtr(int64_t input_idx, int64_t workload_idx) device const {
        if (input_idx < 0 || input_idx >= num_inputs_) {
            return nullptr;
        }
        return GetWorkloadDataPtr<T>(inputs_[input_idx],
                                     inputs_contiguous_[input_idx],
                                     workload_idx);
    }
    
    /// Get output Tensor data pointer based on \p workload_idx.
    ///
    /// \param workload_idx The index of the compute workload, similar to
    /// thread_id, if a thread only processes one workload.
    device char* GetOutputPtr(int64_t workload_idx) device const {
        return GetWorkloadDataPtr(outputs_[0], outputs_contiguous_[0],
                                  workload_idx);
    }
    
    /// Get output Tensor data pointer based on \p workload_idx.
    ///
    /// \param workload_idx The index of the compute workload, similar to
    /// thread_id, if a thread only processes one workload.
    ///
    /// Note: Assumes that sizeof(T) matches the output's dtype size, but does
    /// not check this constraint for performance reasons.
    template <typename T>
    device T* GetOutputPtr(int64_t workload_idx) device const {
        return GetWorkloadDataPtr<T>(outputs_[0], outputs_contiguous_[0],
                                     workload_idx);
    }
    
    /// Get output Tensor data pointer based on \p workload_idx.
    ///
    /// \param output_idx Output tensor index.
    /// \param workload_idx The index of the compute workload, similar to
    /// thread_id, if a thread only processes one workload.
    device char* GetOutputPtr(int64_t output_idx, int64_t workload_idx) device const {
        return GetWorkloadDataPtr(outputs_[output_idx],
                                  outputs_contiguous_[output_idx],
                                  workload_idx);
    }
    
    /// Get output Tensor data pointer based on \p workload_idx.
    ///
    /// \param output_idx Output tensor index.
    /// \param workload_idx The index of the compute workload, similar to
    /// thread_id, if a thread only processes one workload.
    template <typename T>
    device T* GetOutputPtr(int64_t output_idx, int64_t workload_idx) device const {
        return GetWorkloadDataPtr<T>(outputs_[output_idx],
                                     outputs_contiguous_[output_idx],
                                     workload_idx);
    }
    
protected:
    /// Get data pointer from a TensorRef with \p workload_idx.
    /// Note: can be optimized by computing all input ptrs and output ptr
    /// together.
    device char* GetWorkloadDataPtr(const device TensorRef& tr,
                                    bool tr_contiguous,
                                    int64_t workload_idx) device const {
        // For 0-sized input reduction op, the output Tensor
        // workload_idx == 1 > NumWorkloads() == 0.
        if (workload_idx < 0) {
            return nullptr;
        }
        if (tr_contiguous) {
            return static_cast<device char*>(tr.data_ptr_) +
            workload_idx * tr.dtype_byte_size_;
        } else {
            int64_t offset = 0;
            for (int64_t i = 0; i < ndims_; ++i) {
                offset += workload_idx / primary_strides_[i] *
                tr.byte_strides_[i];
                workload_idx = workload_idx % primary_strides_[i];
            }
            return static_cast<device char*>(tr.data_ptr_) + offset;
        }
    }
    
    /// Get data pointer from a TensorRef with \p workload_idx.
    /// Note: can be optimized by computing all input ptrs and output ptr
    /// together.
    ///
    /// Note: Assumes that sizeof(T) matches the data's dtype size, but does
    /// not check this constraint for performance reasons.
    template <typename T>
    device T* GetWorkloadDataPtr(const device TensorRef& tr,
                                 bool tr_contiguous,
                                 int64_t workload_idx) device const {
        // For 0-sized input reduction op, the output Tensor
        // workload_idx == 1 > NumWorkloads() == 0.
        if (workload_idx < 0) {
            return nullptr;
        }
        if (tr_contiguous) {
            return static_cast<device T*>(tr.data_ptr_) + workload_idx;
        } else {
            int64_t offset = 0;
            for (int64_t i = 0; i < ndims_; ++i) {
                offset += workload_idx / primary_strides_[i] *
                tr.byte_strides_[i];
                workload_idx = workload_idx % primary_strides_[i];
            }
            return static_cast<device T*>(static_cast<device void*>(tr.data_ptr_ + offset));
        }
    }
    
    /// Number of input and output Tensors.
    int64_t num_inputs_ = 0;
    
    /// Array of input TensorRefs.
    TensorRef inputs_[MAX_INPUTS];
    
    /// Array of output TensorRefs.
    TensorRef outputs_[MAX_OUTPUTS];
    
    /// Array of contiguous flags for all input TensorRefs.
    bool inputs_contiguous_[MAX_INPUTS];
    
    /// Array of contiguous flags for all output TensorRefs.
    bool outputs_contiguous_[MAX_OUTPUTS];
    
    /// The default strides for primary_shape_ for internal use only. Used to
    /// compute the actual strides and ultimately the index offsets.
    int64_t primary_strides_[MAX_DIMS];
    
    /// Indexer's global number of dimensions.
    int64_t ndims_ = 0;
};


} // namespace metal
} // namespace u3d

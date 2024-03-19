//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <unified3d/utility/Logging.h>
#include <optional>

namespace u3d::core {

// Avoids circular include.
class Tensor;

/// \brief TensorKey is used to represent single index, slice or advanced
/// indexing on a Tensor.
///
/// See https://numpy.org/doc/stable/reference/arrays.indexing.html for details.
class TensorKey {
public:
    /// Instantiates a TensorKey with single index mode.
    ///
    /// \code
    /// t[0]   : t.GetItem({TensorKey::Index(0)})
    /// t[2]   : t.GetItem({TensorKey::Index(2)})
    /// t[2, 3]: t.GetItem({TensorKey::Index(2), TensorKey::Index(3)})
    /// \endcode
    ///
    /// \param index: Index to the tensor.
    static TensorKey Index(int64_t index);

    /// Instantiates a TensorKey with slice mode.
    ///
    /// \code
    /// t[0:10:2]: t.GetItem({TensorKey::Slice(0    , 10  , 2   )})
    /// t[:-1]   : t.Getitem({TensorKey::Slice(None , None, -1  )})
    /// t[3:]    : t.GetItem({TensorKey::Slice(3    , None, None)})
    /// \endcode
    ///
    /// \param start: Start index. None means starting from the 0-th element.
    /// \param stop: Stop index. None means stopping at the last element.
    /// \param step: Step size. None means step size 1.
    static TensorKey Slice(std::optional<int64_t> start,
                           std::optional<int64_t> stop,
                           std::optional<int64_t> step);

    /// Instantiates a TensorKey with tensor-index (advanced indexing) mode.
    ///
    /// \code
    /// [[1, 2], [3:]]: advanced indexing on dim-0, slicing on dim-1.
    /// t.GetItem({
    ///     TensorKey::IndexTensor(Tensor::Init<int64_t>({1, 2}, device),
    ///     TensorKey::Slice(3, None, None),
    /// });
    /// \endcode
    ///
    /// \param index_tensor: Indexing tensor of dtype int64_t or bool.
    static TensorKey IndexTensor(const Tensor& index_tensor);

    enum class TensorKeyMode { Index, Slice, IndexTensor };
    ~TensorKey() = default;

    /// Returns TensorKey mode.
    [[nodiscard]] TensorKeyMode GetMode() const;

    /// Convert TensorKey to a string representation.
    [[nodiscard]] std::string ToString() const;

public:
    /// Get (single) index.
    /// For TensorKeyMode::Index only.
    [[nodiscard]] int64_t GetIndex() const;

    /// Get start index. Throws exception if start is None.
    /// For TensorKeyMode::Slice only.
    [[nodiscard]] int64_t GetStart() const;

    /// Get stop index. Throws exception if start is None.
    /// For TensorKeyMode::Slice only.
    [[nodiscard]] int64_t GetStop() const;

    /// Get step index. Throws exception if start is None.
    /// For TensorKeyMode::Slice only.
    [[nodiscard]] int64_t GetStep() const;

    /// When dim_size is know, convert the None values in start, stop, step with
    /// to concrete values and returns a new TensorKey.
    /// \code
    /// E.g. if t.shape == (5,), t[:4]:
    ///      before compute: Slice(None,    4, None)
    ///      after compute : Slice(   0,    4,    1)
    /// E.g. if t.shape == (5,), t[1:]:
    ///      before compute: Slice(   1, None, None)
    ///      after compute : Slice(   1,    5,    1)
    /// \endcode
    /// For TensorKeyMode::Slice only.
    [[nodiscard]] TensorKey InstantiateDimSize(int64_t dim_size) const;

    /// Get advanced indexing tensor.
    /// For TensorKeyMode::IndexTensor only.
    [[nodiscard]] Tensor GetIndexTensor() const;

private:
    class Impl;
    class IndexImpl;
    class SliceImpl;
    class IndexTensorImpl;
    std::shared_ptr<Impl> impl_;
    TensorKey(const std::shared_ptr<Impl>& impl);
};

}  // namespace u3d::core

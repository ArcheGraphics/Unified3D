//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <string>

namespace u3d::utility {

/// \brief Returns true if the file is a supported compressed file path. It does
/// not check if the file exists. It only checks the file extension.
/// \param file_path The file path to check.
bool IsSupportedCompressedFilePath(const std::string& file_path);

/// \brief Function to extract compressed files.
/// \param file_path Path to file. Example: "/path/to/file/file.zip"
/// \param extract_dir Directory path where the file will be extracted to. If
/// the directory does not exist, it will be created.
void Extract(const std::string& file_path, const std::string& extract_dir);

}  // namespace u3d::utility

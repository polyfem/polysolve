# Copy Polysolve header files into the build directory
function(polysolve_copy_headers)
  foreach(filepath IN ITEMS ${ARGN})
    get_filename_component(filename "${filepath}" NAME)
    if(${filename} MATCHES ".*\.(hpp|h|ipp|tpp)$")
      configure_file(${filepath} ${PROJECT_BINARY_DIR}/include/polysolve/${filename})
    endif()
  endforeach()
endfunction()

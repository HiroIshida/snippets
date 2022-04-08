function(setup_python_as_submodule relative_submodules_dir)

  if(ARGN)
    message(FATAL_ERROR "catkin_python_setup() called with unused arguments: ${ARGN}")
  endif()

  if(${PROJECT_NAME}_GENERATE_MESSAGES)
    message(FATAL_ERROR "generate_messages() must be called after catkin_python_setup() in project '${PROJECT_NAME}'")
  endif()
  if(${PROJECT_NAME}_GENERATE_DYNAMIC_RECONFIGURE)
    message(FATAL_ERROR "generate_dynamic_reconfigure_options() must be called after catkin_python_setup() in project '${PROJECT_NAME}'")
  endif()

  # if the Python install directory didn't exist before
  # the cached environment won't contain it in the PYTHONPATH
  if(NOT EXISTS "${CATKIN_DEVEL_PREFIX}/${PYTHON_INSTALL_DIR}")
    file(MAKE_DIRECTORY "${CATKIN_DEVEL_PREFIX}/${PYTHON_INSTALL_DIR}")
    # refresh environment cache
    safe_execute_process(COMMAND ${GENERATE_ENVIRONMENT_CACHE_COMMAND})
  endif()

  if(NOT EXISTS "${CATKIN_DEVEL_PREFIX}/${PYTHON_INSTALL_DIR}/${PROJECT_NAME}")
    file(MAKE_DIRECTORY "${CATKIN_DEVEL_PREFIX}/${PYTHON_INSTALL_DIR}/${PROJECT_NAME}")
    file(TOUCH "${CATKIN_DEVEL_PREFIX}/${PYTHON_INSTALL_DIR}/${PROJECT_NAME}/__init__.py")
  endif()

  if(EXISTS ${${PROJECT_NAME}_SOURCE_DIR}/${relative_submodules_dir}/__init__.py)
      message(FATAL_ERROR "submodule directory must not have __init__.py as it will be automatically generated")
  endif()

  file(GLOB PYTHON_SUBMODULE_DIRS "${${PROJECT_NAME}_SOURCE_DIR}/${relative_submodules_dir}/*")
  list(LENGTH PYTHON_SUBMODULE_DIRS modules_count)
  math(EXPR modules_range "${modules_count} - 1")
  foreach(index RANGE ${modules_range})
    list(GET PYTHON_SUBMODULE_DIRS ${index} submodule_dir)
    get_filename_component(submodule_name ${submodule_dir} NAME)
    execute_process(COMMAND ln -sf 
        ${submodule_dir} 
        ${CATKIN_DEVEL_PREFIX}/${PYTHON_INSTALL_DIR}/${PROJECT_NAME}/${submoduel_name})
  endforeach()

  set(${PROJECT_NAME}_CATKIN_PYTHON_SETUP_HAS_PACKAGE_INIT TRUE PARENT_SCOPE)
endfunction()

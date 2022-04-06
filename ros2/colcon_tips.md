symbolic linkをはることで, コンパイル不要のスクリプトのソースコード変更をinstallし直すことなく反映する. 
```
colcon build --symlink-install
```

## test
- buildするときは `--cmake-args -DBUILD_TESTING=ON`にする.
- `ament test`の代わりに`colcon test --packages-select` を使う. 
- testのexecutableはinstallにはおかれない. buildにおかれる.
- resultもbuildに置かれる.
- CMakeLists に以下を例にして追加
```cmake
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(signed_distance_function-test
    test/src/test_signed_distance_function.cpp
  )
  target_link_libraries(signed_distance_function-test
    signed_distance_function
  )
endif()
```

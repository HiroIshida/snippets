[![Institut Maupertuis logo](http://www.institutmaupertuis.fr/media/gabarit/logo.png)](http://www.institutmaupertuis.fr)

This is a ROS package demonstrating a simple ROS test usage.

# Directories in the project

| Directory  | Description
------------ | -----------
`src` | Contains the service implementation (a ROS node)
`srv` | Contains the service definition
`test` | Contains the test node (service client) and corresponding launch file

# Dependencies
- [Robot Operating System](http://wiki.ros.org/ROS/Installation)

# Compiling and launching the tests


Install the dependencies by following the wiki instructions and cloning the repositories into your catkin workspace.

`cd` to your catkin workspace source directory:
```bash
git clone https://gitlab.com/InstitutMaupertuis/simple_rostest.git
cd ..
```

`catkin_make`
---

```bash
catkin_make run_tests_simple_rostest
```

Or
```bash
catkin_make tests
catkin_make test
```

`catkin tools`
---

```bash
catkin test
```


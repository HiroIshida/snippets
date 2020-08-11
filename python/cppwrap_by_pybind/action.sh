cd build
cmake -DPYTHON_EXECUTABLE=/usr/bin/python2.7 ..
make 
mv example.so ../

cd ..
python test_passbyref.py

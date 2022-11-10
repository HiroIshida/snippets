find . -name "*.pkl" -print0 |xargs -0 -n 1 -P 12 pigz 
find . -name "*.pkl.gz" -print0 |xargs -0 -n 1 -P 12 unpigz 

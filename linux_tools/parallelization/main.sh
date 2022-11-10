find . -name "*.pkl" |xargs -n 1 -P 12 pigz --keep --force
find . -name "*.pkl.gz" -print0 |xargs -0 -n 1 -P 12 unpigz  --keep --force

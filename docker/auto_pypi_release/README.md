Edit `python_versions.txt` to change releasing python versions
```
docker build -t pypi .
```

```
docker run --rm pypi:latest \
    /bin/bash -i -c \
    'source ~/.bashrc; twine upload --skip-existing -u username -p mypw $HOME/tinyfk/dist/*'
```

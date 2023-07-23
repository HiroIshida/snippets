For example, suppose we have the following dataclass and dumped a object of it
```python
@dataclass
class Example:
    a: int
    b: float
```
If the definition changes after the pickling to the following, the load operation will fail causing an error saying "there is no attribute c ..."
```python
@dataclass
class Example:
    a: int
    b: float
    c: Optional[str]
```

To fix this issue, as a workaround, one can define `__setstate__` which is called right at the beginning of the depickling procedure as the following so that it set "c" to None is it's not found in the pickled object:
```python
@dataclass
class Example:
    a: int
    b: float
    c: Optional[str]

    def __setstate__(self, state):
        if "c" not in state:
            state["c"] = None
        self.__dict__.update(state)
```

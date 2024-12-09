g++
```
Polymorphic (unique_ptr) elapsed time: 1933 ms
Casted (static_cast) elapsed time: 1091 ms
Variant (std::visit) elapsed time: 604 ms
Variant (get_if) elapsed time: 662 ms
Reuse Variant (get_if) elapsed time: 873 ms
Enum + Switch elapsed time: 656 ms
```
clang++
```
Polymorphic (unique_ptr) elapsed time: 1772 ms
Casted (static_cast) elapsed time: 1744 ms
Variant (std::visit) elapsed time: 771 ms
Variant (get_if) elapsed time: 777 ms
Reuse Variant (get_if) elapsed time: 1388 ms
Enum + Switch elapsed time: 780 ms
```

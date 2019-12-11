# julia メモ

## dict
Dict([("A", 1), ("B", 2)])

## initialize julia array
m = reshape([], 0, 2)
or 
m = Array{Float64}(undef, 0, 0)

## julia custom package 
https://qiita.com/mametank/items/43330a9452f0039ca22d
pkg > generate HelloWorld

## when get stuck on "redefinition of constant Type" error ...
https://docs.julialang.org/en/v1/manual/faq/#How-can-I-modify-the-declaration-of-a-type-in-my-session?-1


## julia develop
package mode -> dev

## julia module export 
if the file included in the module is a package, then we must do 
```
export ASDF, evaluate
include("adaptive_distance_fields.jl")
using .AdaptivelySampledDistanceFields: ASDF, evaluate
```
If it's not a module, then just `include` is enough.

## julia array initialization 
`Array{Int}(undef, 0, 0, ...)`

## 野良パッケージの使い方
Suppose the directory of your package is "~/include". Then, hit `push!(LOAD_PATH, "/home/username/include")`. Note that somehow `~/` must be replaced by `~/home/username/`. This operation must be done every time you use the package.(true?)
## local package をdevに追加する方法
say, you have package `~/home/doc/MyPack.jl`. Then `cd ~/home/doc` and enter REPL, and do `Pkg> dev MyPack.jl`
(参考)[https://docs.julialang.org/en/v1.0/stdlib/Pkg/]

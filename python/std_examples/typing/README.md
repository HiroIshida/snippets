# covariance と contravarianceについて
covariance type は, subclass での置き換えを許し, contravarianceでは superclassでの置き換えを許す.

juliaの似たような議論
https://discourse.julialang.org/t/why-1-2-3-is-not-a-vector-number/52645/17


# なぜmutable containerはcovariantなりえないのか.
https://stackoverflow.com/questions/62814180/is-there-a-covariant-mutable-version-of-list
https://mypy.readthedocs.io/en/stable/common_issues.html#invariance-vs-covariance

# protocolが適している例.
`B <: A` とする. `C` というmixinを加えた`BC`と`AC`を考えることができる. このとき, protocolならば`BC <: AC`の関係を表せるが継承だと表せない.

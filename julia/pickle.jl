using Serialization
struct Hoge
  aho
  baka
end
hoge = Hoge(1, 2)
Serialization.serialize("tmp.jld", hoge)
data = Serialization.deserialize("tmp.jld")

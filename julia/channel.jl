# same as generator
function producer(c::Channel)
  put!(c, 1)
  put!(c, 3)
  put!(c, 5)
end

for e in Channel(producer)
  println(e)
end

# the following code using do-block works same as the code above
c = Channel() do c
  put!(c, 1)
  put!(c, 3)
  put!(c, 5)
end

for e in c
  println(e)
end


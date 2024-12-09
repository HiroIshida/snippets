import build.tmp as p
sphere = p.Sphere(1.0)
box = p.Box(1.0, 1.0, 1.0)
p.func(box)
p.func(sphere)
p.loop([box, sphere])

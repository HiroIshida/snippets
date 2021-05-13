using DataStructures

struct Node
    name::String
end
struct Value
    a::Float64
    b::Float64
end
Base.isless(v1::Value, v2::Value) = (v1.a < v1.b)

function example_simple()
    open_list = PriorityQueue{Node, Float64}()

    enqueue!(open_list, Node("a"), 2)
    enqueue!(open_list, Node("b"), 3)
    enqueue!(open_list, Node("c"), 1)
    dequeue!(open_list)
end

function example_custom()
    open_list = PriorityQueue{Node, Value}()
    enqueue!(open_list, Node("a"), Value(2, 2))
    enqueue!(open_list, Node("b"), Value(3, 2))
    enqueue!(open_list, Node("c"), Value(1, 2))
    dequeue!(open_list)
end

example_simple() == example_custom()



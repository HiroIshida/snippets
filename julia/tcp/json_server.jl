using Sockets 
using JSON

method1(x) = x
method2(x) = x^2
method3(x, y) = x + y

function dispatch(method_name, args)
    resp = begin
        if method_name == "method1"
            method1(args[1])
        elseif method_name == "method2"
            method2(args[1])
        elseif method_name == "method3"
            method3(args[1], args[2])
        end
    end
    dict = Dict([("resp", resp)])
    json = JSON.json(dict)
    return json
end

server = listen(ip"127.0.0.1", 2000)
sock = accept(server)
while true
    str = readline(sock)
    if length(str)>3
        dict = JSON.parse(str)
        println(dict["args"])
        json_resp = dispatch(dict["method_name"], dict["args"])
        write(sock, json_resp * "\n")
    else
        write(sock, "hoge\n")
    end
end


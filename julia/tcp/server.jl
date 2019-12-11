using Sockets 
server = listen(ip"127.0.0.1", 2000)
sock = accept(server)
while true
    write(sock, "echo: " * readline(sock) * "\n")
end


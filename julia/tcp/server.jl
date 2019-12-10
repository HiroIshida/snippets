using Sockets # for version 1.0

server = listen(ip"127.0.0.1", 2000)
while true
    sock = accept(server)
    @async begin 
      write(sock, "Connected to echo server.\r\n" * string(randn()))
      close(sock)
    end
end

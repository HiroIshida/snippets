import http.client

if __name__ == "__main__":
    conn = http.client.HTTPConnection("localhost", 8000)
    conn.request("GET", "/")
    response = conn.getresponse()
    print("Status:", response.status)
    print("Headers:", response.getheaders())
    print("Body:", response.read().decode())
    conn.close()

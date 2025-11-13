from socket import *
import os, struct

PATH = '/Users/jaden/projects/Learning/ComputerNetworks/programming/'
CHUNK = 1400

serverPort = 12000
serverSocket = socket(AF_INET, SOCK_DGRAM)
serverSocket.bind(('', serverPort))
print("server ready to receive")

while True:
    request, addr = serverSocket.recvfrom(2048)
    if request.decode() != "rdy":
        continue

    reply = "rdy"
    serverSocket.sendto(reply.encode(), addr)

    filename, addr = serverSocket.recvfrom(2048)
    filename = filename.decode().strip()
    full = os.path.join(PATH, filename)

    if not os.path.exists(full):
        msg= "file not found"
        serverSocket.sendto(msg.encode(), addr)
        break
    
    serverSocket.sendto(filename.encode(), addr)

       # send size first
    size = str(os.path.getsize(full))
    serverSocket.sendto(size.encode(), addr)

    while True:
        request, addr = serverSocket.recvfrom(2048)
        if request.decode() == "rdyD":
            break



    with open(full, 'rb') as f:
        while True:
            data = f.read(CHUNK)
            if not data:
                break
            serverSocket.sendto(data, addr)

    request, addr = serverSocket.recvfrom(2048)
    if request.decode() == "close":
        continue
    

    print(f"Sent {filename} ({size} bytes) to {addr}")
    
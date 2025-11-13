from socket import *
import struct, os

serverName = '127.0.0.1'
serverPort = 12000
PATH = '/Users/jaden/projects/Learning/ComputerNetworks/programming/'

clientSocket = socket(AF_INET,SOCK_DGRAM)
clientSocket.settimeout(5.0)

message = "rdy"
clientSocket.sendto(message.encode(),(serverName, serverPort))
print("Sent rdy to server")

reply, serverAddress = clientSocket.recvfrom(2048)
if reply.decode() != "rdy":
    raise SystemExit("server not ready")

filename = input('Enter filename').strip()
clientSocket.sendto(filename.encode(),(serverName, serverPort))


reply, serverAddress = clientSocket.recvfrom(2048)

if reply.decode() != filename:
    raise SystemExit("server did not ack file")

hdr, srv = clientSocket.recvfrom(8)
# unpack size of incoming file
size = int(hdr.decode())
print(f"expecting {size} bytes")


message = "rdyD"
clientSocket.sendto(message.encode(),(serverName, serverPort))
print("Sent rdyD to server")


dst = os.path.join(PATH, "ff" + filename)
received = 0

with open(dst, 'wb') as f:
    while received < size:
        chunk, srv = clientSocket.recvfrom(65535)
        f.write(chunk)
        received += len(chunk)

msg = "close"
clientSocket.sendto(msg.encode(),(serverName, serverPort))

print(f"download complete:({received} bytes)")

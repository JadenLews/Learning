from socket import *
import struct, os

serverName = '127.0.0.1'
serverPort = 12000
PATH = '/Users/jaden/projects/Learning/ComputerNetworks/programming/'
CHUNK = 1400

clientSocket = socket(AF_INET,SOCK_DGRAM)
clientSocket.settimeout(5.0)

message = "rdy"
clientSocket.sendto(message.encode(),(serverName, serverPort))
print("Sent rdy to server")

reply, serverAddress = clientSocket.recvfrom(2048)
if reply.decode() != "rdy":
    raise SystemExit("Server not ready")

filename = input('Enter filename to download:').strip()
clientSocket.sendto(filename.encode(),(serverName, serverPort))


reply, serverAddress = clientSocket.recvfrom(2048)

if reply.decode() == 'ERR_NOFILE':
    raise SystemExit("Server says file does not exist")
if reply.decode() != filename:
    raise SystemExit("Server did not ack filename")


message = "rdyD"
clientSocket.sendto(message.encode(),(serverName, serverPort))
print("Sent rdyD to server")

hdr, srv = clientSocket.recvfrom(8)
(size,) = struct.unpack('!Q', hdr)
print(f"Expecting {size} bytes")

dst = os.path.join(PATH, "ff" + filename)
received = 0

with open(dst, 'wb') as f:
    while received < size:
        chunk, srv = clientSocket.recvfrom(65535)
        f.write(chunk)
        received += len(chunk)

msg = "close"
clientSocket.sendto(msg.encode(), serverPort)

print(f"File download complete: {dst} ({received} bytes)")

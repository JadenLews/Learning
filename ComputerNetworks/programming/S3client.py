# client_two_packets.py
from socket import *
import os, struct

SERVER = '127.0.0.1'
PORT   = 12000
PATH   = '/Users/jaden/projects/Learning/ComputerNetworks/programming/'
TIMEOUT = 2.0

def main():
    filename = input("Enter filename to download: ").strip()

    sock = socket(AF_INET, SOCK_DGRAM)
    sock.settimeout(TIMEOUT)

    # handshake
    msg = 'rdy'
    sock.sendto(msg.encode(), (SERVER, PORT))
    rdy, srv = sock.recvfrom(1024)
    if rdy != msg.encode():
        raise SystemExit("not ready")
    print("handshake complete")

    # send filename
    sock.sendto(filename.encode(), srv)

    flnm, srv = sock.recvfrom(1024)
    if flnm != filename.encode():
        raise SystemExit("wrong file")
    
    # request # of chunks
    msg = '# of chunks'
    sock.sendto(msg.encode(), srv)


    # header 8 bytes
    total_chunks, srv = sock.recvfrom(65535)
    total_chunks = int(total_chunks.decode())
    print(f"Expecting {total_chunks} chunks")

    # storage
    chunks = [None] * total_chunks
    received = 0

    # ready download
    msg = 'rdyD'
    sock.sendto(msg.encode(), srv)

    while received < total_chunks:
        try:
            # receive index packet (4 bytes)
            pkt_idx, srv = sock.recvfrom(4)
            idx = int(pkt_idx.decode())

            #pkt_size, srv = sock.recvfrom(65535)
            #size = int(pkt_size.decode())


            # receive packet for that index
            data, srv = sock.recvfrom(65535)
            chunks[idx] = data
            received += 1

            # next one
            msg = 'next'
            sock.sendto(msg.encode(), srv)

        except timeout:
            # ask again on timeout
            msg = 'next'
            sock.sendto(msg.encode(), srv)




    # write file
    dst = os.path.join(PATH, f'ff{filename}')
    with open(dst, 'wb') as f:
        for i in range(total_chunks):
            f.write(chunks[i] if chunks[i] is not None else b'')
    print(f"download complete: {dst} ({received}/{total_chunks} chunks)")

    # close
    sock.sendto(b'close', srv)
    sock.close()

if __name__ == '__main__':
    main()
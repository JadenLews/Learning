# client_random_order.py
from socket import *
import os, struct

SERVER = '127.0.0.1'
PORT = 12000
PATH = '/Users/jaden/projects/Learning/ComputerNetworks/programming/'
SOCK_TIMEOUT = 2.0

def main():
    desired_chunks = '100' 

    sock = socket(AF_INET, SOCK_DGRAM)
    sock.settimeout(5.0)

    # handshake
    sock.sendto(b'rdy', (SERVER, PORT))
    rdy, srv = sock.recvfrom(2048)
    if rdy != b'rdy':
        raise SystemExit("Server not ready")

    filename = input("Enter filename to download: ").strip()
    sock.sendto(filename.encode(), (SERVER, PORT))

    ack, srv = sock.recvfrom(2048)
    if ack == b'ERR_NOFILE':
        raise SystemExit("No file")
    if ack.decode() != filename:
        raise SystemExit("wrong filename")

    # send desired number of chunks
    sock.sendto(desired_chunks.encode(), srv)

    # server replies: "NUM total chunk_size"
    reply, srv = sock.recvfrom(2048)
    parts = reply.decode().strip().split()
    if len(parts) != 3 or parts[0] != 'NUM':
        raise SystemExit(f"Unexpected reply: {reply!r}")
    total = int(parts[1])
    chunk_size = int(parts[2])
    print(f"Server reports: total chunks={total}, chunk_size={chunk_size}")

    # ready download
    sock.sendto(b'rdyD', srv)

    # storage
    chunks = [None] * total
    received = 0

    # first prompt
    sock.settimeout(SOCK_TIMEOUT)
    sock.sendto(b'chunk index', srv)

    while received < total:
        try:
            pkt, srv = sock.recvfrom(65535)
        except timeout:
            sock.sendto(b'next', srv)
            continue

        if len(pkt) < 4:
            sock.sendto(b'next', srv)
            continue

        (idx,) = struct.unpack('!I', pkt[:4])
        if idx == 0xFFFFFFFF:
            if received >= total:
                break
            sock.sendto(b'next', srv)
            continue

        data = pkt[4:]
        if 0 <= idx < total and chunks[idx] is None:
            chunks[idx] = data
            received += 1

        sock.sendto(b'next', srv)

    # write assembled file
    dst = os.path.join(PATH, 'ff' + filename)
    with open(dst, 'wb') as f:
        for i in range(total):
            f.write(chunks[i] if chunks[i] is not None else b'')
    print(f"Wrote {dst} ({received}/{total} chunks)")

    sock.sendto(b'close', srv)
    sock.close()

if __name__ == '__main__':
    main()
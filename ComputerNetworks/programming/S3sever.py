# server_two_packets.py
from socket import *
import os, math, random, struct

PORT = 12000
PATH = '/Users/jaden/projects/Learning/ComputerNetworks/programming/'
MAX = 1400 

def build_chunks(data: bytes, wanted: int = 100):
    size = len(data)
    wanted = max(1, wanted)
    chunk_size = max(1, min(MAX, math.ceil(size / wanted)))
    chunks = [data[i:i+chunk_size] for i in range(0, size, chunk_size)]
    return chunks, chunk_size

def main():
    sock = socket(AF_INET, SOCK_DGRAM)
    sock.bind(('', PORT))
    print(f"Server ready on UDP {PORT}")

    while True:
        # handshake
        req, addr = sock.recvfrom(1024)
        msg = 'rdy'
        if req != msg.encode():
            continue
        sock.sendto(msg.encode(), addr)
        print(f"handshake complete")

        # filename
        fname_raw, addr = sock.recvfrom(4096)
        fname = fname_raw.decode().strip()
        full = os.path.join(PATH, fname)
        if not os.path.exists(full):
            msg = 'file not found'
            sock.sendto(msg.encode(), addr)
            continue
            
        sock.sendto(fname_raw, addr)
        

        with open(full, 'rb') as f:
            data = f.read()

        chunks, chunk_size = build_chunks(data, wanted=100)
        total = len(chunks)
        # rec "# of chunks"

        rec, addr = sock.recvfrom(4096)
        if rec.decode().strip() != '# of chunks':
            print("Did not receive # of chunks")
            continue

        # header: total_chunks
        msg = str(total)
        sock.sendto(msg.encode(), addr)

        # wait for "rdyD"
        ready, addr = sock.recvfrom(1024)
        if ready.decode().strip() != 'rdyD':
            continue

        order = list(range(total))
        random.shuffle(order)
        cursor = 0

        # serve chunks on "next"
        while True:
            if cursor < total:
                idx = order[cursor]
                idx = str(idx)
                sock.sendto(idx.encode(), addr)  # index packet
                idx = int(idx)
                sock.sendto(chunks[idx], addr)     # data packet
                cursor += 1
            else:
                break

            req, addr = sock.recvfrom(1024)
            txt = req.decode().strip().lower()

            if txt == 'close':
                print(f"server transfer done")
                break
            if txt != 'next':
                continue


        req, addr = sock.recvfrom(1024)
        if req.decode().strip() == 'close':
            print(f"Transfer of {fname} complete")
        

if __name__ == '__main__':
    main()
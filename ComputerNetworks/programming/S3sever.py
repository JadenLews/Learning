# server_random_order.py
from socket import *
import os, math, random, struct

PATH = '/Users/jaden/projects/Learning/ComputerNetworks/programming/'
PORT = 12000
MAX_UDP = 1400 

def build_chunks(data: bytes, desired_chunks: int):
    size = len(data)
    chunk_size = max(1, math.ceil(size / max(1, desired_chunks)))
    chunk_size = min(chunk_size, MAX_UDP - 4)
    chunks = [data[i:i+chunk_size] for i in range(0, size, chunk_size)]
    return chunks, chunk_size

def main():
    sock = socket(AF_INET, SOCK_DGRAM)
    sock.bind(('', PORT))
    print(f"Server ready on UDP {PORT}")

    while True:
        req, addr = sock.recvfrom(2048)
        if req.decode(errors='ignore').strip() != 'rdy':
            continue
        sock.sendto(b'rdy', addr)

        fname_raw, addr = sock.recvfrom(2048)
        fname = fname_raw.decode(errors='ignore').strip()
        full = os.path.join(PATH, fname)
        if not os.path.exists(full):
            sock.sendto(b'ERR_NOFILE', addr)
            continue
        sock.sendto(fname.encode(), addr)

        q_raw, addr = sock.recvfrom(2048)
        try:
            desired = int(q_raw.decode(errors='ignore').strip())
        except ValueError:
            desired = 100

        with open(full, 'rb') as f:
            data = f.read()
        chunks, chunk_size = build_chunks(data, desired)
        total = len(chunks)

        sock.sendto(f'NUM {total} {chunk_size}'.encode(), addr)

        # wait for "rdyD"
        ready, addr = sock.recvfrom(2048)
        if ready.decode(errors='ignore').strip() != 'rdyD':
            continue

        order = list(range(total))
        random.shuffle(order)
        cursor = 0

        # serve until client says close
        while True:
            msg, addr = sock.recvfrom(2048)
            txt = msg.decode(errors='ignore').strip().lower()

            if txt == 'close':
                print(f"Transfer of {fname} complete for {addr}")
                break

            if txt not in ('chunk index', 'next'):
                continue

            if cursor < total:
                idx = order[cursor]
                payload = chunks[idx]
                pkt = struct.pack('!I', idx) + payload 
                sock.sendto(pkt, addr)
                cursor += 1
            else:
                # no more
                sock.sendto(struct.pack('!I', 0xFFFFFFFF), addr)

if __name__ == '__main__':
    main()
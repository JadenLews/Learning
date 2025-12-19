import java.io.*;
import java.net.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public class Group_Peer {

    // === GLOBAL CONFIG FOR THIS 5-PEER RING ===

    // Tracker location (change these to match your tracker machine)
    private static final String TRACKER_HOST = "localhost";  // TODO: set to your tracker IP
    private static final int    TRACKER_PORT = 12000;
    private static final int    INITIAL_CHUNKS_FROM_TRACKER = 10;

    // 5 peers in the ring.
    // Index 0 -> peer1, 1 -> peer2, etc.
    // You can set all of these to "localhost" for testing.
    private static final String[] PEER_IPS = {
        "localhost",   // peer1
        "localhost",   // peer2
        "localhost",  // peer3
        "localhost",  // peer4
        "localhost"    // peer5
    };

    // Upload ports for each peer (same index as IPs)
    private static final int[] PEER_UPLOAD_PORTS = {
        1001, 1002, 1003, 1004, 1005
    };

    // Chunk directories for each peer (same index as IPs)
    private static final String[] PEER_CHUNK_DIRS = {
        "chunks_p1",
        "chunks_p2",
        "chunks_p3",
        "chunks_p4",
        "chunks_p5"
    };

    private static final int NUM_PEERS_IN_RING = PEER_IPS.length;
    private static final int CHUNK_SIZE = 10 * 1024;

    private final String peerId;
    private final String trackerHost;
    private final int trackerPort;
    private final int uploadPort;
    private final String downloadNeighborHost;
    private final int downloadNeighborPort;
    private final Path chunkDir;
    private int initialChunksFromTracker;

    // shared between threads
    private final Set<Integer> ownedChunks = ConcurrentHashMap.newKeySet();
    private volatile int totalChunks = -1;
    private volatile String fileName = "unknown";

    // CONSTRUCTOR

    public Group_Peer(String peerId,
                      String trackerHost, int trackerPort,
                      int uploadPort,
                      String downloadNeighborHost, int downloadNeighborPort,
                      String chunkDir,
                      int initialChunksFromTracker) {

        this.peerId = peerId;
        this.trackerHost = trackerHost;
        this.trackerPort = trackerPort;
        this.uploadPort = uploadPort;
        this.downloadNeighborHost = downloadNeighborHost;
        this.downloadNeighborPort = downloadNeighborPort;
        this.chunkDir = Paths.get(chunkDir);
        this.initialChunksFromTracker = initialChunksFromTracker;
    }

    // STARTUP 

    public void start() throws IOException {
        Files.createDirectories(chunkDir);

        // 1: talk to tracker, get chunks
        connectToTrackerAndDownloadInitialChunks();

        // 2: write summary files
        writeSummaryFile();

        // 3: upload server thread
        Thread uploadThread = new Thread(this::runUploadServer, "UploadServer-" + peerId);
        uploadThread.start();

        // 4: start download client thread
        Thread downloadThread = new Thread(this::runDownloadClient, "DownloadClient-" + peerId);
        downloadThread.start();

        System.out.println("Peer " + peerId + " started. UploadPort=" + uploadPort +
                ", downloadNeighbor=" + downloadNeighborHost + ":" + downloadNeighborPort);
    }

    // TALK TO TRACKER

    private void connectToTrackerAndDownloadInitialChunks() {
        try (Socket socket = new Socket(trackerHost, trackerPort)) {
            InputStream in = socket.getInputStream();
            OutputStream out = socket.getOutputStream();

            // rdy / rdy
            sendLine(out, "rdy");
            String resp = readLine(in);
            if (!"rdy".equals(resp)) {
                System.err.println("Peer " + peerId + ": expected 'rdy' " + resp);
                return;
            }

            // fName
            resp = readLine(in); 
            fileName = resp;
            sendLine(out, resp);

            System.out.println("Peer " + peerId + ": tracker fileName=" + fileName);

            // # of chunks
            resp = readLine(in);
            totalChunks = Integer.parseInt(resp);
            sendLine(out, resp);
            System.out.println("Peer " + peerId + ": totalChunks=" + totalChunks);
            initialChunksFromTracker = totalChunks;
            // ready to download
            sendLine(out, "rdyD");

            // choose indices
            List<Integer> chosen = chooseInitialChunkIndices(totalChunks, initialChunksFromTracker);
            System.out.println("Peer " + peerId + ": initial chunk indices " + chosen);

            for (int idx : chosen) {
                requestChunkFromTracker(idx, in, out);
                sendLine(out, "next");
            }

            sendLine(out, "close");

        } catch (IOException e) {
            System.err.println("Peer " + peerId + ": error talking to tracker" + e.getMessage());
        }
    }

    private void requestChunkFromTracker(int index,
                                         InputStream in,
                                         OutputStream out) throws IOException {
        sendLine(out, "chunk index " + index);

        String szLine = readLine(in); // "chunk size X"
        if (szLine == null || !szLine.startsWith("chunk size")) {
            System.err.println("Peer " + peerId + ": invalid size line for chunk " + index + ": " + szLine);
            return;
        }
        int size = Integer.parseInt(szLine.split("\\s+")[2]);
        if (size <= 0) {
            System.err.println("Peer " + peerId + ": chunk " + index + " has size " + size);
            return;
        }

        byte[] buf = rawRead(in, size);
        saveChunkToDisk(index, buf);
        ownedChunks.add(index);

        System.out.println("Peer " + peerId + ": got chunk " + index + " from tracker");
    }

    private int getPeerIndexFromId() {
        int i = peerId.length() - 1;
        while (i >= 0 && Character.isDigit(peerId.charAt(i))) {
            i--;
        }
        if (i == peerId.length() - 1) return 0; 

        String numStr = peerId.substring(i + 1);
        try {
            int oneBased = Integer.parseInt(numStr);
            return Math.max(0, (oneBased - 1) % NUM_PEERS_IN_RING);
        } catch (NumberFormatException e) {
            return 0;
        }
    }

    private List<Integer> chooseInitialChunkIndices(int total, int count) {
        List<Integer> result = new ArrayList<>();
        int myIdx = getPeerIndexFromId(); 

        for (int i = 0; i < total && result.size() < count; i++) {
            if (i % NUM_PEERS_IN_RING == myIdx) {
                result.add(i);
            }
        }

        System.out.println("Peer " + peerId + ": initial chunk indices " + result);
        return result;
    }

    private void runUploadServer() {
        try (ServerSocket serverSocket = new ServerSocket(uploadPort)) {
            System.out.println("Peer " + peerId + ": upload server on port " + uploadPort);

            while (true) {
                Socket neighbor = serverSocket.accept();
                System.out.println("Peer " + peerId + ": upload neighbor connected from "
                        + neighbor.getRemoteSocketAddress());
                new Thread(() -> handleUploadNeighbor(neighbor),
                        "UploadHandler-" + peerId).start();
            }
        } catch (IOException e) {
            System.err.println("Peer " + peerId + ": upload server error - " + e.getMessage());
        }
    }

    private void handleUploadNeighbor(Socket socket) {
        try {
            InputStream in = socket.getInputStream();
            OutputStream out = socket.getOutputStream();

            // handshake rdy / rdy
            String line = readLine(in);
            if (!"rdy".equals(line)) {
                System.out.println("Peer " + peerId + " UPeer: expected 'rdy', got " + line);
                return;
            }
            sendLine(out, "rdy");

            // wait for "chunkIDList"
            line = readLine(in);
            if (!"chunkIDList".equals(line)) {
                System.out.println("Peer " + peerId + " UPeer: expected 'chunkIDList', got " + line);
                return;
            }

            // send our chunk ID list
            sendChunkIdList(out);

            line = readLine(in);
            if ("close".equals(line)) {
                // neighbor doesn't need anything after seeing our list
                System.out.println("Peer " + peerId + " UPeer: neighbor closed after list");
                return;
            }
            if (!"rdyD".equals(line)) {
                System.out.println("Peer " + peerId + " UPeer: expected 'rdyD' or 'close', got " + line);
                return;
            }

            // respond to requests
            while ((line = readLine(in)) != null) {
                if (line.startsWith("chunk index")) {
                    int idx = Integer.parseInt(line.split("\\s+")[2]);
                    sendChunkToNeighbor(idx, out);
                } else if ("close".equals(line)) {
                    System.out.println("Peer " + peerId + " UPeer: neighbor closed");
                    break;
                } else {
                    // ignore other stuff for now
                }
            }

        } catch (IOException e) {
            System.err.println("Peer " + peerId + " UPeer: error - " + e.getMessage());
        } finally {
            try { socket.close(); } catch (IOException ignored) {}
        }
    }

    private void sendChunkIdList(OutputStream out) throws IOException {
        sendLine(out, "LIST " + ownedChunks.size());
        for (int id : ownedChunks) {
            sendLine(out, Integer.toString(id));
        }
        sendLine(out, "END");
        System.out.println("Peer " + peerId + " UPeer: sent chunk ID list " + ownedChunks);
    }

    private void sendChunkToNeighbor(int index, OutputStream out) throws IOException {
        Path path = chunkDir.resolve("chunk_" + index + ".dat");
        if (!Files.exists(path)) {
            sendLine(out, "chunk size 0");
            System.out.println("Peer " + peerId + " UPeer: requested chunk " + index +
                    " but do not have it");
            return;
        }
        byte[] data = Files.readAllBytes(path);
        sendLine(out, "chunk size " + data.length);
        out.write(data);
        out.flush();

        System.out.println("Peer " + peerId + " UPeer: sent chunk " + index);
    }

    private void runDownloadClient() {
        try { Thread.sleep(1000); } catch (InterruptedException ignored) {}

        while (true) {
            try (Socket socket = new Socket(downloadNeighborHost, downloadNeighborPort)) {
                InputStream in = socket.getInputStream();
                OutputStream out = socket.getOutputStream();

                System.out.println("Peer " + peerId + " DPeer: connected to "
                        + downloadNeighborHost + ":" + downloadNeighborPort);

                // handshake
                sendLine(out, "rdy");
                String line = readLine(in);
                if (!"rdy".equals(line)) {
                    System.err.println("Peer " + peerId + " DPeer: expected 'rdy', got " + line);
                    continue;
                }

                // request chunk ID list
                sendLine(out, "chunkIDList");
                Set<Integer> neighborChunks = receiveChunkIdList(in);
                System.out.println("Peer " + peerId + " DPeer: neighbor has " + neighborChunks);

                // compute missing
                Set<Integer> missing = new HashSet<>(neighborChunks);
                missing.removeAll(ownedChunks);
                System.out.println("Peer " + peerId + " DPeer: missing from neighbor: " + missing);

                if (missing.isEmpty()) {
                    sendLine(out, "close");

                    // if already have the whole file, done
                    if (totalChunks > 0 && ownedChunks.size() >= totalChunks) {
                        System.out.println("Peer " + peerId + " DPeer: have all chunks!");
                        reconstructFileIfComplete();
                        break;
                    }

                    // otherwise, wait a bit and try again later
                    try { Thread.sleep(1000); } catch (InterruptedException ignored) {}
                    continue;
                }

                // ready to download chunks
                sendLine(out, "rdyD");

                for (int idx : missing) {
                    sendLine(out, "chunk index " + idx);
                    String szLine = readLine(in); // "chunk size X"
                    if (szLine == null || !szLine.startsWith("chunk size")) {
                        System.err.println("Peer " + peerId + " DPeer: invalid size line for chunk "
                                + idx + ": " + szLine);
                        break;
                    }
                    int size = Integer.parseInt(szLine.split("\\s+")[2]);
                    if (size <= 0) continue;
                    byte[] data = rawRead(in, size);
                    saveChunkToDisk(idx, data);
                    ownedChunks.add(idx);
                    System.out.println("Peer " + peerId + " DPeer: downloaded chunk " + idx);
                }

                sendLine(out, "close");

                if (totalChunks > 0 && ownedChunks.size() >= totalChunks) {
                    System.out.println("Peer " + peerId + " DPeer: have all chunks!");
                    reconstructFileIfComplete();
                    break;
                }

                // Allow time for ring propagation before the next round
                try { Thread.sleep(1000); } catch (InterruptedException ignored) {}

            } catch (IOException e) {
                System.err.println("Peer " + peerId + " DPeer: error - " + e.getMessage());
                // neighbor might not be up yet; wait and retry
                try { Thread.sleep(1000); } catch (InterruptedException ignored) {}
            }
        }
    }

    private Set<Integer> receiveChunkIdList(InputStream in) throws IOException {
        Set<Integer> ids = new HashSet<>();
        String header = readLine(in); // "LIST n"
        if (header == null || !header.startsWith("LIST")) return ids;
        String line;
        while ((line = readLine(in)) != null) {
            if ("END".equals(line)) break;
            ids.add(Integer.parseInt(line.trim()));
        }
        return ids;
    }

    // UTILITIES 

    private void saveChunkToDisk(int index, byte[] data) throws IOException {
        Path path = chunkDir.resolve("chunk_" + index + ".dat");
        Files.write(path, data);
    }

    private void writeSummaryFile() throws IOException {
        Path summary = chunkDir.resolve("summary.txt");
        try (BufferedWriter w = Files.newBufferedWriter(summary)) {
            w.write("fileName=" + fileName + "\n");
            w.write("totalChunks=" + totalChunks + "\n");
            w.write("ownedChunks=" + ownedChunks + "\n");
        }
        System.out.println("Peer " + peerId + ": wrote summary " + summary);
    }

    private void reconstructFileIfComplete() {
        if (totalChunks <= 0) return;
        if (ownedChunks.size() < totalChunks) return;

        try {
            Path outFile = chunkDir.resolve("RECONSTRUCTED_" + fileName);
            try (OutputStream out = Files.newOutputStream(outFile)) {
                for (int i = 0; i < totalChunks; i++) {
                    Path path = chunkDir.resolve("chunk_" + i + ".dat");
                    if (!Files.exists(path)) {
                        System.err.println("Peer " + peerId +
                                ": missing chunk " + i + " during reconstruction");
                        return;
                    }
                    Files.copy(path, out);
                }
            }
            System.out.println("Peer " + peerId + ": reconstructed file to " + outFile);
        } catch (IOException e) {
            System.err.println("Peer " + peerId + ": reconstruction error - " + e.getMessage());
        }
    }

    // HELPERS

    private static void sendLine(OutputStream out, String msg) throws IOException {
        out.write((msg + "\n").getBytes("UTF-8"));
        out.flush();
    }

    private static String readLine(InputStream in) throws IOException {
        StringBuilder sb = new StringBuilder();
        int ch;
        while ((ch = in.read()) != -1) {
            if (ch == '\n') break;
            if (ch != '\r') sb.append((char) ch);
        }
        if (sb.length() == 0 && ch == -1) {
            return null; // EOF
        }
        return sb.toString();
    }

    private static byte[] rawRead(InputStream in, int size) throws IOException {
        byte[] buf = new byte[size];
        int off = 0;
        while (off < size) {
            int r = in.read(buf, off, size - off);
            if (r == -1) throw new EOFException("Unexpected EOF");
            off += r;
        }
        return buf;
    }

    // MAIN:
    // java Group_Peer <peerIndex>
    // where peerIndex is 0..4 corresponding to PEER_IPS/PORTS/DIRS

    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            System.err.println("Usage: java Group_Peer <peerIndex>");
            System.err.println("  where peerIndex is 0.." + (PEER_IPS.length - 1));
            System.exit(1);
        }

        int myIndex;
        try {
            myIndex = Integer.parseInt(args[0]);
        } catch (NumberFormatException e) {
            System.err.println("peerIndex must be an integer");
            System.exit(1);
            return;
        }

        if (myIndex < 0 || myIndex >= PEER_IPS.length + 1) {
            System.err.println("peerIndex must be between 1 and " + (PEER_IPS.length));
            System.exit(1);
        }
        myIndex = myIndex - 1;

        String peerId = "peer" + (myIndex + 1);
        String myIp = PEER_IPS[myIndex];
        int uploadPort = PEER_UPLOAD_PORTS[myIndex];
        String downloadHost = PEER_IPS[(myIndex + 1) % NUM_PEERS_IN_RING];
        int downloadPort = PEER_UPLOAD_PORTS[(myIndex + 1) % NUM_PEERS_IN_RING];
        String chunkDir = PEER_CHUNK_DIRS[myIndex];

        System.out.println("Starting " + peerId + " (index " + myIndex + "):");
        System.out.println("  myIp=" + myIp);
        System.out.println("  trackerHost=" + TRACKER_HOST + ":" + TRACKER_PORT);
        System.out.println("  uploadPort=" + uploadPort);
        System.out.println("  downloadNeighbor=" + downloadHost + ":" + downloadPort);
        System.out.println("  chunkDir=" + chunkDir);

        Group_Peer peer = new Group_Peer(
                peerId,
                TRACKER_HOST,
                TRACKER_PORT,
                uploadPort,
                downloadHost,
                downloadPort,
                chunkDir,
                INITIAL_CHUNKS_FROM_TRACKER
        );
        peer.start();
    }
}
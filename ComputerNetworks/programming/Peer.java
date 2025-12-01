import java.io.*;
import java.net.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public class Peer {
    private static final int NUM_PEERS_IN_RING = 5;

    private static final int CHUNK_SIZE = 10 * 1024;

    private final String peerId;
    private final String trackerHost;
    private final int trackerPort;
    private final int uploadPort;
    private final String downloadNeighborHost;
    private final int downloadNeighborPort;
    private final Path chunkDir;
    private final int initialChunksFromTracker;

    // shared between threads
    private final Set<Integer> ownedChunks = ConcurrentHashMap.newKeySet();
    private volatile int totalChunks = -1;
    private volatile String fileName = "unknown";

    /* ========= CONSTRUCTOR ========= */

    public Peer(String peerId,
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

    /* ========= STARTUP ========= */

    public void start() throws IOException {
        Files.createDirectories(chunkDir);

        // Step 1: talk to tracker & get initial chunks
        connectToTrackerAndDownloadInitialChunks();

        // Step 2: write summary file
        writeSummaryFile();

        // Step 3: start upload server thread (UPeer)
        Thread uploadThread = new Thread(this::runUploadServer, "UploadServer-" + peerId);
        uploadThread.start();

        // Step 4: start download client thread (DPeer)
        Thread downloadThread = new Thread(this::runDownloadClient, "DownloadClient-" + peerId);
        downloadThread.start();

        System.out.println("Peer " + peerId + " started. UploadPort=" + uploadPort +
                ", downloadNeighbor=" + downloadNeighborHost + ":" + downloadNeighborPort);
    }

    












    /* ========= TALK TO TRACKER ========= */

    private void connectToTrackerAndDownloadInitialChunks() {
        try (Socket socket = new Socket(trackerHost, trackerPort)) {
            InputStream in = socket.getInputStream();
            OutputStream out = socket.getOutputStream();

            // rdy / rdy
            sendLine(out, "rdy");
            String resp = readLine(in);
            if (!"rdy".equals(resp)) {
                System.err.println("Peer " + peerId + ": expected 'rdy', got " + resp);
                return;
            }

            // fName
            sendLine(out, "fName");
            resp = readLine(in); // "fName foo.txt"
            if (resp != null && resp.startsWith("fName")) {
                String[] parts = resp.split("\\s+", 2);
                if (parts.length == 2) {
                    fileName = parts[1];
                }
            }
            System.out.println("Peer " + peerId + ": tracker fileName=" + fileName);

            // # of chunks
            sendLine(out, "# of chunks");
            resp = readLine(in); // "# of chunks N"
            if (resp != null && resp.startsWith("# of chunks")) {
                String[] parts = resp.split("\\s+");
                totalChunks = Integer.parseInt(parts[3]);
                System.out.println("Peer " + peerId + ": totalChunks=" + totalChunks);
            }

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
            System.err.println("Peer " + peerId + ": error talking to tracker - " + e.getMessage());
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




    /** Turn "peer1", "peer2", ... into 0, 1, 2, ... */
    private int getPeerIndexFromId() {
        int i = peerId.length() - 1;
        while (i >= 0 && Character.isDigit(peerId.charAt(i))) {
            i--;
        }
        if (i == peerId.length() - 1) return 0;  // no trailing digits, default 0

        String numStr = peerId.substring(i + 1);
        try {
            int oneBased = Integer.parseInt(numStr);
            return Math.max(0, (oneBased - 1) % NUM_PEERS_IN_RING); // 0-based
        } catch (NumberFormatException e) {
            return 0;
        }
    }

    /** Partition chunk IDs across peers with no overlap. */
    private List<Integer> chooseInitialChunkIndices(int total, int count) {
        List<Integer> result = new ArrayList<>();
        int myIdx = getPeerIndexFromId(); // 0..NUM_PEERS_IN_RING-1

        for (int i = 0; i < total && result.size() < count; i++) {
            if (i % NUM_PEERS_IN_RING == myIdx) {
                result.add(i);
            }
        }

        System.out.println("Peer " + peerId + ": initial chunk indices " + result);
        return result;
    }

    /* ========= UPLOAD SERVER (UPeer) ========= */

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

        // wait for "rdyD"
        line = readLine(in);
        if (!"rdyD".equals(line)) {
            System.out.println("Peer " + peerId + " UPeer: expected 'rdyD', got " + line);
            return;
        }

        // now respond to "chunk index i" requests
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
    // format: "LIST <count>" then each id on its own line, then "END"
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

    /* ========= DOWNLOAD CLIENT (DPeer) ========= */

private void runDownloadClient() {
    // Give others time to start their upload servers
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

                // if we already have the whole file, we’re done
                if (totalChunks > 0 && ownedChunks.size() >= totalChunks) {
                    System.out.println("Peer " + peerId + " DPeer: have all chunks!");
                    reconstructFileIfComplete();
                    break;
                }

                // otherwise, wait a bit and try again later (neighbor may learn more)
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

    /* ========= FILE & SUMMARY UTILITIES ========= */

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
            System.out.println("Peer " + peerId + ": reconstructed file -> " + outFile);
        } catch (IOException e) {
            System.err.println("Peer " + peerId + ": reconstruction error - " + e.getMessage());
        }
    }

    /* ========= LOW-LEVEL I/O HELPERS ========= */

    private static void sendLine(BufferedWriter out, String msg) throws IOException {
        out.write(msg);
        out.write("\n");
        out.flush();
    }

    private static void expectLine(BufferedReader in, String expected) throws IOException {
        String line = in.readLine();
        if (!expected.equals(line)) {
            throw new IOException("Expected '" + expected + "', got '" + line + "'");
        }
    }

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

    /* ========= MAIN ========= */

    // Usage:
    // java Peer <peerId> <trackerHost> <trackerPort> <uploadPort>
    //           <downloadNeighborHost> <downloadNeighborPort>
    //           <chunkDir> <initialChunksFromTracker>
    public static void main(String[] args) throws Exception {
        if (args.length != 8) {
            System.err.println("Usage: java Peer <peerId> <trackerHost> <trackerPort> <uploadPort> " +
                    "<downloadNeighborHost> <downloadNeighborPort> <chunkDir> <initialChunksFromTracker>");
            System.exit(1);
        }

        String peerId = args[0];
        String trackerHost = args[1];
        int trackerPort = Integer.parseInt(args[2]);
        int uploadPort = Integer.parseInt(args[3]);
        String downloadHost = args[4];
        int downloadPort = Integer.parseInt(args[5]);
        String chunkDir = args[6];
        int initialFromTracker = Integer.parseInt(args[7]);

        Peer peer = new Peer(peerId, trackerHost, trackerPort,
                             uploadPort, downloadHost, downloadPort,
                             chunkDir, initialFromTracker);
        peer.start();
    }
}
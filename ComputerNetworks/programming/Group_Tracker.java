import java.io.*;
import java.net.*;
import java.nio.file.*;

public class Group_Tracker {

    private static final int CHUNK_SIZE = 10 * 1024; // 10 kB

    // === Hard-coded config ===
    private static final int TRACKER_PORT = 8000;

    // TODO: change this to the full path of the file you want to share
    private static final String FILE_TO_SHARE =
            "/Users/jaden/projects/Learning/ComputerNetworks/programming/archive.zip";

    private final String fileName;
    private final Path chunkDir;
    private final int totalChunks;

    public Group_Tracker(String filePath) throws IOException {
        Path path = Paths.get(filePath);
        this.fileName = path.getFileName().toString();

        this.chunkDir = path.getParent().resolve(fileName + "_chunks");
        Files.createDirectories(chunkDir);

        this.totalChunks = splitIntoChunks(path, CHUNK_SIZE);
        System.out.println("Tracker: file=" + fileName + ", total chunks=" + totalChunks);
        System.out.println("Tracker: chunks stored in " + chunkDir.toAbsolutePath());
    }

    /* ========== Helper: line I/O ========== */

    private static void sendLine(OutputStream out, String msg) throws IOException {
        out.write((msg + "\n").getBytes("UTF-8"));
        out.flush();
    }

    private static String readLine(InputStream in) throws IOException {
        StringBuilder sb = new StringBuilder();
        int ch;
        while ((ch = in.read()) != -1) {
            if (ch == '\n') break;
            if (ch != '\r') {
                sb.append((char) ch);
            }
        }
        if (sb.length() == 0 && ch == -1) {
            return null; // EOF
        }
        return sb.toString();
    }

    /* ========== Startup ========== */

    public void startServer(int port) throws IOException {
        ServerSocket serverSocket = new ServerSocket(port);
        System.out.println("Tracker listening on TCP port " + port);

        while (true) {
            Socket peerSocket = serverSocket.accept();
            System.out.println("Tracker: peer connected from " + peerSocket.getRemoteSocketAddress());
            new Thread(() -> handlePeer(peerSocket)).start();
        }
    }

    /* ========== Protocol handling per peer ========== */

    private void handlePeer(Socket socket) {
        try {
            InputStream in = socket.getInputStream();
            OutputStream out = socket.getOutputStream();

            // 1) rdy / rdy
            String line = readLine(in);
            if (!"rdy".equals(line)) {
                System.out.println("Tracker: expected 'rdy', got " + line);
                return;
            }
            sendLine(out, "rdy");

            // 2) send file name, expect echo (Group_Peer reads name, sends it back)
            sendLine(out, fileName);
            line = readLine(in); // echo
            if (line == null || !line.equals(fileName)) {
                System.out.println("Tracker: expected echo of fileName, got " + line);
                return;
            }

            // 3) send # of chunks, expect echo
            sendLine(out, Integer.toString(totalChunks));
            line = readLine(in); // echo
            if (line == null || !line.equals(Integer.toString(totalChunks))) {
                System.out.println("Tracker: expected echo of totalChunks, got " + line);
                return;
            }

            // 4) wait for rdyD
            line = readLine(in);
            if (!"rdyD".equals(line)) {
                System.out.println("Tracker: expected 'rdyD', got " + line);
                return;
            }

            // 5) loop: handle "chunk index X", "next", "close"
            while (true) {
                line = readLine(in);
                if (line == null) {
                    System.out.println("Tracker: peer closed connection");
                    break;
                }

                if (line.startsWith("chunk index")) {
                    // client is requesting a chunk
                    String[] parts = line.split("\\s+");
                    if (parts.length < 3) {
                        System.out.println("Tracker: malformed 'chunk index' line: " + line);
                        return;
                    }
                    int index = Integer.parseInt(parts[2]);
                    sendChunk(out, index);

                    // After sending chunk, we expect either "next" or "close"
                    String ctl = readLine(in);
                    if (ctl == null) {
                        System.out.println("Tracker: peer closed after chunk");
                        break;
                    }
                    if ("close".equals(ctl)) {
                        System.out.println("Tracker: peer sent 'close'");
                        break;
                    }
                    if (!"next".equals(ctl)) {
                        System.out.println("Tracker: expected 'next' or 'close', got " + ctl);
                        break;
                    }
                    // if "next", loop and wait for another "chunk index ..."
                } else if ("close".equals(line)) {
                    System.out.println("Tracker: peer sent 'close'");
                    break;
                } else {
                    // ignore unknown lines for robustness
                    System.out.println("Tracker: ignoring line: " + line);
                }
            }

        } catch (IOException e) {
            System.out.println("Tracker: error with peer - " + e.getMessage());
        } finally {
            try { socket.close(); } catch (IOException ignored) {}
        }
    }

    private void sendChunk(OutputStream out, int index) throws IOException {
        Path chunkPath = chunkDir.resolve("chunk_" + index + ".dat");
        if (!Files.exists(chunkPath)) {
            System.out.println("Tracker: chunk file missing for index " + index);
            sendLine(out, "chunk size 0");
            return;
        }

        long sizeLong = Files.size(chunkPath);
        int size = (int) sizeLong; // safe: chunks are at most CHUNK_SIZE
        sendLine(out, "chunk size " + size);

        try (InputStream inChunk = Files.newInputStream(chunkPath)) {
            byte[] buffer = new byte[1000];
            int sentSoFar = 0;
            while (sentSoFar < size) {
                int toRead = Math.min(buffer.length, size - sentSoFar);
                int n = inChunk.read(buffer, 0, toRead);
                if (n == -1) break;
                out.write(buffer, 0, n);
                sentSoFar += n;
            }
        }
        out.flush();
        System.out.println("Tracker: sent chunk " + index + " (" + sizeLong + " bytes)");
    }

    /* ========== Utilities ========== */

    // Split file into fixed-size chunk files on disk
    private int splitIntoChunks(Path src, int chunkSize) throws IOException {
        int idx = 0;
        byte[] buffer = new byte[chunkSize];

        try (InputStream in = Files.newInputStream(src)) {
            int read;
            while ((read = in.read(buffer)) != -1) {
                Path chunkPath = chunkDir.resolve("chunk_" + idx + ".dat");
                try (OutputStream out = Files.newOutputStream(chunkPath)) {
                    out.write(buffer, 0, read);
                }
                idx++;
            }
        }
        return idx;
    }

    /* ========== Main (no command line args) ========== */

    public static void main(String[] args) throws Exception {
        Group_Tracker tracker = new Group_Tracker(FILE_TO_SHARE);
        tracker.startServer(TRACKER_PORT);
    }
}
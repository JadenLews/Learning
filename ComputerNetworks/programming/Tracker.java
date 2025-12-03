import java.io.*;
import java.net.*;
import java.nio.file.*;
import java.util.ArrayList;
import java.util.List;

public class Tracker {

    private static final int CHUNK_SIZE = 10 * 1024; // 10 kB

    private final String fileName;
    private final Path chunkDir;
    private final int totalChunks;

    public Tracker(String filePath) throws IOException {
        Path path = Paths.get(filePath);
        this.fileName = path.getFileName().toString();

        this.chunkDir = path.getParent().resolve(fileName + "_chunks");
        Files.createDirectories(chunkDir);

        // chunk files on disk 
        this.totalChunks = splitIntoChunks(path, CHUNK_SIZE);
        System.out.println("Tracker: file=" + fileName + ", total chunks=" + totalChunks);
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
            if (ch != '\r'){
                sb.append((char) ch);
            }
        }
        if (sb.length() == 0 && ch == -1) {
            return null; // end of file
        }
        return sb.toString();
    }

    // STARTUP

    public void startServer(int port) throws IOException {
        ServerSocket serverSocket = new ServerSocket(port);
        System.out.println("Tracker listening on TCP port " + port);

        while (true) {
            Socket peerSocket = serverSocket.accept();
            System.out.println("Tracker: peer connected from " + peerSocket.getRemoteSocketAddress());
            new Thread(() -> handlePeer(peerSocket)).start();
        }
    }

    // PROTOCOL HANDLING

    private void handlePeer(Socket socket) {
        try {
            InputStream in = socket.getInputStream();
            OutputStream out = socket.getOutputStream();

            // 1 rdy / rdy
            String line = readLine(in);
            if (!"rdy".equals(line)) {
                System.out.println("Tracker: expected 'rdy' got " + line);
                return;
            }
            sendLine(out, "rdy");

            // 2 file name exchange
            sendLine(out, fileName);
            line = readLine(in);
            if (!fileName.equals(line)) {
                System.out.println("Tracker: expected file name '" + fileName + "', got '" + line + "'");
                return;
            }

            // 3. # of chunks
            sendLine(out, "" + totalChunks);
            line = readLine(in);
            if (!line.equals("" + totalChunks)) {
                System.out.println("Tracker: expected '# of chunks " + totalChunks + "', got " + line);
                return;
            }

            // 4. rdyD
            line = readLine(in);
            if (!"rdyD".equals(line)) {
                System.out.println("Tracker: expected 'rdyD', got " + line);
                return;
            }

            // 5. loop chunk index -> next -> close
            while ((line = readLine(in)) != null) {
                if (line.startsWith("chunk index")) {
                    int index = Integer.parseInt(line.split("\\s+")[2]);
                    handleChunkRequest(index, out);
                } else if ("next".equals(line)) {
                    continue;
                } else if ("close".equals(line)) {
                    System.out.println("Tracker: closed connection");
                    break;
                } else {
                    System.out.println("Tracker: wrong command " + line);
                }
            }

        } catch (IOException e) {
            System.out.println("Tracker: error peer - " + e.getMessage());
        } finally {
            try { socket.close(); } catch (IOException ignored) {}
        }
    }

    private void handleChunkRequest(int index, OutputStream out) throws IOException {
        if (index < 0 || index >= totalChunks) {
            System.out.println("Tracker: invalid chunk index " + index);
            sendLine(out, "chunk size 0");
            return;
        }

        Path chunkPath = chunkDir.resolve("chunk_" + index + ".dat");
        if (!Files.exists(chunkPath)) {
            System.out.println("Tracker: chunk file missing for index " + index);
            sendLine(out, "chunk size 0");
            return;
        }

        long size = Files.size(chunkPath);
        sendLine(out, "chunk size " + size);

        // give this chunk file to peer
        try (InputStream in = Files.newInputStream(chunkPath)) {
            byte[] buffer = new byte[8192];
            int n;
            while ((n = in.read(buffer)) != -1) {
                out.write(buffer, 0, n);
            }
        }
        out.flush();

        System.out.println("Tracker: sent chunk " + index + " (" + size + " bytes)");
    }

    // UTILITIES 
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

    // MAIN 

    // How to run: java Tracker <port> <filePath>
    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("java Tracker <port> <filePath>");
            System.exit(1);
        }
        int port = Integer.parseInt(args[0]);
        String filePath = args[1];

        Tracker tracker = new Tracker(filePath);
        tracker.startServer(port);
    }
}
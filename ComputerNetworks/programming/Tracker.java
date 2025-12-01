import java.io.*;
import java.net.*;
import java.nio.file.*;
import java.util.ArrayList;
import java.util.List;

public class Tracker {

    private static final int CHUNK_SIZE = 10 * 1024; // 10 KB

    private final String fileName;
    private final List<byte[]> chunks;

    public Tracker(String filePath) throws IOException {
        Path path = Paths.get(filePath);
        this.fileName = path.getFileName().toString();
        this.chunks = splitIntoChunks(Files.readAllBytes(path), CHUNK_SIZE);
        System.out.println("Tracker: file=" + fileName +
                ", totalChunks=" + chunks.size());
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

    /* ========== STARTUP ========= */

    public void startServer(int port) throws IOException {
        ServerSocket serverSocket = new ServerSocket(port);
        System.out.println("Tracker listening on TCP port " + port);

        while (true) {
            Socket peerSocket = serverSocket.accept();
            System.out.println("Tracker: peer connected from "
                    + peerSocket.getRemoteSocketAddress());
            new Thread(() -> handlePeer(peerSocket)).start();
        }
    }

    /* ========== PROTOCOL HANDLING (CLIENT–SERVER DIAGRAM) ========= */

    private void handlePeer(Socket socket) {
        try {
            InputStream in = socket.getInputStream();
            OutputStream out = socket.getOutputStream();

            // 1. handshake: rdy / rdy
            String line = readLine(in);
            if (!"rdy".equals(line)) {
                System.out.println("Tracker: expected 'rdy', got " + line);
                return;
            }
            sendLine(out, "rdy");

            // 2. fName exchange
            line = readLine(in);
            if ("fName".equals(line)) {
                sendLine(out, "fName " + fileName);
            } else {
                System.out.println("Tracker: expected 'fName', got " + line);
                return;
            }

            // 3. # of chunks
            line = readLine(in);
            if ("# of chunks".equals(line)) {
                sendLine(out, "# of chunks " + chunks.size());
            } else {
                System.out.println("Tracker: expected '# of chunks', got " + line);
                return;
            }

            // 4. rdyD
            line = readLine(in);
            if (!"rdyD".equals(line)) {
                System.out.println("Tracker: expected 'rdyD', got " + line);
                return;
            }

            // 5. loop: chunk index / next / close
            while ((line = readLine(in)) != null) {
                if (line.startsWith("chunk index")) {
                    int index = Integer.parseInt(line.split("\\s+")[2]);
                    handleChunkRequest(index, out);
                } else if ("next".equals(line)) {
                    continue;
                } else if ("close".equals(line)) {
                    System.out.println("Tracker: peer closed connection");
                    break;
                } else {
                    System.out.println("Tracker: unknown command: " + line);
                }
            }

        } catch (IOException e) {
            System.out.println("Tracker: error with peer - " + e.getMessage());
        } finally {
            try { socket.close(); } catch (IOException ignored) {}
        }
    }

    private void handleChunkRequest(int index, OutputStream out) throws IOException {
        if (index < 0 || index >= chunks.size()) {
            System.out.println("Tracker: invalid chunk index " + index);
            sendLine(out, "chunk size 0");
            return;
        }

        byte[] chunk = chunks.get(index);
        sendLine(out, "chunk size " + chunk.length);
        out.write(chunk);
        out.flush();

        System.out.println("Tracker: sent chunk " + index + " (" + chunk.length + " bytes)");
    }


    /* ========== UTILITIES ========= */

    private static List<byte[]> splitIntoChunks(byte[] data, int chunkSize) {
        List<byte[]> result = new ArrayList<>();
        int offset = 0;
        while (offset < data.length) {
            int end = Math.min(offset + chunkSize, data.length);
            int len = end - offset;
            byte[] chunk = new byte[len];
            System.arraycopy(data, offset, chunk, 0, len);
            result.add(chunk);
            offset = end;
        }
        return result;
    }

    /* ========= MAIN ========= */

    // Usage: java Tracker <port> <filePath>
    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: java Tracker <port> <filePath>");
            System.exit(1);
        }
        int port = Integer.parseInt(args[0]);
        String filePath = args[1];

        Tracker tracker = new Tracker(filePath);
        tracker.startServer(port);
    }
}
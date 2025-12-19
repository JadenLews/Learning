import java.io.*;
import java.net.*;
import java.util.*;
//java Client <MyPort> <NeighborIP> <NeighborPort> <MyID> <Total>
import java.util.concurrent.atomic.AtomicIntegerArray;

public class Jaden {
    private static int myPort;
    private static String neighborHost;
    private static int neighborPort;
    private static int myID;
    private static int totalClients;
    private static final String TRACKER_HOST = "172.20.10.3";
    private static final int TRACKER_PORT = 12000;
    private static AtomicIntegerArray myBitfield;
    private static volatile int chunksOwnedCount = 0;
    private static int totalChunksInNetwork = 0;
    private static String myDir;
    private static String finalFileName = "output.dat";
    public static void main(String[] args) {
        if (args.length < 5) {
            System.out.println("Usage: java Client <MyPort> <NeighborHost> <NeighborPort> <MyID> <TotalClients>");
            return;
        }
        myPort = Integer.parseInt(args[0]);
        neighborHost = args[1];
        neighborPort = Integer.parseInt(args[2]);
        myID = Integer.parseInt(args[3]);
        totalClients = Integer.parseInt(args[4]);

        myDir = "peer_" + myPort;
        new File(myDir).mkdir();

        System.out.println("Client " + myID + " Started on Port " + myPort);
        downloadFromTracker();
        new Thread(() -> startServer()).start();
        try { Thread.sleep(1000); } catch (InterruptedException e) {}
        new Thread(() -> startClient()).start();
        while(true) {
            try { Thread.sleep(10000); } catch (InterruptedException e) {}
        }
    }

    // helper
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
        if (sb.length() == 0 && ch == -1) return null;
        return sb.toString();
    }

    private static void downloadFromTracker() {
        System.out.println("Tracker is connecting");
        try (Socket socket = new Socket(TRACKER_HOST, TRACKER_PORT)) {
            InputStream in = socket.getInputStream();
            OutputStream out = socket.getOutputStream();

            sendLine(out, "rdy");
            if (!"rdy".equals(readLine(in))) throw new IOException("Tracker not ready");

            finalFileName = readLine(in);
            sendLine(out, finalFileName);

            String chunksStr = readLine(in);
            totalChunksInNetwork = Integer.parseInt(chunksStr);
            sendLine(out, chunksStr);

            myBitfield = new AtomicIntegerArray(totalChunksInNetwork);
            sendLine(out, "rdyD");

            // Logic: Clients download a subset (ID % Total)
            for (int i = 0; i < totalChunksInNetwork; i++) {
                if ((i % totalClients) == myID) {
                    sendLine(out, "chunk index " + i);
                    String sizeLine = readLine(in);
                    if (sizeLine == null) break;
                    long size = Long.parseLong(sizeLine.split("\\s+")[2]);
                    if (size > 0) {
                        byte[] buffer = new byte[(int)size];
                        int bytesRead = 0;
                        while(bytesRead < size) {
                            int res = in.read(buffer, bytesRead, (int)size - bytesRead);
                            if(res == -1) break;
                            bytesRead += res;
                        }
                        saveChunk(i, buffer);
                        sendLine(out, "next");
                    }
                }
            }
            sendLine(out, "close");
            System.out.println("Tracker download finished. Owns " + chunksOwnedCount);
        } catch (IOException e) { e.printStackTrace(); }
    }

    // --- P2P SERVER (Updated to Text Protocol) ---
    private static void startServer() {
        try (ServerSocket serverSocket = new ServerSocket(myPort)) {
            while (true) {
                Socket clientSocket = serverSocket.accept();
                new Thread(new UploadHandler(clientSocket)).start();
            }
        } catch (IOException e) { e.printStackTrace(); }
    }

    static class UploadHandler implements Runnable {
        private Socket socket;
        public UploadHandler(Socket s) { this.socket = s; }

        @Override
        public void run() {
            try (InputStream in = socket.getInputStream();
                 OutputStream out = socket.getOutputStream()) {

                while(true) {
                    String msg = readLine(in);
                    if (msg == null) break;

                    if(msg.equals("close")) break;
                    if (!msg.equals("rdy")) continue;
                    sendLine(out, "rdy");

                    msg = readLine(in);
                    if (msg != null && msg.equals("chunkIDList")) {
                        List<Integer> ownedIDs = new ArrayList<>();
                        for(int i = 0; i < totalChunksInNetwork; i++) {
                            if (myBitfield.get(i) == 1) ownedIDs.add(i);
                        }
                        // Text Protocol: "LIST <size>" -> IDs -> "END"
                        sendLine(out, "LIST " + ownedIDs.size());
                        for (Integer id : ownedIDs) sendLine(out, String.valueOf(id));
                        sendLine(out, "END");
                    }

                    msg = readLine(in);
                    if (msg == null || !msg.equals("rdyD")) continue;

                    while (true) {
                        msg = readLine(in);
                        if (msg == null || msg.equals("done") || msg.equals("close")) break;

                        if (msg.startsWith("chunk index ")) {
                            int reqId = Integer.parseInt(msg.split(" ")[2]);
                            if(reqId < totalChunksInNetwork && myBitfield.get(reqId) == 1) {
                                File f = new File(myDir, "chunk_" + reqId + ".dat");
                                if (f.exists()) {
                                    byte[] data = java.nio.file.Files.readAllBytes(f.toPath());
                                    // Text Protocol: "chunk size <bytes>"
                                    sendLine(out, "chunk size " + data.length);
                                    out.write(data);
                                    out.flush();

                                    // *** LOGGING ADDED HERE ***
                                    System.out.println("server Sent chunk " + reqId);

                                } else { sendLine(out, "chunk size 0"); }
                            } else { sendLine(out, "chunk size 0"); }
                        }
                    }
                }
            } catch (IOException e) {}
        }
    }

    // client
    private static void startClient() {
        while (chunksOwnedCount < totalChunksInNetwork) {
            try {
                Thread.sleep(1000);
                try (Socket socket = new Socket(neighborHost, neighborPort);
                     OutputStream out = socket.getOutputStream();
                     InputStream in = socket.getInputStream()) {
                    while (chunksOwnedCount < totalChunksInNetwork) {
                        sendLine(out, "rdy");
                        String resp = readLine(in);
                        if (resp == null || !resp.equals("rdy")) break;
                        sendLine(out, "chunkIDList");
                        // Parse List: "LIST n" -> lines -> "END"
                        String listHeader = readLine(in);
                        if (listHeader == null || !listHeader.startsWith("LIST")) break;
                        List<Integer> neighborHas = new ArrayList<>();
                        while(true) {
                            String idLine = readLine(in);
                            if(idLine == null || idLine.equals("END")) break;
                            neighborHas.add(Integer.parseInt(idLine));
                        }
                        List<Integer> missing = new ArrayList<>();
                        for(Integer id : neighborHas) {
                            if(myBitfield.get(id) == 0) missing.add(id);
                        }
                        sendLine(out, "rdyD");
                        if (missing.isEmpty()) {
                            sendLine(out, "done");
                            Thread.sleep(1000);
                            continue;
                        }
                        System.out.println("Download Peer Found " + missing.size() + " new chunks.");
                        for (int id : missing) {
                            sendLine(out, "chunk index " + id);
                            String sizeLine = readLine(in); // "chunk size X"
                            if (sizeLine == null) break;

                            int size = Integer.parseInt(sizeLine.split("\\s+")[2]);
                            if(size > 0) {
                                byte[] data = new byte[size];
                                int read = 0;
                                while(read < size) {
                                    int r = in.read(data, read, size - read);
                                    if(r == -1) break;
                                    read += r;
                                }
                                saveChunk(id, data);
                            }
                        }
                        sendLine(out, "done");
                    }
                }
            } catch (Exception e) {
                // e.printStackTrace();
            }
        }
        //  combine
        System.out.println("downlad  complete.");
        combineChunks();
        System.out.println("File reassembled.");
    }

    private static void saveChunk(int id, byte[] data) throws IOException {
        if (myBitfield.get(id) == 1) return;
        try (FileOutputStream fos = new FileOutputStream(new File(myDir, "chunk_" + id + ".dat"))) {
            fos.write(data);
        }
        myBitfield.set(id, 1);
        synchronized(Jaden.class) { chunksOwnedCount++; }
    }

    private static void combineChunks() {
        File outputFile = new File(myDir, finalFileName);
        try (FileOutputStream fos = new FileOutputStream(outputFile)) {
            for (int i = 0; i < totalChunksInNetwork; i++) {
                File f = new File(myDir, "chunk_" + i + ".dat");
                if (f.exists()) java.nio.file.Files.copy(f.toPath(), fos);
            }
            System.out.println("SUCCESS " + finalFileName);
        } catch (IOException e) { e.printStackTrace(); }
    }
}

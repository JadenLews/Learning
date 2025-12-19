import java.io.*;
import java.net.*;
import java.nio.file.Files;
import java.util.*;
import java.util.concurrent.*;

public class Group_Peer {

    private String TRACKER_IP;
    private int TRACKER_PORT;
    private String UPLOAD_NEIGHBOR_IP;
    private int UPLOAD_NEIGHBOR_PORT;
    private String DOWNLOAD_NEIGHBOR_IP;
    private int DOWNLOAD_NEIGHBOR_PORT;
    private String fileName;
    private int totalChunks = 250;

    private final Set<Integer> ownedChunks = ConcurrentHashMap.newKeySet();
    private final Map<Integer, File> chunkFiles = new ConcurrentHashMap<>();

    public static void main(String[] args) throws Exception {
        if (args.length < 8) {
            System.out.println("Usage: java Peer <TRACKER_IP> <TRACKER_PORT> <UPLOAD_NEIGHBOR_IP> <UPLOAD_NEIGHBOR_PORT> <DOWNLOAD_NEIGHBOR_IP> <DOWNLOAD_NEIGHBOR_PORT> <fileName> <totalChunks>");
            return;
        }

        Group_Peer peer = new Group_Peer();

        //Init parameters
        peer.TRACKER_IP = args[0];
        peer.TRACKER_PORT = Integer.parseInt(args[1]);
        peer.UPLOAD_NEIGHBOR_IP = args[2];
        peer.UPLOAD_NEIGHBOR_PORT = Integer.parseInt(args[3]);
        peer.DOWNLOAD_NEIGHBOR_IP = args[4];
        peer.DOWNLOAD_NEIGHBOR_PORT = Integer.parseInt(args[5]);
        peer.fileName = args[6];
        peer.totalChunks = Integer.parseInt(args[7]);

        peer.start();
    }

    private void start() throws Exception {
        //Download initial chunks
        downloadChunksFromTracker();

        //Start upload server thread
        new Thread(new UploadServer()).start();

        //Start download client thread
        new Thread(new DownloadClient()).start();
    }

    private void downloadChunksFromTracker() throws IOException {
        Socket sock = new Socket(TRACKER_IP, TRACKER_PORT);
        InputStream is = sock.getInputStream();
        OutputStream os = sock.getOutputStream();
        
        //rdy
        sendLine(os, "rdy");
        String line = readLine(is);
        if (!"rdy".equals(line)) throw new IOException("Handshake failed");

        //Send filename
        String trackerFileName = readLine(is);
        sendLine(os, trackerFileName);  // echo filename back

        //Send totalChunks
        String chunksStr = readLine(is);
        this.totalChunks = Integer.parseInt(chunksStr);
        sendLine(os, chunksStr);  // echo chunk count back

        //rdyD
        sendLine(os, "rdyD");

        //Request each chunk by index
        for (int i = 0; i < totalChunks; i++) {
            sendLine(os, "chunk index " + i);

            String sizeLine = readLine(is);
            String[] parts = sizeLine.split("\\s+");
            int chunkSizeInt = Integer.parseInt(parts[2]);

            System.out.println("Downloading chunk " + i + " of size " + chunkSizeInt);

            byte[] chunkData = new byte[chunkSizeInt];
            int bytesRead = 0;
            while (bytesRead < chunkSizeInt) {
                int r = is.read(chunkData, bytesRead, chunkSizeInt - bytesRead);
                if (r == -1) throw new IOException("Unexpected EOF");
                bytesRead += r;
            }

            File chunkFile = new File("chunk_" + i);
            Files.write(chunkFile.toPath(), chunkData);
            ownedChunks.add(i);
            chunkFiles.put(i, chunkFile);
            System.out.println("Downloaded chunk " + i);

            sendLine(os, "next");
        }

        sendLine(os, "close");
        sock.close();
        System.out.println("Initial chunks downloaded from tracker: " + ownedChunks.size());
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
        if (sb.length() == 0 && ch == -1) return null;
        return sb.toString();
    }


    //Upload neighbor server thread
    private class UploadServer implements Runnable {

        public void run() {

            try (ServerSocket ss = new ServerSocket(UPLOAD_NEIGHBOR_PORT)) {

                System.out.println("Upload server listening on port " + UPLOAD_NEIGHBOR_PORT);

                while (true) {

                    Socket client = ss.accept();
                    new Thread(() -> handleUploadClient(client)).start();
                }

            } catch (IOException e) { e.printStackTrace(); }
        }

        private void handleUploadClient(Socket sock) {


            try {

                BufferedReader in = new BufferedReader(new InputStreamReader(sock.getInputStream()));
                PrintWriter out = new PrintWriter(sock.getOutputStream(), true);
                OutputStream os = sock.getOutputStream();

                //Send owned chunk list to neighbor
                System.out.println("Sending owned chunk list " + ownedChunks + " to upload neighbor : " + UPLOAD_NEIGHBOR_PORT);
                out.println(String.join(" ", ownedChunks.stream().map(String::valueOf).toArray(String[]::new)));

                String line;
                while ((line = in.readLine()) != null) {
                    if ("close".equals(line)) break;
                    int requestedChunk = Integer.parseInt(line);
                    System.out.println("Upload neighbor requested chunk " + requestedChunk);
                    File chunkFile = chunkFiles.get(requestedChunk);
                    if (chunkFile != null) {
                        byte[] data = Files.readAllBytes(chunkFile.toPath());
                        out.println(data.length);
                        System.out.println("Sending chunk " + requestedChunk + " of size " + data.length);
                        os.write(data);
                        os.flush();
                        System.out.println("Sent chunk " + requestedChunk);
                    } else {
                        out.println("ERROR");
                    }
                }

                sock.close();
                System.out.println("Upload connection closed");

            } catch (IOException e) { e.printStackTrace(); }
        }
    }

    // Download neighbor client thread
    private class DownloadClient implements Runnable {

        public void run() {

            while (ownedChunks.size() < totalChunks) {

                try (Socket sock = new Socket(DOWNLOAD_NEIGHBOR_IP, DOWNLOAD_NEIGHBOR_PORT)) {

                    BufferedReader in = new BufferedReader(new InputStreamReader(sock.getInputStream()));
                    PrintWriter out = new PrintWriter(sock.getOutputStream(), true);
                    InputStream is = sock.getInputStream();

                    //Get neighbor owned chunk list
                    String chunkList = in.readLine();
                    Set<Integer> neighborChunks = new HashSet<>();
                    for (String s : chunkList.split(" ")) {
                        if (!s.isEmpty())
                            neighborChunks.add(Integer.parseInt(s));
                    }

                    System.out.println("Neighbor has chunks: " + neighborChunks);

                    //Identify missing chunks
                    Set<Integer> needed = new HashSet<>(neighborChunks);
                    needed.removeAll(ownedChunks);
                    System.out.println("Need to download chunks: " + needed);

                    for (int chunkId : needed) {

                        out.println(chunkId);
                        int size = Integer.parseInt(in.readLine());
                        byte[] data = new byte[size];
                        int read = 0;
                        while (read < size) {
                            int n = is.read(data, read, size - read);
                            if (n == -1) throw new IOException("Stream ended unexpectedly");
                            read += n;
                        }

                        File chunkFile = new File("chunk_" + chunkId);
                        Files.write(chunkFile.toPath(), data);
                        ownedChunks.add(chunkId);
                        chunkFiles.put(chunkId, chunkFile);
                        System.out.println("Downloaded chunk " + chunkId);
                    }

                    out.println("close");
                    Thread.sleep(2000);
                   
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }

            System.out.println("All chunks downloaded.");
            try {
                mergeChunks();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private void mergeChunks() throws IOException {
        FileOutputStream fos = new FileOutputStream("reassembled_" + fileName);
        for (int i = 0; i < totalChunks; i++) {
            File chunkFile = chunkFiles.get(i);
            if (chunkFile != null)
                Files.copy(chunkFile.toPath(), fos);
            else
                System.out.println("Missing chunk during merge: " + i);
        }
        fos.close();
        System.out.println("File reassembled");
    }
}
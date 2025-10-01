import java.io.*;
import java.net.*;

public class ServerTCP {
    private ServerSocket serverSocket;

    public void start(int port) throws IOException {
        serverSocket = new ServerSocket(port);
        System.out.println("ServerTCP Listening on port " + port);
        while (true) {
            try (Socket clientSocket = serverSocket.accept();
                 BufferedReader in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
                 PrintWriter out = new PrintWriter(clientSocket.getOutputStream(), true)) {

                String greeting = in.readLine(); 
                System.out.println("ServerTCP Received: " + greeting);
            }
        }
    }

    public static void main(String[] args) throws IOException {
        int port = (args.length > 0) ? Integer.parseInt(args[0]) : 6666;
        new ServerTCP().start(port);
    }
}
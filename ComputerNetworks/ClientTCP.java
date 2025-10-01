import java.io.*;
import java.net.*;

public class ClientTCP {
    private Socket clientSocket;
    private PrintWriter out;
    private BufferedReader in;

    public void startConnection(String ip, int port) throws IOException {
        clientSocket = new Socket(ip, port);
        out = new PrintWriter(clientSocket.getOutputStream(), true);
        in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
    }

    public String sendMessage(String msg) throws IOException {
        out.println(msg); 
        return in.readLine(); 
    }

    public void stopConnection() throws IOException {
        in.close();
        out.close();
        clientSocket.close();
    }

    public static void main(String[] args) throws IOException {
        String host = (args.length > 0) ? args[0] : "127.0.0.1";
        int port = (args.length > 1) ? Integer.parseInt(args[1]) : 6666;
        String message = (args.length > 2) ? args[2] : "hello server";

        ClientTCP client = new ClientTCP();
        client.startConnection(host, port);
        String resp = client.sendMessage(message);
        System.out.println("ClientTCP Server replied: " + resp);
        client.stopConnection();
    }
}
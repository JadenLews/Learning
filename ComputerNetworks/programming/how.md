/Users/jaden/projects/Learning/ComputerNetworks/programming/HW5.pdf

Argument
Description
Example
<port>
TCP port number to listen for peers connecting
8000
<filePath>
Full path to the file you want to split into chunks
/home/user/testfile.pdf

Argument
Description
Example
<peerId>
Your name/identifier (just a string used in logs)
peer1
<trackerHost>
IP address or hostname of tracker
localhost, or 192.168.1.23
<trackerPort>
TCP port that tracker is listening on
8000
<uploadPort>
This peer’s own server port (for neighbors to connect to)
9001
<downloadNeighborHost>
The IP/hostname of this peer’s download neighbor (ring topology)
localhost, or another machine IP
<downloadNeighborPort>
Port that neighbor is listening on
9002
<chunkDir>
Local directory to store chunks
chunks_p1
<initialChunksFromTracker>
How many random chunks to request from tracker initially
10

javac Tracker.java Peer.java

java Tracker 8000 /Users/jaden/projects/Learning/ComputerNetworks/programming/HW5.pdf
Usage: java Peer <peerId> <trackerHost> <trackerPort> <uploadPort> " +
"<downloadNeighborHost> <downloadNeighborPort> <chunkDir> <initialChunksFromTracker>

java Peer peer1 localhost 8000 9001 172.20.10.9 9005 chunks_p1 20 172.20.10.8

java Peer peer2 localhost 8000 9002 172.20.10.8 9001 chunks_p2 10 172.20.10.3

java Peer peer3 localhost 8000 9003 172.20.10.3 9002 chunks_p3 10 172.20.10.11

java Peer peer4 localhost 8000 9004 172.20.10.11 9003 chunks_p4 10 172.20.10.10

java Peer peer5 localhost 8000 9005 172.20.10.10 9004 chunks_p5 10 172.20.10.9

java Peer peer1 localhost 8000 9001 localhost 9005 chunks_p1 100

java Peer peer2 localhost 8000 9002 localhost 9001 chunks_p2 10

java Peer peer3 localhost 8000 9003 localhost 9002 chunks_p3 10

java Peer peer4 localhost 8000 9004 localhost 9003 chunks_p4 10

java Peer peer5 localhost 8000 9005 localhost 9004 chunks_p5 10

java Group_Peer p1 localhost 8000 9001 localhost 9005 chunks_p1
java Group_Peer p2 localhost 8000 9002 localhost 9001 chunks_p2
java Group_Peer p3 localhost 8000 9003 localhost 9002 chunks_p3
java Group_Peer p4 localhost 8000 9004 localhost 9003 chunks_p4
java Group_Peer p5 localhost 8000 9005 localhost 9004 chunks_p5

java Group_Tracker 8000 /Users/jaden/projects/Learning/ComputerNetworks/programming/HW5.pdf

my ip 172.20.10.3
Alex: 172.20.10.8

java Group_Tracker 8000 /Users/jaden/projects/Learning/ComputerNetworks/programming/HW5.pdf

javac Tracker.java Peer.java revisedClient.java Client.java Group_Peer.java
java Tracker 12000 /Users/jaden/projects/Learning/ComputerNetworks/programming/HW5.pdf
java Client 1001 127.0.0.1 1002 1 11
java revisedClient 2 127.0.0.1 12000 1002 127.0.0.1 1003 tom 11
java Group_peer.java 2

java Jaden 1000 127.0.0.1 1001 0
java Client 1001 127.0.0.1 1002 1 5
java Jaden 1002 127.0.0.1 1003 2
java Jaden 1003 127.0.0.1 1004 3
java Jaden 1004 127.0.0.1 1000 4

<!-- <TRACKER_IP> <TRACKER_PORT> <UPLOAD_NEIGHBOR_IP> <UPLOAD_NEIGHBOR_PORT> <DOWNLOAD_NEIGHBOR_IP> <DOWNLOAD_NEIGHBOR_PORT> <fileName> <totalChunks>
java Group_Peer localhost 12000 localhost 1004 localhost 1001 alex 200 -->




java Jaden 1000 127.0.0.1 1001 0

<MyPort> <NeighborIP> <NeighborPort> <MyID> <Total>
java Client 1001 127.0.0.1 1002 1 5

<MyPort> <NeighborIP> <NeighborPort> <MyID> <Total>
java Jaden 1002 127.0.0.1 1003 2

<peerId> <trackerHost> <trackerPort> <uploadPort> <downloadNeighborHost> <downloadNeighborPort> <chunkDir> <initialChunksFromTracker>
java revisedClient 3 127.0.0.1 12000 1003 127.0.0.1 1004 tom 200

java Jaden 1004 127.0.0.1 1000 4





jaden 172.20.10.3
Tom 172.20.10.11
Raymon 172.20.10.10
antonio 172.20.10.9
alex 172.20.10.8

<MyPort> <NeighborIP> <NeighborPort> <MyID> <Total>

Jaden
java Jaden 1000 172.20.10.11 1001 0 5    

TOM             
java Client 1001 172.20.10.10 1002 1 5    

Raymond
java Client 1002 172.20.10.9 1003 2 5    

Antionio
java Client 1003 172.20.10.8 1004 3 5    

Alex
java Client 1004 172.20.10.3 1000 4 5    
java Jaden 1004 172.20.10.3 1000 4 5   




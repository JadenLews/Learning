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

java Peer peer1 localhost 8000 9001 localhost 9005 chunks_p1 10

java Peer peer2 localhost 8000 9002 localhost 9001 chunks_p2 10

java Peer peer3 localhost 8000 9003 localhost 9002 chunks_p3 10

java Peer peer4 localhost 8000 9004 localhost 9003 chunks_p4 10

java Peer peer5 localhost 8000 9005 localhost 9004 chunks_p5 10
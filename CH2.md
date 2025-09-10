# 2.1 
* Software will run on multiple end systems
* Not writing software to run on network devices
* Network architecture is fixed and provides a specific set of services to applications.
* Application Architecture: Designed by the application developer and dictates how application is structured over the various end systems.
* Client-server architecture:
  * Always a host, called server which serves requests from cliends
  * Web server that takes requests from browsers
  * Clients do not directly communicate
  * Server has a fixed well known address called IP
  * Data center housing a large number of hosts can deal with big loads
* Peer-to-peer architecture:
  * Minimal or no reliance on dedicated servers
  * Exploits direct communication between hosts called peers
  * Devices owned by users
  * Self scalability
  * Challenges in performance, security, reliability
* In operating systems, processes communicate
* When processes are running on same system, they can interprocess communicate
* Processes on different end systems exchange messages across the network
* We typically label one of two processes as client and server.
* Process that initiates communication is client, other is server
* A process sends messages into, and receives from network through a software interface called socket.
* It is also referred to Application Programming Interface (API)
* Internet host is identifies by its IP address
* A destination port is the receiving socket
* Socket is interface between applicaton process and transport layer protocol
* Services that transport layer protocol can offer provide:
  * Reliable data transfer
  * throughput time
  * timing
  * security
* Relaible data transfer: doesnt lose data
* Loss-tolerant applications: can lose a few
* Transport layer can provide: gaurenteed available throughput at some specified rate
* Applications that have throughput requirements are : bandwidth sensitive applications
* Elastic applications: can vary their needs
* Transport layer can also provide timing gaurentees.
* Transport layer can provide an application with more security servicees.
* TCP or UDP provide different set of services:
* TCP: 
  * Includes a connection oriented service and reliable data transfer service.
  * Connection oriented service: Has cleint and server exchange transport layer control information with each other before messages. 
  * Handshake procedure. After handshake TCP connection is said to exist
  * Reliable data transfer service. Processes can rely on TCP to deliver all data sent witout error and in proper order.
  * Includes congestion control mechanism. A service for the general welfare of internet
* UDP:
  * No frills lightweight transport protocol providing minimal services.
  * No handshake.
  * Provides unreliable data transfer service: when a process sends a message int oa UDP socket, no gaurentee message will ever reach.
  * May arrive out of order
  * does not include congestion control
* TLS:
  * A more secure version of TCP inclusing encryption, data integrity, endpoint authentication
* Today internet cannot gaurentee timing or throughput gaurentees


## 2.1.5
* Application Layer Protocol: defines how applications processes running on end systems pass messages to eachother
  * The types of messages exchanged
  * syntax of messages
  * semantics of fields
  * rules for determining when and how a process sends and receives
* Some application layer protocols are specified in RFC and are in public domain.
* Application layer protocol is only one piece of a network application.
* HTTP:
  * Hyper text Transfer Protocol
  * A client program and server program
  * They talk to eachother changing HTTP messages.
  * HTTP defines the structure of the messages and how client and server exchange.
  * Web page consists of objects:
    * Object is a file, like html file, JPEG, javascript, CSS, video clip.
  * for "http://www.someSchool.edu/someDepartment/picture.gif":
    * www.someSchool.edu for a hostname
    * /someDepartment/picture.gif for a path name
    * Server receives requests and responds with HTTP response that contains the objects.
    * HTTP uses TCP as underlying transport protocol
    * Stateless protocol: doesnt remember anything


# 2.2.2
* Non Persistent connections: 
  * Each TCP connection is closed after server sends object
  * Total response time is 2 RTT plus transmission time
* Persistent connections:
* Roud trip time (RTT)
* First line of HTTP request is request line
* Subsequent lines are header lines


















# Class 09/08
* transport Layer: Process to Process communication
* Network Layer: Allow host to host communication, Source to destination communication
* Link Layer: Hop to hop comunication. Enables communication to next device 
* Switch: Only 2 layers. Link Layer device
* Router: Network layer device, 3 layers
* PDU: Protocol Data Unit
* Network Layer: Datagram
* Physical Layer: Bit
* **** Names at each Layer
* 
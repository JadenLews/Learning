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






# Class 09/10
## Processes Communicating
* Process: Program running within a host
* Within same host, two processes communicate using inter-process communication
* Processes in different hosts communicate by exchanging messages
* Client process: Process that initiates communication
* Sercer Process process that waits to be contacted
# Socket
* Process sends/receives messages to/from its socket
  * Socket is analgous to door
    * sned process shoves message out door
    * sending process relies on transport infasructure on other side of door to deliver message to receiving process
    * Two sockets involved: one on each side
  * To receive messages, processes must have identifier
  * Host device has unique 32-bit IP 
  * Identifier includes both IP address and port numbers associated with process on host.
  
## Application Layer protocol defines:
* Types of messages exchanged:
  * Request, response...
* MEssage syntax:
  * What fields in message and how fieldds are delineated
* Message semantics:
  * meaning of information
* Rules for when and how processes send and respond to messages
* Open Protocols:
  * Defined in RFCs, everyone has access to protocol definition
  * allows to interoperability
  * HTTP, SMTP
* Propriertary protocols:
  * Skype, Zoom

## What transport service does app need?
* Data integrity:
  * Some require 100% reliable data transfer
  * Some can tolerate some loss
* Timing:
  * Some require low delay to be effective
* Throughput
  * Some require a min throughput to be effective
  * other can make use of whatever ("elastic")
* Security:
  * Encryption
  * Data integrity
* RDT (Reliable Data Transfer)

## Internet Transport protocol Services
* TCP service:
  * reliable transport between sending and receiving process
  * flow control: sender won’t overwhelm receiver 
  * congestion control: throttle sender when network overloaded
  * connection-oriented: setup required between client and server processes
  * does not provide: timing, minimum throughput guarantee, security
* UDP service:
  * unreliable data transfer between sending and receiving process
  * does not provide: reliability
  * flow control, congestion control, timing, throughput guarantee, security, or connection setup.

## Securing TCP
* Vanilla TCP and UDP sockets:
  * No encryption
  * cleartext passwords sent into socket
* Transport Layer Security (TLS):
  * Provides encrypted TCP connections
  * Data Integrity
  * End point authentification
* TLS implemented in application layer
  * apps use TLS libraries, that use TCP in turn
  *  cleartext sent into “socket” traverse Internet encrypted


# Web and HTTP
* web page consists of objects, each of which can be stored on different Web servers 
* object can be HTML file, JPEG image, Java applet, audio file
* web page consists of base HTML-file which includes several referenced objects, each addressable by a URL, e.g

* HTTP uses TCP:
  * client initiates TCP connection (creates socket) to server, port 80
  * Server accepts TCP connection from client
  * HTTP messages (application Layer protocol messages) exchanged between browser and Web server
  * TCP connection closed
* HTTP is "stateless":
  * Server maintains no information about past client requests
******* HTTP is an applicatoin layer protocol and uses TCP on port 80


## HTTP connections: two types
* Non-persistent HTTP:
  * TCP connection opened
  * At most one object sent over TCP
  * TCP closed
  * Downloading multiple objects takes multiple connections
* Persistent HTTP:
  * TCP connection opened
  * Multiple objects can be sent over single TCP connection between client and server
  * TCP connecton closed



## Persistent HTTP (HTTP 1.1)
Does http allow persistent connections
* Non-persistent HTTP issues:
  * Requires 2 RTTs per object
  * OS overhead for each TCP connection
  * Browsers often open multiple parallel TCP connections to fetch referenced objects in parallel
* Persistent HTTP:
  * Server leaves connection open after sending response
  * Subsequent HTTP mesages between same client/server sent over open connection
  * client sends request as soon as it encounters a referenced object
  * As little as one RTT for all the referenced objects (cutting response time in half)


## HTTP request message
* Two types of HTTP messages:
  * Request
  * response
* HTTP request message
  * ASCII (Human-readable format)

## Other HTTP request messages
* Post method:
  * Web page often includes form input
  * User input sent from client to server in entity body of HTTP POST request message
* Get Method: (for sending data to server)
  * Include user data in URL field of HTTP 
  * Get request message (following a '?')
* Head Method:
  * Requests headers (only) that qould be returned if specified URL were requested with an HTTP GET method
* PUT method: 
  * Uploads new file (object) to server 
  * Completely replaces file that exists at specified URL with content in entity body of POST HTTP request message

## Http response status codes
* Status codes appears in 1st lines in server-to-client response
* Example codes:
  * 200 OK
    * Request succeeded, requested object later in this messgage
  * 301 Moved Permanently
    * Requested object moved, new location specified later in this message (in Location: field)
  * 400 Bad request:
    * Request msg not understoood by server
  * 404 Not found:
    * Requested document not found on server



## Web caches
* Goal: satisfy cleint requests without involving origin server
  * User congifures browser to point to a (local) Web cache
  * Browser sends zll HTTP requests to cahce
    * If object in cache: cache returns object to client
    * else: cache requests object from origin server, caches received object, then returns object to client
## Web caches (aka proxy servers)
* Web cache acts as both server and client
  * server for original requesting cleint
  * client to origin server
* Server tells cache about object's allowable caching in response header

* Why web caching?
  * reduce response time for client request
    * cache is closer to client
  * reduce traffic on an institution's access link
  * Internet is dense with caches
    * enables "poor" content providers to more effectively deliver content



## Browser caching: Conditional GET
* Goal: don’t send object if browser has up-to-date cached version:
  * no object transmission delay (or use of network resources)
  *  client: specify date of browser- cached copy in HTTP request If-modified-since: <date>
  *  server: response contains no object if browser-cached copy is up-to-date: HTTP/1.0 304 Not Modified
  



## HTTP/2
* Key goal: decreased delay in multi-object HTTP requests
* HTTP1.1: introduced multiple, pipelined GETs over single TCP connection
  * server responds in-order (FCFS: first-come-first-served scheduling) to GET requests
  * with FCFS, small object may have to wait for transmission (head-of- line (HOL) blocking) behind large object(s)
  * loss recovery (retransmitting lost TCP segments) stalls object transmission





## Maintaining user/server state: cookies
* Web sites and client browser uses cookies to maintain some state betwen transactions
* four components:
  * cookie header line of HTTP response
  * cookie header line of HTTP response message
  * cookie file kept on users host, manage by users browser
  * back-end database as Website
* What are cookies used for:
  * Authorization
  * shopping cart
  * recommendations
  * user session state(Web email)
* Challenge: How to keep state?
  * At protocol endpoints:
    * Maintain state at sender receiver over multiple transactions
  * In messages:
    * cookies in HTTP messages carry state

## Cookies: tracking a users browsing behavior
* Cookies can be used to
  * track user behavior on a given website (first party cookies)
  *  track user behavior across multiple websites (third party cookies)
  *  racking may be invisible to user:
*  


## Web caches
* Goal: satisfy client requests without involving origin server
* user configures browser to point to a (local) Web cache
* browser sends all HTTP requests to cache
  *  if object in cache: cache returns object to client
  *  else cache requests object from origin server, caches received object, then returns object to client
## Web caches (aka proxy servers)
* Web cache acts as both client and server
* server tells cache about object’s allowable caching in response header:
* Why Web caching?
  * reduce response time for client request
    * cache is closer to client
  * reduce traffic on an institution’s access link

## HTTP/2
* Key goal: decreased delay in multi-object HTTP requests
* HTTP1.1: introduced multiple, pipelined GETs over single TCP connection
*  server responds in-order (FCFS: first-come-first-served scheduling) to GET requests
*  with FCFS, small object may have to wait for transmission (head-of- line (HOL) blocking) behind large object(s)
*  loss recovery (retransmitting lost TCP segments) stalls object transmission
*  



## E-mail: mail servers

* Mail servers
  * Mailbox contains incoming messages for user
  * Message queue of outgoing mail messages
  * SMTP protocol between mail servers to send email messages
    * client sending mail server
    * "server" receiving mail server

QUIZ:
Test on transport mode... protocol, which layer is it on... , how does it send data, what is port number
for SMTP you just need to do a handshake T/F
HTTP 1.0 non persistent, 
PORT NUMBERS!!!

* Email: Three major components
  * User agent
  * Mail server
  * Simple mail transfer protocol: SMTP


## SMTP RFC (5321):
* Uses TCP for relaibly transfer email message from client to server, port 25
* Direct Transfer: Sending server (acting like cleint) to receiving sender
* Three phases of transfer:
  * SMTP handshaking (greeting)
  * SMTP transfer of messages
  * SMTP closure
* Command/response interaction (like HTTP)
  * commands: ASCII text
  * response: status code and phrase


## SMTP: observations
Comparison with HTTP:
* HTTP: client pull
* SMTP: client push
* Both have ASCII command/response interaction, status codes
* HTTP: each object encapsulated in its own response message
* SMTP: multiple objects sent in multipart message
* SMTP uses persistent connections
* SMTP reauires message (header & body) to be in 7-bit ASCII
* SmTP server uses CRLF.CRLF to determine end of message



## retreiving email: mail access protocols
* SMTP: delivery/srotrage of email messages to receivers server
* mail access protocol: retrieval from server
  * IMAP: Internet Mail Access Protocol: messages stored on server , IMAP provides retreival, deletion, foder s of stored messages on server
* HTTP: gmail, hotmail, Yahoo!Mail, etc, provides web-based interface on top of STMP (to send), IMAP (or POP) to retreive e-mail messages






## DNS: Domain Name System
* people many identifiers:
  * SSN, name, passort #
* Internet hosts, routers:
  * IP address (32 bit) - used for addresssing datagrams
  * "name", e.g., cs.umass.edu used by humans
* Q: how to map between IP address and name, and vice versa


* Domain name System (DNS):
  * Distrubuted database implemented in heirarchy of many name servers
  * application layer protocol: hosts DNA servers communicate to resolve names (address/name translation)
    * Note: coer internet function, implemented as application layer protcol
    * complexity at networks "edge"




## DNS: services, structure
* DNS Service:
  * Hostname to IP address translation
  * host aliasing
    * canonical, alias names
  * Mail server aliasing
  * load distribution
    * replicated Web servers: Many IP addresses correspond to one name
* Q: why not centralize DNS?
  * Single point of failure
  * traffic volume
  * distant centralized database
  * maintenance
* A: doesn't scale@
  * Comcast DNS servers alone: 600B DNS queries/day
  * Akamai DNS server alone: 2.2T DNS queries/day'


## Thinking about the DNS
Humongous distributed database:
* ~ billion records, each simple

Handles many trillions of queries/day:
* Many more reads than writes
* performance matters: almost every internet transaction interacts with DNS - msecs count!
  
organizationally, physically decentralized:
* millions of defferent organizations responsible for their records

"bulletproof": reliability, security




## DNS: a distributed, heirarchical database
* Client wants Ip address for www..... 1st approximation:
  * cleint queries root server to find .com DNS servef
  * client queries .com DNS server to get amazon.com DNS server
  * client queries amazon.com DNS server to get IP address for www.amazon.com


DNS sec port number 

127.0.0.1 loop back address





bittorrent unstructured



http quic version 3 uses udp or something
smtp 25, 
top level domain
give me the functio of each layer, and protocol of each layer
transport layer: process to process communication Logical communication******
transport layer segmentized it and gives it to network layer
two major transport layer protocls tcp and udp
logical communication vs physical communicaton
network layer: communiation between hosts
IP for host, port for process : how demultiplexer works

cannot have more than 1 process on UDP

java tcp
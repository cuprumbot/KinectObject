#pragma once

#include<io.h>
#include<stdio.h>
#include<winsock.h>
#include <iostream>
#include <string>
#pragma comment(lib,"ws2_32.lib")

class Socket {
public:
	// Constructor
	Socket();

	// Destructor
	~Socket();

	void createSocket(int);

	SOCKET out_socket;			// outbound socket
	int port;					// port number

	void setMessage(char*);
	void sendMessage();

private:
	WSADATA wsa;
	SOCKET s;
	struct sockaddr_in server, client;
	int c;
	char message[50];
};
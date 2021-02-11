#pragma once

#include "Socket.h"
#include <string.h>
#include<winsock.h>

Socket::Socket() {

}

void Socket::createSocket(int port) {
    printf("\nInitialising Winsock...\n");
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0)
    {
        printf("Failed. Error Code : %d", WSAGetLastError());
        return;
    }
    printf("Initialised.\n");
    
    printf("Creating socket...\n");
    if ((s = socket(AF_INET, SOCK_STREAM, 0)) == INVALID_SOCKET)
    {
        printf("Could not create socket : %d", WSAGetLastError());
    }
    printf("Socket created.\n");

    server.sin_family = AF_INET;
    server.sin_addr.s_addr = INADDR_ANY;
    server.sin_port = htons(port);

    printf("Binding...\n");
    if (bind(s, (struct sockaddr*)&server, sizeof(server)) == SOCKET_ERROR)
    {
        printf("Bind failed with error code : %d", WSAGetLastError());
    }
    puts("Bind done");

    listen(s, 3);

    printf("Waiting for incoming connection...\n");
    c = sizeof(struct sockaddr_in);
    out_socket = accept(s, (struct sockaddr*)&client, &c);
    if (out_socket == INVALID_SOCKET)
    {
        printf("accept failed with error code : %d", WSAGetLastError());
    }
    printf("Connection accepted");
}

Socket::~Socket() {
    closesocket(s);
    WSACleanup();
}

void Socket::setMessage(char* msg) {
    strcpy_s(message, msg);
}

void Socket::sendMessage() {
    send(out_socket, message, strlen(message), 0);
}
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <pthread.h>


#define N_ACTIONS       12
#define N_OBSERVATIONS  45


typedef struct {
  int sockfd;
  pthread_t thread_id;
  struct sockaddr_in robot_addr;
  struct sockaddr_in host_addr;
  float obs[N_OBSERVATIONS];
} UDPRx;



void receive(void *udp_ptr) {
  UDPRx *udp = (UDPRx *)udp_ptr;
  int n;
  float rx_buffer[N_OBSERVATIONS];

  socklen_t len = sizeof(udp->robot_addr);

  // create performance counter
  struct timespec start, end;

  clock_gettime(CLOCK_MONOTONIC, &start);

  while (1) {
    n = recvfrom(udp->sockfd, (void *)rx_buffer, sizeof(float) * N_OBSERVATIONS, MSG_WAITALL, (struct sockaddr *)&udp->robot_addr, &len);
    
    if (n == sizeof(float) * N_OBSERVATIONS) {
      for (int i = 0; i < N_OBSERVATIONS; i++) {
        udp->obs[i] = rx_buffer[i];
      }
    }
    else {
      // sleep for 1ms
      usleep(1000);
    }


    // perfromance
    clock_gettime(CLOCK_MONOTONIC, &end);


    long seconds = end.tv_sec - start.tv_sec;
    long ns = end.tv_nsec - start.tv_nsec;

    // Correct for rollover
    if (start.tv_nsec > end.tv_nsec) {
      --seconds;
      ns += 1000000000;
    }

    clock_gettime(CLOCK_MONOTONIC, &start);

    double freq = 1.0 / (seconds + ns / 1000000000.0);

    // printf("Frequency: %f\n", freq);
  }
}


int initialize(UDPRx *udp, 
    const char *host_ip, unsigned int host_port, 
    const char *robot_ip, unsigned int robot_port
  ) {

  memset(&udp->robot_addr, 0, sizeof(udp->robot_addr));
  memset(&udp->host_addr, 0, sizeof(udp->host_addr));

  udp->host_addr.sin_family = AF_INET;
  udp->host_addr.sin_addr.s_addr = inet_addr(host_ip);
  udp->host_addr.sin_port = htons(host_port);

  udp->robot_addr.sin_family = AF_INET;
  udp->robot_addr.sin_addr.s_addr = inet_addr(robot_ip);
  udp->robot_addr.sin_port = htons(robot_port);

  // Create socket file descriptor
  if ((udp->sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
    printf("[ERROR]: socket creation failed");
    return -1;
  }

  if (bind(udp->sockfd, (const struct sockaddr *)&udp->host_addr, sizeof(udp->host_addr)) < 0 ) {
    printf("[ERROR]: bind failed");
    return -1;
  }

  // Create a thread running the receive() function
  if(pthread_create(&udp->thread_id, NULL, receive, (void *)udp) != 0) {
    perror("Failed to create thread");
    return 1;
  }

  printf("Receive thread created, thread ID: %ld\n", (long)udp->thread_id);

  printf("Server listening on port %d\n", ntohs(udp->host_addr.sin_port));

  // // Wait for the thread to finish
  // pthread_join(udp->thread_id, NULL);

  return 0;
}

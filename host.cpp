#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <thread>
#include <vector>


#define HOST_IP     "192.168.44.101"
#define HOST_PORT   9000

#define ROBOT_IP    "192.168.44.1"
#define ROBOT_PORT  8000


#define N_ACTIONS       12
#define N_OBSERVATIONS  45


std::vector<float> obs(N_OBSERVATIONS);
std::vector<float> acs(N_ACTIONS);
float rx_buffer[1024];


void rxHandler(int sockfd, struct sockaddr_in addr) {
  int n;
  socklen_t len;

  printf("RX Handler started\n");

  while (1) {
    len = sizeof(addr); // Length of client's address

    n = recvfrom(sockfd, (void *)rx_buffer, 1024, MSG_WAITALL, (struct sockaddr *) &addr, &len);
    

    if (n == sizeof(float) * N_ACTIONS) {
      for (int i = 0; i < N_ACTIONS; i++) {
        acs[i] = rx_buffer[i];
      }
      printf("acs: %f %f %f\n", acs[0], acs[1], acs[2]);
    }
  }
}

void txHandler(int sockfd, struct sockaddr_in addr) {
  auto period = std::chrono::milliseconds(10);

  while (1) {
    obs[33] = acs[0];
    obs[34] = acs[1];
    obs[35] = acs[2];

    void *tx_buffer = (void *)obs.data();
    
    sendto(sockfd, tx_buffer, sizeof(float) * N_OBSERVATIONS, MSG_CONFIRM, (const struct sockaddr *) &addr, sizeof(addr));
    
    printf("obs: %f %f %f\n", obs[33], obs[34], obs[35]);

    auto next_time = std::chrono::steady_clock::now() + period;
    std::this_thread::sleep_until(next_time);
  }
}




int main() {
  int sockfd;
  char rx_buffer[1024];
  char tx_buffer[1024] = "Hello from client";

  struct sockaddr_in robot_addr, host_addr;

  memset(&robot_addr, 0, sizeof(robot_addr));
  memset(&host_addr, 0, sizeof(host_addr));

  host_addr.sin_family = AF_INET;
  host_addr.sin_addr.s_addr = inet_addr(HOST_IP);
  host_addr.sin_port = htons(HOST_PORT);

  robot_addr.sin_family = AF_INET;
  robot_addr.sin_addr.s_addr = inet_addr(ROBOT_IP);
  robot_addr.sin_port = htons(ROBOT_PORT);

  // Create socket file descriptor
  if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
    printf("[ERROR]: socket creation failed");
    return -1;
  }

  if (bind(sockfd, (const struct sockaddr *)&robot_addr, sizeof(robot_addr)) < 0 ) {
    printf("[ERROR]: bind failed");
    return -1;
  }

  printf("Server listening on port %d\n", ntohs(robot_addr.sin_port));

  std::thread rxThread(rxHandler, sockfd, robot_addr);
  std::thread txThread(txHandler, sockfd, host_addr);

  while (1) {
    obs[33] = acs[0];
    obs[34] = acs[1];
    obs[35] = acs[2];
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }


  rxThread.join();
  txThread.join();




  return 0;
}

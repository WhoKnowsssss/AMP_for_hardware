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
#define N_ACS_PARAMS    3
#define N_OBSERVATIONS  45

#define N_ACTION_STEPS  1
#define N_OBS_STEPS     8


#define SEND_RATE_HZ    50

typedef struct {
  int sockfd;
  pthread_t thread_id;
  struct sockaddr_in robot_addr;
  struct sockaddr_in host_addr;
  float obs[N_OBSERVATIONS];
} UDPRx;

typedef struct {
  pthread_t thread_id;
  float action_queue[N_ACTION_STEPS][N_ACTIONS];
  float acs_params[N_ACS_PARAMS];
  float observation_history[N_OBS_STEPS+1][N_OBSERVATIONS];
  u_int8_t stepDiffusionFlag;
  u_int8_t newActionFlag;
  float* new_action_queue;
  UDPRx* udp;
} DiffusionWrapper;

void set_udp_pointer(DiffusionWrapper *wrapper, UDPRx *udp) {
  wrapper->udp = udp;
}

void set_new_action_queue(DiffusionWrapper *wrapper, float *new_action_queue) {
  wrapper->new_action_queue = new_action_queue;
}

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

void send_action(void *wrapper_ptr) {
  DiffusionWrapper *wrapper = (DiffusionWrapper *)wrapper_ptr;
  struct timespec next;
  float tx_buffer[N_ACTIONS+N_ACS_PARAMS] = {0}; // Example data, replace with actual data generation logic
  int period_ns = 1000000000 / SEND_RATE_HZ; // Period in nanoseconds

  // Get the current time
  clock_gettime(CLOCK_MONOTONIC, &next);
  printf("[INFO]: Send thread created, thread ID: %ld\n", (long)wrapper->thread_id);
  int idx = 0;
  int global_idx = 0;

  struct timespec start, end;

  clock_gettime(CLOCK_MONOTONIC, &start);

  while (1) {
    // printf("[INFO]: Sending action\n");
    // receive new obs
    memmove(wrapper->observation_history[0], wrapper->observation_history[1], (N_OBSERVATIONS * sizeof(float)) * (N_OBS_STEPS));
    // have a loop to set obs history from 1 to N_OBSERVATIONS
    // wrapper->observation_history[N_OBS_STEPS][0] = global_idx;
    // global_idx += 1;
    // memset(wrapper->observation_history[N_OBS_STEPS], 0, N_OBSERVATIONS * sizeof(float));
    memcpy(wrapper->observation_history[N_OBS_STEPS], wrapper->udp->obs, N_OBSERVATIONS * sizeof(float));

    idx = idx % N_ACTION_STEPS;
    if (idx == N_ACTION_STEPS - 2) {
        wrapper->stepDiffusionFlag = 1;
    }
    wrapper->stepDiffusionFlag = 1;
    // printf("Diffusion Flag: %d, idx: %d\n", wrapper->stepDiffusionFlag, idx);

    if (idx == 0) {
      if (wrapper->newActionFlag == 0){
        printf("Missed Diffusion Step! \n");
      }
      wrapper->newActionFlag = 0;
        memcpy(wrapper->action_queue[0], wrapper->new_action_queue, N_ACTIONS * sizeof(float) * N_ACTION_STEPS);
        // Print action queue
        // for (int i = 0; i < N_ACTION_STEPS; i++) {
        //   printf("Action %d: ", i);
        //   for (int j = 0; j < N_ACTIONS; j++) {
        //     printf("%f ", wrapper->action_queue[i][j]);
        //   }
        //   printf("\n");
        // }
    }

    memcpy(tx_buffer, wrapper->action_queue[idx], N_ACTIONS * sizeof(float));
    memcpy(((float *)tx_buffer) + N_ACTIONS, wrapper->acs_params, N_ACS_PARAMS * sizeof(float));
    // printf("Action Params: ");
    // for (int i = 0; i < N_ACS_PARAMS; i++) {
    //   printf("%f ", tx_buffer[i + N_ACTIONS]);
    // }
    // printf("\n");

    // printf("Sending action: ");
    // for (int i = 0; i < N_ACTIONS; i++) {
    //   printf("%f ", tx_buffer[i]);
    // }
    // printf("\n");
    sendto(wrapper->udp->sockfd, tx_buffer, sizeof(tx_buffer), 0, (const struct sockaddr *)&wrapper->udp->robot_addr, sizeof(wrapper->udp->robot_addr));
    idx += 1;

    // Calculate next send time
    next.tv_nsec += period_ns;
    if (next.tv_nsec >= 1000000000) {
      next.tv_sec++;
      next.tv_nsec -= 1000000000;
    }
    // Wait until next send time
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, NULL);

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

    // printf("Frequency: %f\n", freq,  );
  }
}

void set_diffusion_flag(DiffusionWrapper *wrapper, u_int8_t flag) {
  wrapper->stepDiffusionFlag = flag;
}

int init_diffusion_wrapper(
    DiffusionWrapper *wrapper, UDPRx *udp, float *new_action_queue
  ) {
  memset(wrapper->action_queue, 0, N_ACTION_STEPS * N_ACTIONS * sizeof(float));
  memset(wrapper->observation_history, 0, (N_OBS_STEPS + 1) * N_OBSERVATIONS * sizeof(float));
  wrapper->stepDiffusionFlag = 1;
  wrapper->newActionFlag = 0;

  set_udp_pointer(wrapper, udp);
  wrapper->new_action_queue = new_action_queue;

  // Init Obs History
  for (int i = 0; i < N_OBS_STEPS + 1; i++) {
    memcpy(wrapper->observation_history[i], udp->obs, N_OBSERVATIONS * sizeof(float));
  }

  // Create a thread running the send_action() function
  if(pthread_create(&wrapper->thread_id, NULL, send_action, (void *)wrapper) != 0) {
    perror("[ERROR]: Failed to create send thread");
    return 1;
  }
  return 0;
}

int initialize(UDPRx *udp, 
    const char *host_ip, unsigned int host_port, 
    const char *robot_ip, unsigned int robot_port
  ) {

  memset(&udp->robot_addr, 0, sizeof(udp->robot_addr));
  memset(&udp->host_addr, 0, sizeof(udp->host_addr));
  memset(udp->obs, 0, N_OBSERVATIONS * sizeof(float));

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
    perror("[ERROR]: Failed to create receive thread");
    return 1;
  }

  printf("[INFO]: Receive thread created, thread ID: %ld\n", (long)udp->thread_id);

  printf("[INFO]: Server listening on port %d\n", ntohs(udp->host_addr.sin_port));

  // // Wait for the thread to finish
  // pthread_join(udp->thread_id, NULL);

  return 0;
}

#include <stdio.h>
#include <time.h>
#include <stdlib.h>

int main(){
  int not_synced = 1;
  
  while(not_synced){
    not_synced = system("ntpd -q -n -p 10.0.0.5");
  }
#ifdef VERBOSE
  printf("Synced to server successfully.\n");
#endif
  
  struct timespec start_time, sleep_time, remaining_time;
  clock_gettime(CLOCK_REALTIME, &start_time);
  
  sleep_time.tv_sec = 59 - (start_time.tv_sec % 60) ;
  sleep_time.tv_nsec = 1000000000L - start_time.tv_nsec;

#ifdef VERBOSE
  printf("Waiting for %ld seconds and %ld nanoseconds.\n", sleep_time.tv_sec, sleep_time.tv_nsec);
#endif
  
  nanosleep(&sleep_time , &remaining_time);
  return 0;
}
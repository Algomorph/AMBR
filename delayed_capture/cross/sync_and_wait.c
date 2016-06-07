#include <stdio.h>
#include <time.h>
#include <stdlib.h>

int main(){
  int sync_success = 0;
  
  while(!sync_success){
    sync_success = system("ntpd -q -n -p 10.0.0.5");
  }
  
  struct timespec start_time, sleep_time, remaining_time;
  clock_gettime(CLOCK_REALTIME, &start_time);
  
  sleep_time.tv_sec = 59 - (start_time.tv_sec % 60) ;
  sleep_time.tv_nsec = 1000000000L - start_time.tv_nsec;
  
  nanosleep(&sleep_time , &remaining_time);
  return 0;
}
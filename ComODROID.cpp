#include <iostream>
#include <stdio.h>
#include <fstream>

using namespace std;

int main()
{
  system("echo manual > /sys/devices/platform/odroidu2-fan/fan_mode");
  int value; cout<<"Value | vE(0,255): "; cin>>value;
    
  char exec[64] = "echo 000 > /sys/devices/platform/odroidu2-fan/pwm_duty";
  string str = to_string(value); char const* vlij = str.c_str();
  for(int i=int(strlen(vlij))-1; i>=0; i--){exec[i-strlen(vlij)+8]=vlij[i];}

  system(exec);
    
  cout<<"Value passed"<<endl;
}

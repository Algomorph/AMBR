#time-sync to the host machine
ntpd -q -n -p 10.0.0.5

target_epoch=$(( $(date +%s) - ($(date +%s)%60) + 60 )) 
current_epoch=$(date +%s)
counter=0
while [ $(date +%s) -lt $target_epoch ]; do
    let counter+=1
done
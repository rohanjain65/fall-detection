#!/bin/bash

# Replace with your latest valid session ID
SESSION_ID="wq7568hmbcaumljjz0a1eld0in4cfyux"

# Base URL
BASE_URL="https://rose1.ntu.edu.sg/dataset/actionRecognition/download"

# File counter for s001 to s017
file_index=1
MAX_PARALLEL=4
active=0

for (( id=125; id<=141; id++ )); do
  filename=$(printf "nturgbd_rgb_s%03d.zip" "$file_index")

  curl -L -o "$filename" "$BASE_URL/$id" \
    -H "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7" \
    -H "Accept-Encoding: gzip, deflate, br, zstd" \
    -H "Accept-Language: en-US,en;q=0.9" \
    -H "Connection: keep-alive" \
    -H "Cookie: sessionid=$SESSION_ID" \
    -H "Host: rose1.ntu.edu.sg" \
    -H "Referer: https://rose1.ntu.edu.sg/dataset/actionRecognition/download" \
    -H "Sec-Fetch-Dest: document" \
    -H "Sec-Fetch-Mode: navigate" \
    -H "Sec-Fetch-Site: same-origin" \
    -H "Sec-Fetch-User: ?1" \
    -H "Upgrade-Insecure-Requests: 1" \
    -H "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36" \
    -H 'sec-ch-ua: "Chromium";v="134", "Not:A-Brand";v="24", "Google Chrome";v="134"' \
    -H 'sec-ch-ua-mobile: ?0' \
    -H 'sec-ch-ua-platform: "Windows"' &

  ((file_index++))
  ((active++))

  if (( active >= MAX_PARALLEL )); then
    wait -n
    ((active--))
  fi
done

# Wait for any remaining downloads to finish
wait

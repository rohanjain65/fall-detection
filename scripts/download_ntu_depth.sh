#!/bin/bash

# Replace this with your valid session ID (re-export if expired)
SESSION_ID="ghd57yejo9polepdp6a365k00t41itjw"

# Base URL
BASE_URL="https://rose1.ntu.edu.sg/dataset/actionRecognition/download"

# Start at s005 (id=56) to s017 (id=80), increment by 2
file_index=5
MAX_PARALLEL=4
active=0

for (( id=56; id<=80; id+=2 )); do
  filename=$(printf "nturgbd_depth_masked_s%03d.zip" "$file_index")

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

# Wait for remaining downloads to finish
wait

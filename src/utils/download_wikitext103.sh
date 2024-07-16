#!/bin/bash

URL="https://dax-cdn.cdn.appdomain.cloud/dax-wikitext-103/1.0.1/wikitext-103.tar.gz"
SAVEDIR="data"
FILE="wikitext-103.tar.gz"

# download
if [ -f "$FILE" ];then
    echo "already exist"
else;
    # TODO tarファイルの保存先をオプションで指定できるようにしたい
    wget -O "$SAVEDIR/$FILE" "$URL"
fi

# extract
tar -xzC "$SAVEDIR" -f "$SAVEDIR/$FILE"

# remove `tar`file
rm "$SAVEDIR/$FILE"
#!/bin/bash

URL="https://dax-cdn.cdn.appdomain.cloud/dax-wikitext-103/1.0.1/wikitext-103.tar.gz"

FILE="data/wikitext-103.tar.gz"

if [ -f "$FILE" ];then
    echo "already exist"
else;
    # TODO tarファイルの保存先をオプションで指定できるようにしたい
    wget -O "$FILE" "$URL"
fi


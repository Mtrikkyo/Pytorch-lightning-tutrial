#!/bin/bash

URL="https://dax-cdn.cdn.appdomain.cloud/dax-wikitext-103/1.0.1/wikitext-103.tar.gz"
SAVEDIR="data"
FILE="WikiText103"

if [ -f "$SAVEDIR/$FILE/.gitkeep" ];then
    
    echo "already exist"
else;
    
    # download
    # TODO tarファイルの保存先をオプションで指定できるようにしたい
    wget -O "$SAVEDIR/$FILE.tar.gz" "$URL"
    
    # extract
    tar -xzC "$SAVEDIR" -f "$SAVEDIR/$FILE.tar.gz"
    
    # rename file
    mv "$SAVEDIR/wikitext-103" "$SAVEDIR/$FILE"
    
    # remove `tar`file
    rm "$SAVEDIR/$FILE.tar.gz"
    
    # make .gitkeep file
    touch "$SAVEDIR/$FILE/.gitkeep"
fi
#!/bin/bash

URL="http://www.daviddlewis.com/resources/testcollections/reuters21578/reuters21578.tar.gz"
SAVEDIR="data"
FILE="Reuters21578"

if [ -f "$SAVEDIR/$FILE/.gitkeep" ];then
    
    echo "already exist"
else;
    # make $FILE dir
    mkdir "$SAVEDIR/$FILE"
    
    # download
    # TODO tarファイルの保存先をオプションで指定できるようにしたい
    wget -O "$SAVEDIR/$FILE/$FILE.tar.gz" "$URL"
    
    # extract
    tar -xzC "$SAVEDIR/$FILE" -f "$SAVEDIR/$FILE/$FILE.tar.gz"
    
    # # rename file
    # mv "$SAVEDIR/wikitext-103" "$SAVEDIR/$FILE"
    
    # remove `tar`file
    rm "$SAVEDIR/$FILE/$FILE.tar.gz"
    
    # make .gitkeep file
    touch "$SAVEDIR/$FILE/.gitkeep"
fi
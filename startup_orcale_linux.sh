#!/bin/sh

while getopts k:s: flag
do
    case "${flag}" in
        k) TWITTER_KEY=${OPTARG};;
        s) TWITTER_SECRET_KEY=${OPTARG};;
    esac
done

sudo docker run -p 80:8501 -e TWITTER_KEY=$TWITTER_KEY -e TWITTER_SECRET_KEY=$TWITTER_SECRET_KEY trending-sentiments
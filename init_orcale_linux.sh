#!/bin/sh

while getopts k:s: flag
do
    case "${flag}" in
        k) TWITTER_KEY=${OPTARG};;
        s) TWITTER_SECRET_KEY=${OPTARG};;
    esac
done

sudo dnf update -y
sudo dnf config-manager --add-repo=https://download.docker.com/linux/centos/docker-ce.repo
sudo dnf install git tmux docker-ce -y
sudo systemctl start docker
sudo systemctl enable docker
git clone https://github.com/Dormanator/trending-sentiments
cd trending-sentiments
sudo docker build -t trending-sentiments .
tmux new -t trending-sentiments
sudo docker run -p 80:8501 -e TWITTER_KEY=$TWITTER_KEY -e TWITTER_SECRET_KEY=$TWITTER_SECRET_KEY trending-sentiments
tmux detach
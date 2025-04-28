#!/bin/bash

# Set your Git identity
git config --global user.email "djain2095@gmail.com"
git config --global user.name "sofasogood"

# Set up credential storage
git config --global credential.helper store

echo "Git identity configured!"
echo "The next time you push or pull, enter your personal access token instead of a password."
echo "It will be stored for future use."

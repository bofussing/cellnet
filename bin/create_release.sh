#!/bin/bash

# if on hpc 


path="$1"
version="$2"

# if either path or version are empty
if [ -z "$path" ] || [ -z "$version" ]; then
    echo "Usage: $0 <path to release's results folder> <version tag>"
    exit 1
fi

zip -r release.zip "$path/model_export"
zip -u release.zip release.yml
zip -u release.zip release.py

printf "@ release.yml\n@=environment.yml\n" | zipnote -w release.zip
printf "@ release.py\n@=main.py\n" | zipnote -w release.zip

gh release create -n --target draft "$version" "release.zip"

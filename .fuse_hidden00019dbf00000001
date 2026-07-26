#!/bin/bash

cd /home/sts33/2eme/ODB

git add .

if ! git diff --cached --quiet; then
    git commit -m "Sauvegarde $(date)"
    git push
fi

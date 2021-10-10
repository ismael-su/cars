#!/bin/bash

# Exit in case of error
set -e

# Build and run containers
docker-compose up -d

# Hack to wait for postgres container to be up before running alembic migrations
sleep 5;


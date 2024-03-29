#!/bin/bash
cd ../feature_repos/data
CURRENT_TIME=$(date -u +"%Y-%m-%d")
feast materialize-incremental $CURRENT_TIME
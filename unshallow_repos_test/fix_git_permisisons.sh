#!/bin/bash

cat /shared_workspace_mfs/arthur/coder/unshallow_repos_test/all_repos_shallow.txt | \
xargs -I {} -P 40 bash -c '
    repo_path="$1"
    if [ -d "$repo_path" ]; then
        chmod -R 777 "$repo_path"
        echo "Changed permissions for: $repo_path"
    else
        lowercase_path=$(echo "$repo_path" | tr "[:upper:]" "[:lower:]")
        if [ -d "$lowercase_path" ]; then
            chmod -R 777 "$lowercase_path"
            echo "Changed permissions for: $lowercase_path"
        else
            echo "Path not found: $repo_path"
        fi
    fi
' _ {}

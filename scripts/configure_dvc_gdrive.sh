#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${DVC_GDRIVE_SERVICE_ACCOUNT_JSON:-}" ]]; then
  echo "DVC_GDRIVE_SERVICE_ACCOUNT_JSON not set"
  echo "Set it to the full path of your service account JSON file"
  exit 1
fi

if [[ ! -f "$DVC_GDRIVE_SERVICE_ACCOUNT_JSON" ]]; then
  echo "Service account JSON not found at: $DVC_GDRIVE_SERVICE_ACCOUNT_JSON"
  exit 1
fi

dvc remote modify gdrive use_service_account true
# Use the file path for local DVC
# shellcheck disable=SC2086

dvc remote modify gdrive gdrive_service_account_json_file_path "$DVC_GDRIVE_SERVICE_ACCOUNT_JSON"

echo "Configured DVC to use service account JSON"

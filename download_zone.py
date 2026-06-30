from huggingface_hub import snapshot_download

snapshot_download(repo_id = "heig-vd-geo/GridNet-HD",
                  repo_type = "dataset",
                  local_dir = "data/gridnet_hd",
                  allow_patterns = "t1z5b/*"
                  )
print("Done.")
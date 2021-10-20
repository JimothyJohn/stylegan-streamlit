#!/usr/bin/env python
import requests

url = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqcat-512x512.pkl"

r = requests.get(url)

with open("stylegan2-afhqcat-512x512.pkl", 'wb') as f:
    f.write(r.content) 
    f.close()

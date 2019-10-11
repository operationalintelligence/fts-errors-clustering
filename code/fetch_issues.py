#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Copyright 2018 Luca Clissa
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
"""
Created on Tue Oct  8 10:13:27 2019

@author: Panos Paparrigopoulos
"""

import urllib.request, json, sys

# check arguments for the output filename
if len(sys.argv)==1:
      output_name = "issues.json"
elif len(sys.argv)==2:
      output_name = sys.argv[1]
else:
     raise Exception("More than one argument specified: please provide just the name of the output file.")        

# read issues from Rucio API (all pages)
res = []
with urllib.request.urlopen("http://rucio-opint.web.cern.ch/api/issues/") as url:
    data = json.loads(url.read().decode())
    res.append(data['results'])
page=1
while data['next']:
    with urllib.request.urlopen("http://rucio-opint.web.cern.ch/api/issues/?page="+str(page)) as url:
        data = json.loads(url.read().decode())
        page += 1
        res.append(data['results'])
        
# save results locally
with open(output_name, 'w') as outfile:
    json.dump(res, outfile)
#!/usr/bin/env python3
"""Select a model through the same serialized controller as the webpage."""
import argparse
import http.client
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('--context',type=int,default=4096,choices=[1024,2048,4096,8192])
    args = parser.parse_args()
    connection = http.client.HTTPConnection('127.0.0.1',8080,timeout=360)
    try:
        connection.request('POST','/api/settings',json.dumps({'model_id':args.model,'context':args.context}),
                           {'Content-Type':'application/json'})
        response = connection.getresponse()
        print(response.read().decode())
        if response.status != 200:
            raise SystemExit(1)
    finally:
        connection.close()

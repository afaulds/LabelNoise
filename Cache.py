import json
import tempfile
from os import path
import hashlib


class Cache:

    def process(key, func, *args):
        clean_key = hashlib.md5(str(key).encode()).hexdigest()
        file_name = tempfile.gettempdir() + "/" + clean_key + ".tmp"
        if Cache.__key_exists(file_name):
            return Cache.__read_file(file_name)
        else:
            val = func(*args)
            Cache.__write_file(file_name, val)
            return val

    def __key_exists(file_name):
        return path.exists(file_name)

    def __read_file(file_name):
        with open(file_name, "r") as infile:
            return json.loads(infile.read())

    def __write_file(file_name, val):
        with open(file_name, "w") as outfile:
            outfile.write(json.dumps(val))

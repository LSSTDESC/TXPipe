import hashlib



def array_hash(x):
    b = x.tobytes()
    # We do not need a cryptographic hash here
    return int(hashlib.md5(b).hexdigest(), 16)

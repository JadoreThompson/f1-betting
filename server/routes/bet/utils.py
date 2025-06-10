from r_mutex import LockClient

from config import LOCK_CHANNEL, REDIS_CLIENT

# lock.run() is called by __main__
# - Prevents deadlock errors incase the Pusher 
#   is editing the record(s)
lock: LockClient = LockClient(REDIS_CLIENT, LOCK_CHANNEL, False)
from r_mutex import LockClient

# initialised by __main__
# - Prevents deadlock errors incase the Pusher 
#   is editing the record(s)
lock: LockClient
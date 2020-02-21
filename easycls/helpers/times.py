import time
import datetime

def format_time(timestamp=time.time(), format=r"%Y%m%d_%H%M%S"):
    """
    Return a formatted time string

    Commonly used format codes:

    %Y Year with century as a decimal number. 
    %m Month as a decimal number [01,12]. 
    %d Day of the month as a decimal number [01,31]. 
    %H Hour (24-hour clock) as a decimal number [00,23]. 
    %M Minute as a decimal number [00,59]. 
    %S Second as a decimal number [00,61]. 
    %z Time zone offset from UTC. 
    %a Locale's abbreviated weekday name. 
    %A Locale's full weekday name. 
    %b Locale's abbreviated month name. 
    %B Locale's full month name. 
    %c Locale's appropriate date and time representation. 
    %I Hour (12-hour clock) as a decimal number [01,12]. 
    %p Locale's equivalent of either AM or PM.
    """
    time_str = time.strftime(format, time.localtime(timestamp))
    return time_str


def readable_time(timestamp=time.time()):
    """
    Return a human-friendly string of a timestamp

    e.g.
        timestamp: 1582029001.6709404
        readable time: Tue Feb 18 20:30:01 2020

    Args:
        timestamp: a UNIX timestamp

    Returns:
        str, a human-friendly readable string of the argument timestamp
    """
    time_readable_str = str(time.asctime(time.localtime(timestamp)))
    
    return time_readable_str


def readable_eta(seconds_left):
    """
    Return a human-friendly string of ETA time and left time

    e.g.
        seconds_left:   90090
        ETA:            Wed Feb 19 21:42:19 2020
        Time-left:      1 day, 1:01:30

    Args:
        seconds_left: int or float, seconds left
    
    Returns:
        tuple of str, ETA string and Time-left string, both in human readable form
    """
    eta_time = readable_time(time.time() + seconds_left)
    time_left = datetime.timedelta(seconds=int(seconds_left))

    return str(eta_time), str(time_left)


# Unit Test
if __name__ == "__main__":
    ts = time.time()
    print(f'Timestamp: {ts} -> {readable_time(ts)}')
    print(f'format time: {format_time(ts)}')

    seconds_left = 3600 * 25 + 90
    eta, Tleft = readable_eta(seconds_left)
    print(f'seconds_left: {seconds_left}\tETA: {eta}, in {Tleft}')
from .constants import GrB_Info


class GraphBlasException(Exception):
    pass

last_error_message = None


def GrB_error():
    return last_error_message


def return_error(error, msg=''):
    global last_error_message
    last_error_message = msg
    return error

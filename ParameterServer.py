import json
import threading
import configparser
import logging
import numpy as np
import IPython

from ParameterServer import *
from ParameterServer.ttypes import *
from thrift.Thrift import TType, TMessageType, TException
from thrift.Thrift import TProcessor
from thrift.transport import TSocket
from thrift.protocol import TBinaryProtocol, TProtocol
from thrift.server import TServer
from ParameterServer.constants import *


config = configparser.ConfigParser()
config.read('ps.conf')

HOST = config['DEFAULT']['HOST']
PORT = config['DEFAULT'].getint('PORT')
LEARNING_RATE = config['DEFAULT'].getfloat('LEARNING_RATE')

# 可能随时间变化
learning_rate = LEARNING_RATE

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG)

mutex = threading.Lock()
parameters = {}
gradients = {}


class ParameterServerHandler:
    def push(self, key, value, time_stamp):
        logger.debug('push key=%s, value=%s, ts=%s', key, value, time_stamp)
        if key not in parameters:
            logger.info('error key: %s', key)
            return 'error key: %s' % key
        if len(value) != parameters[key].size:
            logger.info('size unmatched: %s%s', key, parameters[key].shape)
            return 'size unmatched: %s%s' % (key, parameters[key].shape)
        gradients[key].append(value)
        return 'success'

    def pull(self, key, value, time_stamp):
        logger.debug('push key=%s, value=%s, ts=%s', key, value, time_stamp)
        return 1

    def init(self, key_types_json):
        init_parameters(json.loads(key_types_json))


def init_parameters(key_types):
    logger.info('initializing parameters')
    for key_type in key_types:
        logger.debug(key_type)
        parameters[key_type[0]] = np.random.random(key_type[1])
        gradients[key_type[0]] = []


def update_parameters():
    mutex.acquire()
    for k in parameters:
        tmp = np.zeros(parameters[k].shape)
        for grad in gradients[k]:
            tmp += grad
        parameters[k] -= learning_rate * tmp
        gradients[k].clear()


def run():
    # 创建服务端
    handler = ParameterServerHandler()
    processor = ParameterServer.Processor(handler)

    # 监听端口
    transport = TSocket.TServerSocket(HOST, PORT)

    # 选择传输层
    tfactory = TTransport.TBufferedTransportFactory()

    # 选择传输协议
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    # 创建服务端
    server = TServer.TThreadPoolServer(processor, transport, tfactory, pfactory)

    logger.info('ParameterServer listening %s:%s', HOST, PORT)
    server.serve()


if __name__ == '__main__':
    t = threading.Thread(target=run)
    t.start()
    # run()
    init_parameters([('fc1', (1, 2))])
    IPython.embed()
    t.join()

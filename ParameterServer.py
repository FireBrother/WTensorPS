import json
import threading
import configparser
import logging
from collections import namedtuple

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
UPDATE_THRESH = config['DEFAULT'].getint('UPDATE_THRESH')

# 可能随时间变化
learning_rate = LEARNING_RATE
# 管理当前接入的workers，workers的内容应该包含id, status和clock
workers = {}
max_worker_id = 0
# 当前服务器的参数版本
server_iteration_count = 0
# 出现了几个更新
update_count = 0

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
logger.setLevel(logging.DEBUG)

mutex = threading.Lock()
parameters = {}
gradients = {}


class ParameterServerHandler:
    def __init__(self):
        self.init_mutex = threading.Lock()
        self.register_worker_mutex = threading.Lock()

    def push(self, wid, key, value_json, time_stamp):
        logger.debug('push key=%s, value=%s, ts=%s', key, value_json, time_stamp)
        value = np.array(json.loads(value_json))
        global update_count, UPDATE_THRESH
        if key not in parameters:
            logger.info('error key: %s', key)
            return 'error key: %s' % key
        if value.shape != parameters[key].shape:
            logger.info('size unmatched: %s%s, should be %s', key, value.shape, parameters[key].shape)
            return 'size unmatched: %s%s' % (key, parameters[key].shape)
        if wid not in workers or workers[wid]['status'] != 'working':
            logger.info('worker#%d is not working: %s', wid, workers[wid]['status'] if wid in workers else 'None')
            return 'worker#%d is not working: %s' % (wid, workers[wid]['status'] if wid in workers else 'None')
        if time_stamp < server_iteration_count:
            logger.info('worker#%d behind iteration: %d(%d)', wid, time_stamp, server_iteration_count)
            return 'worker#%d behind iteration: %d(%d)' % (wid, time_stamp, server_iteration_count)
        gradients[key].append(value)
        update_count += 1
        if update_count >= UPDATE_THRESH:
            update_parameters()
            update_count = 0
        return 'success'

    def pull(self, wid, key, time_stamp):
        logger.debug('pull key=%s, ts=%s', key, time_stamp)
        if key not in parameters:
            logger.info('error key: %s', key)
            return 'error key: %s' % key
        if wid not in workers or workers[wid]['status'] != 'working':
            logger.info('worker#%d is not working: %s', wid, workers[wid]['status'] if wid in workers else 'None')
            return 'worker#%d is not working: %s' % (wid, workers[wid]['status'] if wid in workers else 'None')
        return json.dumps([server_iteration_count, parameters[key].tolist()])

    def init(self, wid, key, init_value_json):
        self.init_mutex.acquire()
        if wid not in workers or workers[wid]['status'] != 'working':
            logger.info('worker#%d is not working: %s', wid, workers[wid]['status'] if wid in workers else 'None')
            ret = 'worker#%d is not working: %s' % (wid, workers[wid]['status'] if wid in workers else 'None')
        elif init_parameters(key, json.loads(init_value_json)):
            ret = 'success'
        else:
            ret = '%d already initialized, abort' % key
        self.init_mutex.release()
        return ret

    def register_worker(self):
        self.register_worker_mutex.acquire()
        global max_worker_id
        w = {'id': max_worker_id, 'status': 'working', 'clock': 0}
        workers[w['id']] = w
        max_worker_id += 1
        self.register_worker_mutex.release()
        return w['id']

    def goodbye(self, wid):
        if wid not in workers or workers[wid]['status'] != 'working':
            logger.info('worker#%d is not working: %s', wid, workers[wid]['status'] if wid in workers else 'None')
            return 'worker#%d is not working: %s' % (wid, workers[wid]['status'] if wid in workers else 'None')
        workers[wid]['status'] = 'finished'
        return 'success'


def init_parameters(key, init_value):
    if key in parameters:
        logger.info('reject reinitializing %s' % key)
        return False
    logger.info('initializing parameters')
    logger.debug('%s=%s', key, init_value)
    parameters[key] = np.array(init_value)
    gradients[key] = []
    return True


def update_parameters():
    global server_iteration_count
    mutex.acquire()
    for k in parameters:
        tmp = np.zeros(parameters[k].shape)
        if len(gradients[k]) == 0:
            continue
        for grad in gradients[k]:
            tmp += grad
        # 采用L2正则化项
        parameters[k] -= learning_rate * (tmp + parameters[k])
        gradients[k].clear()
    server_iteration_count += 1
    logger.info('parameters updated: %d', server_iteration_count)
    mutex.release()


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

    logger.info('ParameterServer listening %s:%s, LEARNING_RATE=%s, UPDATE_THRESH=%d',
                HOST, PORT, LEARNING_RATE, UPDATE_THRESH)
    server.serve()


if __name__ == '__main__':
    run()

    # t = threading.Thread(target=run)
    # t.start()
    # IPython.embed()
    # t.join()

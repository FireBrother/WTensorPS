import IPython
import logging
import json
import numpy as np

from ParameterServer import *
from thrift.transport import TSocket, TTransport
from thrift.protocol import TBinaryProtocol, TProtocol


class WorkerClient:
    def __init__(self, host, port):
        self._transport = TSocket.TSocket(host, port)
        self._transport = TTransport.TBufferedTransport(self._transport)
        protocol = TBinaryProtocol.TBinaryProtocol(self._transport)
        self.ps_client = ParameterServer.Client(protocol)
        self.wid = None
        self.clock = 0
        self.logger = logging.getLogger(__name__)

    # 用于手动打开传输的入口函数
    def open(self):
        self._transport.open()
        self.wid = self.ps_client.register_worker()
        return self.wid

    # 用于手动关闭传输的出口函数
    def close(self):
        self.ps_client.goodbye(self.wid)
        self._transport.close()

    # 用于context manager的入口函数
    def __enter__(self):
        self.open()
        return self

    # 用于context manager的出口函数
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def init(self, key, init_value):
        ret = self.ps_client.init(self.wid, key, json.dumps(init_value.tolist()))
        return ret

    def push(self, key, value):
        ret = self.ps_client.push(self.wid, key, json.dumps(value.tolist()), self.clock)
        return ret

    def pull(self, key):
        ret = self.ps_client.pull(self.wid, key, self.clock)
        ret = json.loads(ret)
        self.clock = ret[0]
        return np.array(ret[1])


if __name__ == '__main__':
    # 手动使用的例子
    wc = WorkerClient('localhost', 9090)
    wc.open()
    wc.close()

    # 使用with语句的例子（不再需要手动open和close）
    with WorkerClient('localhost', 9090) as wc:
        print('init', wc.init('fc1', np.array([[1.0, 2.0]])))
        print('init', wc.init('fc2', np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])))
        print('pull', wc.pull('fc1'))
        print('push', wc.push('fc1', np.array([[0.4, 0.5]])))
        print('pull', wc.pull('fc1'))
        print('push', wc.push('fc2', np.array([[0.3, 0.2, 0.5], [1.3, 1.2, 1.5]])))
        print('pull', wc.pull('fc1'))
        print('pull', wc.pull('fc2'))
        print('push', wc.push('fc1', np.array([[0.3, 0.2]])))
        print('pull', wc.pull('fc1'))
        print('push', wc.push('fc1', np.array([[0.3, 0.2]])))
        print('push', wc.push('fc1', np.array([[0.3, 0.2]])))
        print('push', wc.push('fc1', np.array([[0.3, 0.2]])))
        print('pull', wc.pull('fc1'))

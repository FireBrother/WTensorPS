import IPython
import logging

from ParameterServer import *
from thrift.transport import TSocket, TTransport
from thrift.protocol import TBinaryProtocol, TProtocol

transport = TSocket.TSocket('localhost', 9090)
transport = TTransport.TBufferedTransport(transport)
protocol = TBinaryProtocol.TBinaryProtocol(transport)
client = ParameterServer.Client(protocol)
transport.open()
IPython.embed()
transport.close()

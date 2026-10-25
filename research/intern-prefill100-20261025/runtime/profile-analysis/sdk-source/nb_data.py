import copy
from collections import OrderedDict
from struct import Struct
from tools.device_profiler.nb_data_item import DataItem
from tools.device_profiler.utils import load_bin

__all__ = ['NBNode', 'NBProfileData']

class NBNodeCore(object):
    def __init__(self, id):
        self.__id = id
        self.__param = None

    def __str__(self):
        return "NB Node Core {}".format(str(self.id))

    @property
    def id(self):
        return self.__id

    def add_data_item(self, item):
        if self.__param is None:
            self.__param = copy.deepcopy(item.param)
        else:
            for key, val in self.__param.items():
                if key in ['NodeId', 'CoreID', 'NN_LAYER_ID']:
                    continue
                if key in item.param:
                    self.__param[key] += item.param[key]

    def get_cycle_count(self):
        return self.__param['NN_TOTAL_BUSY_CYCLE']

    def get_read_bw(self):
        return self.__param['NN_TOTAL_READ_VALID_BW_DDR']

    def get_write_bw(self):
        return self.__param['NN_TOTAL_WRITE_VALID_BW_DDR']

    def get_mac(self):
        return self.__param['NN_TOTAL_MAC_COUNT']

class NBNode(object):
    def __init__(self, id, max_core_count = 1):
        self.__id = id
        self.__max_core_count = max_core_count
        self.__core = OrderedDict()

    def __str__(self):
        return "NB Node {}".format(str(self.id))

    @property
    def id(self):
        return self.__id

    def __add_core_item(self, item):
        core_id = item.core_id
        core = self.get_core(core_id)
        if core is None:
            core = NBNodeCore(core_id)
            self.__core[core_id] = core
        core.add_data_item(item)

    def add_node_item(self, item):
        assert item.node_id == self.__id, "Invalid data id {} != node id {}".format(item.node_id, self.id)
        self.__add_core_item(item)

    def get_core(self, id):
        return self.__core.get(id, None)

    def get_cycle_count(self):
        core_cycle = [0] * self.__max_core_count
        for id, core in self.__core.items():
            cycle = core.get_cycle_count()
            core_cycle[core.id] = cycle
        return core_cycle

    def get_read_bw(self):
        core_read_bw = [0] * self.__max_core_count
        for id, core in self.__core.items():
            bw = core.get_read_bw()
            core_read_bw[core.id] = bw
        return core_read_bw

    def get_write_bw(self):
        core_write_bw = [0] * self.__max_core_count
        for id, core in self.__core.items():
            bw = core.get_write_bw()
            core_write_bw[core.id] = bw
        return core_write_bw

    def get_mac(self):
        core_mac = [0] * self.__max_core_count
        for id, core in self.__core.items():
            mac = core.get_mac()
            core_mac[core.id] = mac
        return core_mac

class NBDataHeader(object):
    def __init__(self, buf):
        self.__struct = Struct('@4sIIII236s')
        self.__buf = buf
        self.__parse()

    @property
    def net_count(self):
        return self.__net_count

    @property
    def core_count(self):
        return self.__core_count

    @property
    def target(self):
        return hex(self.__target)

    def __parse(self):
        values = self.__struct.unpack(self.__buf)
        self.__magic = values[0].decode('utf-8')
        self.__version = values[1]
        self.__target = values[2] # decimal
        self.__net_count = values[3]
        self.__core_count = values[4]

class NBNetInfo(object):
    def __init__(self, headbuf, max_core_count = 1):
        self.__struct = Struct('@64sIIIII44s') # network header fmt
        self.__headbuf = headbuf
        self.__max_core_count = max_core_count
        self.__item_list = list()
        self.__nb_nodes = OrderedDict()

        self.__header_parse()

    @property
    def data_offset(self):
        return self.__data_offset

    @property
    def data_size(self):
        return self.__data_size

    @property
    def item_list(self):
        return self.__item_list

    @property
    def nb_nodes(self):
        return self.__nb_nodes

    def __add_item(self, item):
        self.__item_list.append(item)

    def __get_nb_node(self, id):
        return self.__nb_nodes.get(id, None)

    def parse_items(self, items_buf):
        start = 0
        end = self.__profile_item_size
        for i in range(self.__profile_item_count):
            item = DataItem(items_buf[start:end])
            self.__add_item(item)

            start = end
            end += self.__profile_item_size

    def build_nb_nodes(self):
        def add_node_item(item):
            node_id = item.node_id
            node = self.__get_nb_node(node_id)
            if node is None:
                node = NBNode(node_id, self.__max_core_count)
                self.__nb_nodes[node_id] = node
            node.add_node_item(item)

        for item in self.__item_list:
            add_node_item(item)

    def __header_parse(self):
        values = self.__struct.unpack(self.__headbuf)
        self.__network_name = values[0].decode('utf-8')
        self.__data_offset = values[1]
        self.__data_size = values[2]
        self.__profile_item_count = values[3]
        self.__profile_item_size = values[4]
        self.__core_count = values[5]

    def is_in_nb_net(self, id):
        node = self.__get_nb_node(id)
        return (node is not None)

    def get_cycle_count(self, id):
        node = self.__nb_nodes[id]
        cycle = node.get_cycle_count()
        return cycle

    def get_read_bw(self, id):
        node = self.__nb_nodes[id]
        read_bw = node.get_read_bw()
        return read_bw

    def get_write_bw(self, id):
        node = self.__nb_nodes[id]
        write_bw = node.get_write_bw()
        return write_bw

    def get_mac(self, id):
        node = self.__nb_nodes[id]
        mac = node.get_mac()
        return mac

class NBProfileData(object):
    def __init__(self, nb_data):
        self.__buffer = load_bin(nb_data)
        self.__header = None
        self.__net_list = list()

        self.__parse_data()
        self.__build_nb_net()

    @property
    def buffer(self):
        return self.__buffer

    def get_profile_net(self, id=0):
        return self.__net_list[id]

    def __parse_data(self):
        head_len = 256
        net_head_len = 128
        # Parse header
        self.__header = NBDataHeader(self.buffer[0:head_len])

        # Parse network
        start = head_len
        end = head_len + net_head_len
        self.__net_list = list()
        for i in range(self.__header.net_count):
            net = NBNetInfo(self.buffer[start:end], self.__header.core_count)
            net.parse_items(self.buffer[net.data_offset:(net.data_offset + net.data_size)])
            self.__net_list.append(net)

            start = end
            end += net_head_len

    def __build_nb_net(self):
        for net in self.__net_list:
            net.build_nb_nodes()

from collections import OrderedDict
from tools.device_profiler.utils import load_json

class TargetType(object):
    NONE = 0
    SHADER = 1
    NN = 2
    TP = 3
    SW = 4
    SCALER = 5 # scaler
    NBG = 6

    @classmethod
    def get_name(cls, type):
        _map = {
            cls.NONE: "NONE",
            cls.SHADER: "SHADER",
            cls.NN: "NN",
            cls.TP: "TP",
            cls.SW: "SW",
            cls.SCALER: "SCALER",
            cls.NBG: "NBG",
        }
        return _map.get(type, 'NONE')

def build_trace_id(obj):
    uid = obj.get('id')
    sub_id = obj.get('sub_id', '')
    trace_id = str(uid) + '_' + str(sub_id)
    return trace_id

class TraceNode(object):
    def __init__(self, trace_id):
        self.__trace_id = trace_id
        self.__shared_ids = list()
        self.__profile_nodes = list()

    def __str__(self):
        return 'Trace Node ' + self.__trace_id

    @property
    def trace_id(self):
        return self.__trace_id

    @property
    def uid(self):
        return int(self.__trace_id.split('_')[0])

    @property
    def sub_id(self):
        sub_id = self.__trace_id.split('_')[1]
        if sub_id == '':
            return None
        else:
            return int(sub_id)

    @property
    def shared_ids(self):
        return self.__shared_ids

    @property
    def profile_nodes(self):
        return self.__profile_nodes

    @profile_nodes.setter
    def profile_nodes(self, val):
        assert isinstance(val, list)
        self.__profile_nodes = val

    def add_share_node(self, obj):
        share_trace_id = build_trace_id(obj)
        if share_trace_id not in self.__shared_ids:
            self.__shared_ids.append(share_trace_id)

    def get_profile_nodes_id(self):
        nodes_id = list()
        for item in self.__profile_nodes:
            nodes_id.append(item['node_id'])
        return nodes_id

class NodeTraceDB(object):
    def __init__(self, node_trace):
        self.__trace_dict = load_json(node_trace)
        self.__trace_nodes = OrderedDict()

        self.__setup_trace_node()

    @property
    def trace_nodes(self):
        return self.__trace_nodes

    def query_node_target(self, node_id):
        for trace_id, trace_node in self.trace_nodes.items():
            for profile_node in trace_node.profile_nodes:
                if node_id == profile_node['node_id']:
                    return profile_node['target']
        return TargetType.NONE

    def get_trace_node(self, trace_id):
        return self.__trace_nodes.get(trace_id, None)

    def get_trace_nodes_by_uid(self, uid):
        trace_nodes = list()
        for trace_id, trace_node in self.trace_nodes.items():
            if trace_node.uid == uid:
                trace_nodes.append(trace_node)
        return trace_nodes

    def __setup_trace_node(self):
        for t in self.__trace_dict:
            trace_id = build_trace_id(t['uid'])
            trace_node = self.get_trace_node(trace_id)
            if trace_node is None:
                trace_node = TraceNode(trace_id)
                self.__trace_nodes[trace_id] = trace_node
            for s in t['share_node_ids']:
                trace_node.add_share_node(s)
            trace_node.profile_nodes = t['profile_node_ids']